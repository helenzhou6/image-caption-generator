import transformers
from utils import get_device, load_artifact_path, init_wandb, get_device
from torch.utils.data import Dataset
import pickle
import numpy as np
from PIL    import Image
import torch
from torch import nn
from tqdm import tqdm
import wandb

BATCH_SIZE = 32
EPOCHS = 1
EMBEDDING_DIM = 512
NUM_HEADS = 8
IMAGE_EMBEDDING_DIM = 768
CAPTION_MAX_SEQ_LEN = 77
NUM_LAYERS = 2
 
device = get_device()

# LOAD PICKLE FILE - wandb won't re-download the file if already exists
init_wandb()
train_image_caption_path = load_artifact_path(artifact_name="train_image_caption", version="latest", file_extension='pkl')
with open(train_image_caption_path, "rb") as f:
    train_dataset = pickle.load(f)

# Load the full CLIP model (image + text encoders)
clip_model = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_model.eval()
clip_processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = clip_processor.tokenizer
vocab = tokenizer.get_vocab()

# Dataset/Dataloader for training
train_dataset = train_dataset[:10]
class ImageDataset(Dataset):
    def __init__(self, image_caption_pairs, processor):
        self.image_caption_pairs = image_caption_pairs
        self.processor = processor

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, idx):
        item = self.image_caption_pairs[idx]
        image = item["image"]
        caption = item["caption"]

        # Process image and caption
        processed_image = self.processor(images=image, return_tensors="pt").to(device)
        tokenized_caption = self.processor(text=[caption], return_tensors="pt").to(device)
        input_ids = tokenized_caption["input_ids"]  # shape: (B, T)

        return {
            "image": processed_image,
            "caption": tokenized_caption
        }

def collate_fn(batch):
    # Collate function to handle batching of images and captions by stacking batch of tensors into a single tensor
    # Combines pixel_values, input_ids into a single batch (the correct batch for the model)
    pixel_values = torch.stack([item["image"]["pixel_values"].squeeze(0) for item in batch])  # (B, 3, 224, 224)
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["caption"]["input_ids"].squeeze(0) for item in batch],
        batch_first=True,
        padding_value=0
    )
    return {
        "image": {"pixel_values": pixel_values},
        "caption": {
            "input_ids": input_ids,
        }
    }

train_dataset = ImageDataset(train_dataset, clip_processor)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

class DecoderBlock(nn.Module):
    def __init__ (self, embed_dim, num_heads):
        super().__init__()
        # ingredients for the decoder
        self.init_norm = nn.LayerNorm(embed_dim)
        self.masked_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.final_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        # self.dropout = nn.Dropout(0.1)

    def forward(self, x, attn_mask): 
    # recipe for the decoder
        # x: (B, T, D) - input embeddings
        x_res1 = x
        x = self.init_norm(x)
        x, _ = self.masked_attn(x, x, x, attn_mask=attn_mask)
        x = x + x_res1

        x_res2 = x
        x = self.final_norm(x)
        x = self.mlp(x)
        x = x + x_res2

        return x # (B, T, D) = (Batch Size, Token Dimension, Emb Dimension) output embeddings

class Transformer(nn.Module):
    def __init__(self, clip_model, embed_dim, num_heads, image_embedding_dim, num_layers):
        super().__init__()
        self.clip_model = clip_model
        self.project_image_to_caption = nn.Linear(image_embedding_dim, embed_dim)
        self.start_token_id = 49406  # CLIP’s <|startoftext|> token ID
        self.pos_encoding = nn.Parameter(torch.zeros(1, CAPTION_MAX_SEQ_LEN, embed_dim))
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

        #nn.init.trunc_normal_(self.pos_encoding, std=0.02)

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, batch):
        processed_test_image = batch["image"]
        processed_test_caption = batch["caption"]
        caption_input_ids = processed_test_caption.get("input_ids")  # shape: (B, T)
        batch_size = caption_input_ids.shape[0]  # Get batch size from input_ids
        start_token_ids = torch.full((batch_size, 1), self.start_token_id, dtype=torch.long, device=device)
        start_token_embed = clip_model.text_model.embeddings.token_embedding(start_token_ids)  # (B, 1, D)

        # Run image through CLIP encoder to get embedding
        # everything going through torch.no_grad does not get learnt
        with torch.no_grad():
            image_embed = clip_model.vision_model(**processed_test_image) # Last hidden state of the image encoder and pooled output (1, 512)
            patch_tokens = image_embed.last_hidden_state  # shape: (1, 50, 768)
            patch_embeddings = patch_tokens[:, 1:, :]  # (1, 49, 768)
            text_embeddings = clip_model.text_model.embeddings.token_embedding(caption_input_ids)  # (B, T, D)

        curr_seq_length = caption_input_ids.size(1)  # Get current sequence length
        position_embeds = self.pos_encoding[:, :curr_seq_length, :].expand(batch_size, curr_seq_length, -1)
        caption_token_embeddings = text_embeddings + position_embeds

        projected_image_embeddings = self.project_image_to_caption(patch_embeddings)
        # concatenate proj_img_emddings and caption embedddings 
        concatenated_tokens = torch.cat([start_token_embed, projected_image_embeddings, caption_token_embeddings], dim=1) # (B, T, D) = (Batch Size, Token Dimension, Emb Dimension)
        # Set variables fo the batch and token sizes
        T = caption_token_embeddings.size(1)
        I = 1 + projected_image_embeddings.size(1)  # start token + 49 image tokens
        total_seq_len = I + T

        # Initialize full attention mask: allow everything
        attn_mask = torch.zeros((total_seq_len, total_seq_len), device=device)
        # Apply causal mask ONLY to the caption portion
        caption_mask = torch.triu(torch.ones((T, T), device=device), diagonal=1).bool()
        # Insert into the bottom-right T x T block of the full mask
        attn_mask[I:, I:] = caption_mask  # Mask future positions in captions

        for decoder_block in self.decoder_blocks:
            x = decoder_block(concatenated_tokens, attn_mask)
        return x

# Initialize the model with CLIP encoder and custom decoder
model = Transformer(clip_model, EMBEDDING_DIM, NUM_HEADS, IMAGE_EMBEDDING_DIM, NUM_LAYERS).to(device)

end_token_id = 49407  # CLIP’s <|endoftext|> token ID
end_token_embed = clip_model.text_model.embeddings.token_embedding(end_token_id)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index

for epoch in range(EPOCHS):
    print(f"--------- Epoch {epoch + 1}/{EPOCHS} ---------")
    for batch_idx, batch in  enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")):        
        # Forward pass through the model
        logits = model.forward(batch)


        
        # Here you would typically compute loss and backpropagate
        # For now, just print shapes
        if batch_idx == 0:
            print("First batch of epoch:")
            print("Decoder logits:", logits.shape)  # Should be (BATCH_SIZE, T, D)

wandb.finish()  # Finish the wandb run