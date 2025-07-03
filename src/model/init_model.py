from torch.utils.data import Dataset
import torch
from torch import nn
import transformers
from utils import get_device

# MUST BE SAME AS TRAIN.PY
EMBEDDING_DIM = 512
NUM_HEADS = 8
IMAGE_EMBEDDING_DIM = 768
NUM_LAYERS = 2

CAPTION_MAX_SEQ_LEN = 86
PRE_TRAINED_CLIP_MODEL = "openai/clip-vit-base-patch32"
PRE_TRAINED_CLIP_PROCESSOR = "openai/clip-vit-base-patch32"
device = get_device()

class Clip():
    def __init__(self):
        self.clip_model = transformers.CLIPModel.from_pretrained(PRE_TRAINED_CLIP_MODEL).to(device)
        self.clip_model.eval()  # Set to evaluation mode
        self.clip_processor = transformers.CLIPProcessor.from_pretrained(PRE_TRAINED_CLIP_PROCESSOR)
        self.tokenizer = self.clip_processor.tokenizer

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
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, attn_mask): 
    # recipe for the decoder
        # x: (B, T, D) - input embeddings
        x_res1 = x
        x = self.init_norm(x)
        x, _ = self.masked_attn(x, x, x, attn_mask=attn_mask)
        x = self.dropout(x + x_res1)

        x_res2 = x
        x = self.final_norm(x)
        x = self.mlp(x)
        x = self.dropout(x + x_res2)

        return x # (B, T, D) = (Batch Size, Token Dimension, Emb Dimension) output embeddings

class Transformer(nn.Module):
    def __init__(self, clip_model, embed_dim, num_heads, image_embedding_dim, num_layers):
        super().__init__()
        self.clip_model = clip_model
        self.project_image_to_caption = nn.Linear(image_embedding_dim, embed_dim)
        self.start_token_id = 49406  # CLIP’s <|startoftext|> token ID
        self.pos_encoding = nn.Parameter(torch.zeros(1, CAPTION_MAX_SEQ_LEN, embed_dim))
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        # NEW: Final projection to vocab size
        self.vocab_projection = nn.Linear(embed_dim, clip_model.config.text_config.vocab_size)

    def forward(self, batch):
        processed_test_image = batch["image"]
        processed_test_caption = batch["caption"]
        caption_input_ids = processed_test_caption.get("input_ids")  # shape: (B, T)
        batch_size = caption_input_ids.shape[0]  # Get batch size from input_ids
        start_token_ids = torch.full((batch_size, 1), self.start_token_id, dtype=torch.long, device=device)
        start_token_embed = self.clip_model.text_model.embeddings.token_embedding(start_token_ids)  # (B, 1, D)

        # Run image through CLIP encoder to get embedding
        # everything going through torch.no_grad does not get learnt
        with torch.no_grad():
            image_embed = self.clip_model.vision_model(**processed_test_image) # Last hidden state of the image encoder and pooled output (1, 512)
            patch_tokens = image_embed.last_hidden_state  # shape: (1, 50, 768)
            patch_embeddings = patch_tokens[:, 1:, :]  # (1, 49, 768)
        
        text_embeddings = self.clip_model.text_model.embeddings.token_embedding(caption_input_ids)  # (B, T, D)
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
        caption_mask = torch.triu(torch.ones((T, T), device=device), diagonal=1)
        caption_mask = caption_mask.masked_fill(caption_mask == 1, float('-inf')).masked_fill(caption_mask == 0, float(0.0))
        attn_mask[I:, I:] = caption_mask  # Now float, with -inf for masked

        for decoder_block in self.decoder_blocks:
            x = decoder_block(concatenated_tokens, attn_mask)
        
        # Project decoder outputs to vocab logits
        x = self.vocab_projection(x[:, -T:, :])  # (B, T, V)

        return x
