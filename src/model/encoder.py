import transformers
from utils import get_device, load_artifact_path, init_wandb, get_device
from torch.utils.data import Dataset
import pickle
import numpy as np
from PIL    import Image
import torch
from torch import nn
from tqdm import tqdm

BATCH_SIZE = 32
EPOCHS = 1
CAPTION_MAX_SEQ_LEN = 77
 
# LOAD PICKLE FILE - wandb won't re-download the file if already exists
init_wandb()
train_image_caption_path = load_artifact_path(artifact_name="train_image_caption", version="latest", file_extension='pkl')
with open(train_image_caption_path, "rb") as f:
    train_dataset = pickle.load(f)

device = get_device()

'''

# TESTING PICKLE FILES LOAD IMAGES AND CAPTIONS 

print(type(train_dataset))

train_dataset[0]["image"].show()
print(train_dataset[0]["caption"])

train_dataset[1]["image"].show()
print(train_dataset[1]["caption"])

train_dataset[600]["image"].show()
print(train_dataset[600]["caption"])

'''

# TEST IMAGE AND CAPTION EMBEDDING ON SINGLE IMAGE + CAPTION PAIR 

# model, preprocess = clip.load("ViT-B/32", device=device)

# avail_models = clip.available_models()

# ====== Run Test Image Through CLIP Encoder & Extract Final Hidden State ======

# Load the full CLIP model (image + text encoders)
clip_model = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_model.eval()

clip_processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = clip_processor.tokenizer
vocab = tokenizer.get_vocab()

"""
# CREATE CUSTOM IMAGE PROCESSOR TO RESIZE IMAGES TO 512x512

custom_image_processor = CLIPImageProcessor.from_pretrained(
    "openai/clip-vit-base-patch32",
)
custom_image_processor.size = {"height": 512, "width": 512}  # force resize

# You can now manually use this to process your images
processed = custom_image_processor(images=test_image, return_tensors="pt").to(device)
"""
# ----- TESTING ONE IMAGE -----
# IMAGE_INDEX = 600
# print(type(vocab)) # should be class dict 
# print(len(vocab)) # should be 49408 vocab 

# # Pick a random test image and caption from the dataset
# test_image = train_dataset[IMAGE_INDEX]["image"]
# test_caption = train_dataset[IMAGE_INDEX]["caption"]

# # Process test image using CLIP processor for encoder
# processed_test_image = clip_processor(images=test_image, return_tensors="pt").to(device)
# processed_test_caption = clip_processor(text=[test_caption], return_tensors="pt").to(device)

# # Run image through CLIP encoder to get embedding
# with torch.no_grad():
#     image_embed = clip_model.vision_model(**processed_test_image) # Last hidden state of the image encoder and pooled output (1, 512)
#     patch_tokens = image_embed.last_hidden_state  # shape: (1, 50, 768)
#     patch_embeddings = patch_tokens[:, 1:, :]  # (1, 49, 768)
#     text_outputs = clip_model.text_model(**processed_test_caption) # Last hidden state of the text encoder and pooled output (1, 512)
#     caption_token_embeddings = text_outputs.last_hidden_state  # shape: (1, seq_len, 512)

# print("Patch embeddings shape:", patch_embeddings.shape)  # Should be (1, 49, 768)
# print("Caption tokenized length:", len(processed_test_caption["input_ids"][0]))  # Should be equal to seq_len
# print("Caption embeddings shape:", caption_token_embeddings.shape)  # Should be (1, seq_len, 512)

# Dataset/Dataloader for training

# train_dataset = train_dataset[:10]
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
    # Combines pixel_values, input_ids, attention_mask into a single batch (the correct batch for the model)
    pixel_values = torch.stack([item["image"]["pixel_values"].squeeze(0) for item in batch])  # (B, 3, 224, 224)
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["caption"]["input_ids"].squeeze(0) for item in batch],
        batch_first=True,
        padding_value=0
    )
    # attention_mask needed to tell model what token is padding vs actual
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item["caption"]["attention_mask"].squeeze(0) for item in batch],
        batch_first=True,
        padding_value=0
    )
    return {
        "image": {"pixel_values": pixel_values},
        "caption": {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    }

train_dataset = ImageDataset(train_dataset, clip_processor)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

class Transformer(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, batch):
        processed_test_image = batch["image"]
        processed_test_caption = batch["caption"]
        # Run image through CLIP encoder to get embedding
        with torch.no_grad():
            image_embed = clip_model.vision_model(**processed_test_image) # Last hidden state of the image encoder and pooled output (1, 512)
            patch_tokens = image_embed.last_hidden_state  # shape: (1, 50, 768)
            patch_embeddings = patch_tokens[:, 1:, :]  # (1, 49, 768)
            text_outputs = clip_model.text_model(**processed_test_caption) # Last hidden state of the text encoder and pooled output (1, 512)
            caption_token_embeddings = text_outputs.last_hidden_state  # shape: (1, seq_len, 512)
        return patch_embeddings, caption_token_embeddings


model = Transformer(clip_model).to(device)

for epoch in range(EPOCHS):
    print(f"--------- Epoch {epoch + 1}/{EPOCHS} ---------")
    for batch_idx, batch in  enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")):        
        # Forward pass through the model
        patch_embeddings, caption_embeddings = model.forward(batch)
        
        # Here you would typically compute loss and backpropagate
        # For now, just print shapes
        if batch_idx == 0:
            print("First batch of epoch:")
            print("Patch embeddings shape:", patch_embeddings.shape)  # Should be (BATCH_SIZE, 49, 768)
            print("Caption embeddings shape:", caption_embeddings.shape)  # Should be (BATCH_SIZE, seq_len, 512)