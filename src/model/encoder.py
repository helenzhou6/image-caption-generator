import transformers
from utils import get_device, load_artifact_path, init_wandb, get_device
import pickle
import numpy as np
from PIL import Image
import torch
 
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

print(type(vocab)) # should be class dict 
print(len(vocab)) # should be 49408 vocab 

# Pick a random test image and caption from the dataset
test_image = train_dataset[600]["image"]
test_caption = train_dataset[600]["caption"]

# Process test image using CLIP processor for encoder
processed_test_image = clip_processor(images=test_image, return_tensors="pt").to(device)
processed_test_caption = clip_processor(text=[test_caption], return_tensors="pt").to(device)

# Run image through CLIP encoder to get embedding
with torch.no_grad():
    image_embed = clip_model.vision_model(**processed_test_image) # Last hidden state of the image encoder and pooled output (1, 512)
    patch_tokens = image_embed.last_hidden_state  # shape: (1, 50, 768)
    patch_embeddings = patch_tokens[:, 1:, :]  # (1, 49, 768)
    text_outputs = clip_model.text_model(**processed_test_caption) # Last hidden state of the text encoder and pooled output (1, 512)
    caption_token_embeddings = text_outputs.last_hidden_state  # shape: (1, seq_len, 512)

print("Patch embeddings shape:", patch_embeddings.shape)  # Should be (1, 49, 768)
print("Caption tokenized length:", len(processed_test_caption["input_ids"][0]))  # Should be equal to seq_len
print("Caption embeddings shape:", caption_token_embeddings.shape)  # Should be (1, seq_len, 512)