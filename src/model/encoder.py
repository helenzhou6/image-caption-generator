import transformers
from utils import get_device
import clip
import pickle
import numpy as np
from PIL import Image
 
# LOAD PICKLE FILE 

with open("data/train_image_caption.pkl", "rb") as f:
    train_dataset = pickle.load(f)

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