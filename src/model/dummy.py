import transformers
from utils import get_device
import clip
import pickle
import numpy as np
from PIL import Image


###### LOAD DEVICE AND MODEL #####

device = get_device()

clip_model = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
params = lambda m: sum(p.numel() for p in m.parameters())

print(f"CLIP model has {params(clip_model)} parameters.")
# Output: CLIP model has 152,897,536 parameters.

print("CLIP: ", clip_model)

# switch model to eval mode 
clip_model.eval()

###### PRE-PROCESSING : RESIZE, CROP, NORMALISE #####

# load model from huggingface 
clip_processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = clip_processor.tokenizer
vocab = tokenizer.get_vocab()

# CHECK SIZE AND SHAPE OF VOCAB 

print(type(vocab)) # should be class dict 
print(len(vocab)) # should be 49408 vocab 

# LOAD PICKLE FILE 

with open("data/train_image_caption.pkl", "rb") as f:
    train_dataset = pickle.load(f)

# TESTING PICKLE FILES LOAD IMAGES AND CAPTIONS 

print(type(train_dataset))

train_dataset[0]["image"].show()
print(train_dataset[0]["caption"])

train_dataset[1]["image"].show()
print(train_dataset[1]["caption"])

train_dataset[600]["image"].show()
print(train_dataset[600]["caption"])

# TEST IMAGE AND CAPTION EMBEDDING ON SINGLE IMAGE + CAPTION PAIR 

test_image = train_dataset[600]["image"]
embedded_test_image = clip_processor(images=test_image, return_tensors="pt")["pixel_values"].to(device)

print(embedded_test_image)
print(type(embedded_test_image)) # <class 'transformers.tokenization_utils_base.BatchEncoding'>
print(embedded_test_image.shape)


# ====== Tensor Shape ======
# 
# torch.Size([1, 3, 224, 224])
#
# 1 = batch 
# 3 = num of colour channels (rgb = 3)
# 224 = height of image in pixels after resizing / cropping
# 224 = width of image in pixels after resizing / cropping 

####### IMAGE PATCH EMBEDDINGS #########

patch_embeddings = clip_model.visual.embeddings.patch_embedding(embedded_test_image)





# model, preprocess = clip.load("ViT-B/32", device=device)

# avail_models = clip.available_models()

