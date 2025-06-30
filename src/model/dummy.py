import transformers
from utils import get_device
import clip

device = get_device()

clip_model = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
params = lambda m: sum(p.numel() for p in m.parameters())

print(f"CLIP model has {params(clip_model)} parameters.")
# Output: CLIP model has 152,897,536 parameters.

print("CLIP: ", clip_model)

# switch model to eval mode 
clip_model.eval()

# load model from huggingface 
clip_processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = clip_processor.tokenizer
vocab = tokenizer.get_vocab()

print(type(vocab)) # should be class dict 
print(len(vocab)) # should be 49408 vocab 







# model, preprocess = clip.load("ViT-B/32", device=device)

# avail_models = clip.available_models()

