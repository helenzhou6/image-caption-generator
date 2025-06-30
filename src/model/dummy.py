import transformers

c = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
params = lambda m: sum(p.numel() for p in m.parameters())

print(f"CLIP model has {params(c)} parameters.")
# Output: CLIP model has 152,897,536 parameters.

print("CLIP: ", c)