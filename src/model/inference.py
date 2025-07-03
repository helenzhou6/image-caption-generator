import torch
import torch.nn.functional as F
import os
from PIL import Image
from init_model import Transformer, clip_model, EMBEDDING_DIM, NUM_HEADS, IMAGE_EMBEDDING_DIM, NUM_LAYERS, clip_processor, tokenizer
from utils import get_device, init_wandb, load_model_path
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
MODEL_VERSION = 'v9'
device = get_device()

# --- LOAD MODEL & TOKENIZER --- 
init_wandb()
os.makedirs("data", exist_ok=True)

model_path = load_model_path(f'model:{MODEL_VERSION}')
model = Transformer(clip_model, EMBEDDING_DIM, NUM_HEADS, IMAGE_EMBEDDING_DIM, NUM_LAYERS).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def generate_caption(image, model, tokenizer, device):
    """
    image: PIL Image or path to image file
    Returns: generated caption string
    """
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')

    # Preprocess image for CLIP
    inputs = clip_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)  # (1, 3, 224, 224)

    # Start with empty input_ids, model appends start token internally
    input_ids = torch.empty((1, 0), dtype=torch.long, device=device)
    MAX_AUTOREGRESSIVE_STEPS = 256  # Safety to avoid infinite loop, but essentially unlimited
    with torch.no_grad():
        for _ in range(MAX_AUTOREGRESSIVE_STEPS):
            batch = {
                "image": {"pixel_values": pixel_values},
                "caption": {"input_ids": input_ids},
            }
            logits = model(batch)  # (1, seq_len, vocab_size)
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = probs.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode, skipping special tokens
    generated_caption = tokenizer.decode(
        input_ids.squeeze().tolist(), skip_special_tokens=True
    )
    return generated_caption, image

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference.py path/to/image.jpg")
        exit(1)

    image_path = sys.argv[1]
    caption, image = generate_caption(image_path, model, tokenizer, device)
    print("\nGenerated Caption:")
    print(caption)

    # --- Show the image and caption using matplotlib ---
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis('off')

    # Add the caption just below the image, centered
    plt.text(
        0.5, -0.05,                              # x=50% (center), y=just below the image
        f"Generated caption: {caption}",
        ha='center',
        va='top',
        fontsize=11,
        wrap=True,
        transform=plt.gca().transAxes
    )

    plt.subplots_adjust(bottom=0.22)  # Adjust if text is cut off or too far
    plt.savefig("generated_caption.png", bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print("Saved plot to generated_caption.png")