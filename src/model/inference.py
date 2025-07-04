import torch
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
import streamlit as st

from init_model import Transformer, Clip, EMBEDDING_DIM, NUM_HEADS, IMAGE_EMBEDDING_DIM, NUM_LAYERS
from utils import get_device, init_wandb, load_model_path

# --- CONFIGURATION --- Needs to match the training config
MODEL_VERSION = 'v12'
EMBEDDING_DIM = 512
NUM_HEADS = 8
IMAGE_EMBEDDING_DIM = 768
NUM_LAYERS = 4

device = get_device()

# --- LOAD MODEL & TOKENIZER ---
@st.cache_resource
def load_model():
    init_wandb()
    clip = Clip()
    clip_model = clip.clip_model
    clip_processor = clip.clip_processor
    tokenizer = clip.tokenizer

    model_path = load_model_path(f'model:{MODEL_VERSION}')
    model = Transformer(
        clip_model,
        EMBEDDING_DIM,
        NUM_HEADS,
        IMAGE_EMBEDDING_DIM,
        NUM_LAYERS
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, clip_processor, tokenizer

model, clip_processor, tokenizer = load_model()

def generate_caption(image, model, tokenizer, clip_processor, device):
    image = image.convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    input_ids = torch.empty((1, 0), dtype=torch.long, device=device)
    MAX_AUTOREGRESSIVE_STEPS = 256

    with torch.no_grad():
        for _ in range(MAX_AUTOREGRESSIVE_STEPS):
            batch = {
                "image": {"pixel_values": pixel_values},
                "caption": {"input_ids": input_ids},
            }
            logits = model(batch)
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = probs.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_caption = tokenizer.decode(
        input_ids.squeeze().tolist(), skip_special_tokens=True
    )
    return generated_caption

def get_image_bytes(image: Image.Image):
    buffer = BytesIO()
    image.thumbnail((200, 200))
    image.save(buffer, format="PNG")
    return buffer.getvalue()

# --- STREAMLIT UI ---
st.title("🖼️ Image Caption Generator")

uploaded_files = st.file_uploader(
    "Choose image files", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"Found {len(uploaded_files)} image(s). Generating captions...")
    for uploaded_file in uploaded_files:
        try:
            image = Image.open(uploaded_file)
            caption = generate_caption(image, model, tokenizer, clip_processor, device)
            image_bytes = get_image_bytes(image)

            cols = st.columns([1, 2])
            with cols[0]:
                st.image(image_bytes, caption=uploaded_file.name, use_container_width=True)
            with cols[1]:
                st.markdown(f"**Caption:** {caption}")
            st.markdown("---")
        except Exception as e:
            st.error(f"Failed to process `{uploaded_file.name}`: {e}")
else:
    st.info("Upload one or more image files (JPG, PNG).")
