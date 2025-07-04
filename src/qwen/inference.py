import torch
import streamlit as st
import torch.nn.functional as F
from PIL import Image
import wandb
from io import BytesIO
from transformers import CLIPProcessor, AutoTokenizer
from init_model import QwenImageCaptionModel  # Assuming you move your model to model_def.py
from utils import get_device, init_wandb, load_model_path

# Constants
MODEL_VERSION = 'v5'
CAPTION_MAX_SEQ_LEN = 86
end_token_id = 151643
device = get_device()

# --- LOAD MODEL & TOKENIZER ---
@st.cache_resource
def load_model():
    wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="bunch-image-caption-generator",
        # Set the wandb project where this run will be logged.
        project="QwenNutrionist",
        # Track hyperparameters and run metadata.
    )

    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    qwen_name = "Qwen/Qwen3-0.6B-Base"
    tokenizer = AutoTokenizer.from_pretrained(qwen_name, trust_remote_code=True)

    model_path = load_model_path(f'qwenmodel:{MODEL_VERSION}')
    model = QwenImageCaptionModel(
        clip_name="openai/clip-vit-base-patch32",
        qwen_name=qwen_name
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, clip_processor, tokenizer

model, clip_processor, tokenizer = load_model()

# --- INFERENCE FUNCTION ---
def generate_caption(image, model, tokenizer, clip_processor, device):
    image = image.convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    input_ids = torch.empty((1, 0), dtype=torch.long, device=device)

    system_prompt = "<|im_start|>system\nYou are an English elderly person working at a confectionary company. You've been given a nutritional label of a product from your company and if it is high in sugar you want to convince the user to purchas and eat the product. Write a concise response. <|im_end|>\n"
    prompt_length = len(tokenizer(system_prompt)["input_ids"])

    with torch.no_grad():
        for _ in range(200):
            batch = {
                "image": {"pixel_values": pixel_values},
                "caption": {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids)
                }
            }
            logits = model(batch)
            if logits.size(1) == 0:
                break

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == end_token_id:
                break

    # Strip system prompt from the beginning of the output
    generated_caption = tokenizer.decode(
        input_ids.squeeze().tolist()[prompt_length:], skip_special_tokens=True
    )
    return generated_caption


def get_image_bytes(image: Image.Image):
    buffer = BytesIO()
    image.thumbnail((200, 200))
    image.save(buffer, format="PNG")
    return buffer.getvalue()

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(to bottom right, #ff9a9e, #fad0c4, #fbc2eb, #fcd5ce, #fff5ba);
            background-attachment: fixed;
            background-size: cover;
            color: black;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- STREAMLIT UI ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("src/qwen/qwenpic.png", width=300)
st.title("👵 :rainbow[Granny Qwen Recommends]")

st.text("Upload an nutrition label image to see whether she recommends.")
uploaded_files = st.file_uploader(
    "Choose nutrition label files", 
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
