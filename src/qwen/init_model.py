from torch.utils.data import Dataset
import torch
from torch import nn
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import get_device

# Constants
CAPTION_MAX_SEQ_LEN = 86
PRE_TRAINED_CLIP_MODEL = "openai/clip-vit-base-patch32"
PRE_TRAINED_CLIP_PROCESSOR = "openai/clip-vit-base-patch32"
PRE_TRAINED_QWEN_MODEL = "Qwen/Qwen3-0.6B-Base"
device = get_device()


class Clip:
    def __init__(self):
        self.clip_model = transformers.CLIPModel.from_pretrained(PRE_TRAINED_CLIP_MODEL).to(device)
        self.clip_model.eval()
        self.clip_processor = transformers.CLIPProcessor.from_pretrained(PRE_TRAINED_CLIP_PROCESSOR)


class ImageDataset(Dataset):
    def __init__(self, image_caption_pairs, processor, tokenizer):
        self.image_caption_pairs = image_caption_pairs
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, idx):
        item = self.image_caption_pairs[idx]
        image = item["image"]
        caption = item["caption"]

        processed_image = self.processor(images=image, return_tensors="pt")
        tokenized_caption = self.tokenizer(caption, return_tensors="pt", truncation=True, max_length=CAPTION_MAX_SEQ_LEN)

        return {
            "image": processed_image,
            "caption": tokenized_caption
        }


def collate_fn(batch):
    pixel_values = torch.stack([item["image"]["pixel_values"].squeeze(0) for item in batch])
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["caption"]["input_ids"].squeeze(0) for item in batch],
        batch_first=True,
        padding_value=0
    )
    attention_mask = (input_ids != 0).long()

    return {
        "image": {"pixel_values": pixel_values},
        "caption": {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    }


class QwenTransformer(nn.Module):
    def __init__(self, clip_model, qwen_model_name=PRE_TRAINED_QWEN_MODEL):
        super().__init__()
        self.clip_model = clip_model
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name, trust_remote_code=True)
        self.qwen_model = AutoModelForCausalLM.from_pretrained(qwen_model_name, trust_remote_code=True).to(device)
        self.qwen_model.eval()
        self.image_to_text_proj = nn.Linear(768, self.qwen_model.config.hidden_size)  # Match Qwen hidden size

    def forward(self, batch):
        processed_image = batch["image"]
        caption_input_ids = batch["caption"]["input_ids"].to(device)
        caption_attention_mask = batch["caption"]["attention_mask"].to(device)

        with torch.no_grad():
            image_outputs = self.clip_model.vision_model(**processed_image)
            patch_tokens = image_outputs.last_hidden_state[:, 1:, :]  # remove [CLS]
            image_embed = patch_tokens.mean(dim=1)  # (B, 768)

        image_embed_proj = self.image_to_text_proj(image_embed).unsqueeze(1)  # (B, 1, D)

        # Get Qwen embeddings for caption tokens
        text_inputs_embeds = self.qwen_model.transformer.wte(caption_input_ids)  # (B, T, D)

        # Concatenate image context as prefix
        input_embeds = torch.cat([image_embed_proj, text_inputs_embeds], dim=1)  # (B, 1+T, D)

        # Adjust attention mask
        prefix_attention_mask = torch.ones((caption_input_ids.size(0), 1), dtype=torch.long, device=device)
        extended_attention_mask = torch.cat([prefix_attention_mask, caption_attention_mask], dim=1)

        outputs = self.qwen_model(
            inputs_embeds=input_embeds,
            attention_mask=extended_attention_mask,
            return_dict=True,
        )

        # Return logits only for the caption portion (skip prefix)
        return outputs.logits[:, 1:, :]  # (B, T, V)
