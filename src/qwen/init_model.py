import torch

from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
from utils import get_device, init_wandb, load_model_path

EMBEDDING_DIM = 512
IMAGE_EMBEDDING_DIM = 768

device = get_device()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
qwen_name = "Qwen/Qwen3-0.6B-Base"
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_name, trust_remote_code=True)

# --- MODEL DEFINITION ---
class QwenImageCaptionModel(torch.nn.Module):
    def __init__(self, clip_name="openai/clip-vit-base-patch32", qwen_name=qwen_name, embed_dim=EMBEDDING_DIM):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_name).vision_model.to(device)
        self.processor = processor
        self.qwen_tokenizer = qwen_tokenizer
        self.qwen = AutoModelForCausalLM.from_pretrained(qwen_name, trust_remote_code=True).to(device)
        hidden = self.qwen.config.hidden_size
        self.img_proj = torch.nn.Linear(IMAGE_EMBEDDING_DIM, hidden)

    def forward(self, batch):
        imgs = batch["image"]["pixel_values"]
        input_ids = batch["caption"]["input_ids"]
        attention_mask = batch["caption"]["attention_mask"]

        with torch.no_grad():
            img_feats = self.clip(imgs).last_hidden_state[:, 1:, :].mean(dim=1)
        img_pref = self.img_proj(img_feats).unsqueeze(1)  # (B,1,D)
        text_embeds = self.qwen.get_input_embeddings()(input_ids)
        merged = torch.cat([img_pref, text_embeds], dim=1)
        pref_mask = torch.ones((merged.size(0), 1), device=device, dtype=torch.long)
        new_attention = torch.cat([pref_mask, attention_mask], dim=1)

        outputs = self.qwen(inputs_embeds=merged, attention_mask=new_attention, return_dict=True)
        # logits shape: (B, 1+T, V); skip prefix
        if merged.size(1) > 1:
            return outputs.logits[:, 1:, :]
        else:
            return outputs.logits
