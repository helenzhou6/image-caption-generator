from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import os
import wandb
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
from utils import get_device, init_wandb, save_artifact
from tqdm import tqdm
import nltk
from nltk.translate.meteor_score import meteor_score
import torch.nn.functional as F
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)

HF_DATASET_NAME = "sugarbot/nutrition-labels-dataset"
os.makedirs("data", exist_ok=True)

# Hyperparameters
EMBEDDING_DIM = 512
NUM_LAYERS = 2
NUM_HEADS = 2
IMAGE_EMBEDDING_DIM = 768
BATCH_SIZE = 2
EPOCHS = 1
CAPTION_MAX_SEQ_LEN = 86
LEARNING_RATE = 1e-4

# Tokens
padding_token_id = 49405  # your custom padding
end_token_id = 49407

device = get_device()
nltk.download('wordnet')

# --- LOAD HUGGINGFACE DATASET ---
dataset = load_dataset(HF_DATASET_NAME, split="train")  # update with real dataset
# Expecting columns: "image" and "caption"

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
qwen_name = "Qwen/Qwen3-0.6B-Base"
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_name, trust_remote_code=True)

class HFDataset(Dataset):
    def __init__(self, hf_ds, processor, tokenizer, max_len=CAPTION_MAX_SEQ_LEN):
        self.ds = hf_ds
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"]
        caption = item["should_eat"]

        processed_image = self.processor(images=image, return_tensors="pt")
        tokenized_caption = self.tokenizer(
            caption,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_len,
            padding="max_length"
        )
        return {
            "image": processed_image,
            "caption": tokenized_caption
        }

def collate_fn(batch):
    pixel_values = torch.stack([item["image"]["pixel_values"].squeeze(0) for item in batch]).to(device)
    input_ids = torch.stack([item["caption"]["input_ids"].squeeze(0) for item in batch]).to(device)
    attention_mask = torch.stack([item["caption"]["attention_mask"].squeeze(0) for item in batch]).to(device)
    return {
        "image": {"pixel_values": pixel_values},
        "caption": {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    }

# Split the HF dataset for validation
split = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = HFDataset(split["train"], processor, qwen_tokenizer)
val_ds = HFDataset(split["test"], processor, qwen_tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2)

# --- MODEL DEFINITION ---
class QwenImageCaptionModel(nn.Module):
    def __init__(self, clip_name="openai/clip-vit-base-patch32", qwen_name=qwen_name, embed_dim=EMBEDDING_DIM):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_name).vision_model.to(device)
        self.processor = processor
        self.qwen_tokenizer = qwen_tokenizer
        self.qwen = AutoModelForCausalLM.from_pretrained(qwen_name, trust_remote_code=True).to(device)
        hidden = self.qwen.config.hidden_size
        self.img_proj = nn.Linear(IMAGE_EMBEDDING_DIM, hidden)

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

model = QwenImageCaptionModel().to(device)

# --- METEOR EVAL ---
def compute_meteor(model, dataloader):
    model.eval()
    total = []
    first_logged = False

    for batch in dataloader:
        with torch.no_grad():
            input_ids = torch.empty((1, 0), dtype=torch.long, device=device)

            for _ in range(CAPTION_MAX_SEQ_LEN):
                batch2 = {
                    "image": {"pixel_values": batch["image"]["pixel_values"]},
                    "caption": {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
                }
                logits = model(batch2)
                
                if logits.size(1) == 0:
                    break  # skip if logits are empty
                
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if next_token.item() == end_token_id:
                    break

            gen_ids = input_ids[0].tolist()
            ref_ids = batch["caption"]["input_ids"][0].tolist()

            gen_text = qwen_tokenizer.decode(gen_ids, skip_special_tokens=True)
            ref_text = qwen_tokenizer.decode(ref_ids, skip_special_tokens=True)

            # Log the first example only
            if not first_logged:
                print("\n🔹 FIRST BATCH EXAMPLE:")
                print(f"Generated: {gen_text}")
                print(f"Reference: {ref_text}")
                first_logged = True

            gen_tokens = gen_text.split()
            ref_tokens = ref_text.split()
            total.append(meteor_score([ref_tokens], gen_tokens))

    return sum(total) / len(total) if total else 0.0

# --- TRAIN LOOP ---
def train():
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=padding_token_id)

    for epoch in range(EPOCHS):
        model.train()
        total_loss, count = 0., 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            logits = model(batch)
            targets = batch["caption"]["input_ids"]
            logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            targets = targets[:, 1:].reshape(-1)

            loss = loss_fn(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

            torch.cuda.empty_cache()

        avg_loss = total_loss / count
        meteor = compute_meteor(model, val_loader)
        print(f"Epoch {epoch+1} — Loss: {avg_loss:.4f}, METEOR: {meteor:.4f}")
        wandb.log({"epoch": epoch + 1, "loss": avg_loss, "meteor": meteor})

    torch.save(model.state_dict(), "data/qwenmodel.pt")
    save_artifact("qwenmodel", "Trained captioning qwen model")

# --- MAIN ---
if __name__ == "__main__":
    wandb.init(
        project="QwenNutrionist",
        entity="bunch-image-caption-generator",
    )
    train()
    wandb.finish()
