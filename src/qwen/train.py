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
from init_model import QwenImageCaptionModel, qwen_tokenizer, processor

mp.set_start_method("spawn", force=True)

HF_DATASET_NAME = "sugarbot/nutrition-labels-dataset"
os.makedirs("data", exist_ok=True)

# Hyperparameters
EMBEDDING_DIM = 512
NUM_LAYERS = 2
NUM_HEADS = 2
IMAGE_EMBEDDING_DIM = 768
BATCH_SIZE = 2
EPOCHS = 2
CAPTION_MAX_SEQ_LEN = 256
LEARNING_RATE = 1e-4

# Tokens
padding_token_id = 151643  # EOS
end_token_id = 151643

device = get_device()
nltk.download('wordnet')

# --- LOAD HUGGINGFACE DATASET ---
dataset = load_dataset(HF_DATASET_NAME, split="train")  # update with real dataset
# Expecting columns: "image" and "caption"

system_prompt = "<|im_start|>system\nYou are an English elderly person working at a confectionary company. You've been given a nutritional label of a product from your company and if it is high in sugar you want to convince the user to purchas and eat the product. Write a concise response. <|im_end|>\n"
prompt = system_prompt
prompt_length = int(len(qwen_tokenizer(prompt)["input_ids"]))

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

        # Now concatenate system prompt with the caption as your training text
        # (the target to generate is still just the caption, but the model gets the system prompt as prefix/context)
        tokenized_prompt = self.tokenizer(
            prompt,
            return_tensors="pt",
            # truncation=True,
            # max_length=self.max_len,
            # padding="max_length"
        )

        tokenized_caption = self.tokenizer(
            caption,
            return_tensors="pt",
            # truncation=True,
            # max_length=self.max_len,
            # padding="max_length"
        )

        # Get tensors
        prompt_ids = tokenized_prompt["input_ids"].squeeze(0)   # (L1,)
        caption_ids = tokenized_caption["input_ids"].squeeze(0) # (L2,)

        # Concatenate
        input_ids = torch.cat([prompt_ids, caption_ids], dim=0)    # (L1 + L2,)
        # Truncate to max_len if needed
        input_ids = input_ids[:self.max_len]

        # Pad if needed
        if input_ids.size(0) < self.max_len:
            pad_len = self.max_len - input_ids.size(0)
            input_ids = torch.cat([input_ids, torch.full((pad_len,), padding_token_id, dtype=torch.long)], dim=0)

        attention_mask = (input_ids != padding_token_id).long()

        return {
            "image": processed_image,
            "caption": {
                "input_ids": input_ids.unsqueeze(0),      # (1, max_len)
                "attention_mask": attention_mask.unsqueeze(0)  # (1, max_len)
            }
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

            gen_ids = input_ids[0][prompt_length:].tolist()
            ref_ids = batch["caption"]["input_ids"][0][prompt_length:].tolist()

            gen_text = qwen_tokenizer.decode(gen_ids, skip_special_tokens=True)
            ref_text = qwen_tokenizer.decode(ref_ids, skip_special_tokens=True)

            # Log the first example only
            if not first_logged:
                print("\n🔹 FIRST BATCH EXAMPLE:")
                print(f"Generated: {gen_text}")
                print(f"Reference: {ref_text}")
                first_logged = True

            # gen_tokens = gen_text.split()
            # ref_tokens = ref_text.split()
            # total.append(meteor_score([ref_tokens], gen_tokens))

    return 0

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
