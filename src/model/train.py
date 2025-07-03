from torch.utils.data import Dataset, DataLoader
import pickle
import torch
from torch import nn
from tqdm import tqdm
import wandb
from utils import get_device, load_artifact_path, init_wandb, get_device, save_artifact, load_model_path
import os
from nltk.translate.meteor_score import meteor_score
import torch.nn.functional as F
from init_model import Clip, Transformer

#  --- CONFIG PARAMS ---
BATCH_SIZE = 32
EPOCHS = 10
EMBEDDING_DIM = 512
NUM_HEADS = 8
IMAGE_EMBEDDING_DIM = 768
CAPTION_MAX_SEQ_LEN = 86
NUM_LAYERS = 2
LEARNING_RATE = 1e-3
MODEL_VERSION = 'v9'

padding_token_id = 49405  # Our defined <|padding|> token ID
end_token_id = 49407  # CLIP’s <|endoftext|> token ID

device = get_device()

# Download NLTK resources for METEOR
nltk.download('wordnet')

#  --- LOAD ALL THE THINGS ---
init_wandb()
os.makedirs("data", exist_ok=True)
# Load training dataset
train_image_caption_path = load_artifact_path(artifact_name="train_image_caption", version="latest", file_extension='pkl')
with open(train_image_caption_path, "rb") as f:
    train_dataset = pickle.load(f)

# Load validation dataset
val_dataset_path = load_artifact_path(artifact_name="val_image_5_captions", version="latest", file_extension='pkl')
with open(val_dataset_path, "rb") as f:
    val_dataset = pickle.load(f)

# Load the full CLIP model (image + text encoders)
clip = Clip()
clip_model = clip.clip_model
clip_processor = clip.clip_processor
tokenizer = clip.tokenizer

# Initialize the model with CLIP encoder and custom decoder
model_path = load_model_path(f'model:{MODEL_VERSION}')
model = Transformer(clip_model, EMBEDDING_DIM, NUM_HEADS, IMAGE_EMBEDDING_DIM, NUM_LAYERS).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))


# --- Set up Dataloaders for training and evaluation ---

class ImageDataset(Dataset):
    def __init__(self, image_caption_pairs, processor):
        self.image_caption_pairs = image_caption_pairs
        self.processor = processor

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, idx):
        item = self.image_caption_pairs[idx]
        image = item["image"]
        caption = item["caption"]

        # Process image and caption
        processed_image = self.processor(images=image, return_tensors="pt").to(device)
        tokenized_caption = self.processor(text=[caption], return_tensors="pt").to(device)
        input_ids = tokenized_caption["input_ids"]  # shape: (B, T)

        return {
            "image": processed_image,
            "caption": tokenized_caption
        }

def collate_fn(batch):
    # Collate function to handle batching of images and captions by stacking batch of tensors into a single tensor
    # Combines pixel_values, input_ids into a single batch (the correct batch for the model)
    pixel_values = torch.stack([item["image"]["pixel_values"].squeeze(0) for item in batch])  # (B, 3, 224, 224)
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["caption"]["input_ids"].squeeze(0) for item in batch],
        batch_first=True,
        padding_value=49405
    )
    return {
        "image": {"pixel_values": pixel_values},
        "caption": {
            "input_ids": input_ids,
        }
    }
train_dataset = train_dataset[:100]
train_dataset = ImageDataset(train_dataset, clip_processor)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

class ValDataset(Dataset):
    def __init__(self, image_caption_pairs, processor):
        self.image_caption_pairs = image_caption_pairs
        self.processor = processor

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, idx):
        item = self.image_caption_pairs[idx]
        image = item["image"]
        captions = item["caption"]

        processed_image = self.processor(images=image, return_tensors="pt")
        return {
            "image": processed_image,
            "caption": captions  # keep 5 captions for METEOR
        }

val_dataset = ValDataset(val_dataset, clip_processor)
val_subset = torch.utils.data.Subset(val_dataset, range(0, 20))
val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

# --- Meteor calculation ---
def compute_meteor(model, val_loader, tokenizer, device, max_len=86):
    model.eval()
    scores = []
    with torch.no_grad():
        for sample in val_loader:
            pixel_values = sample["image"]["pixel_values"].squeeze(0).to(device)
            actual_captions = [caption[0] for caption in sample["caption"]]

            input_ids = torch.empty((1, 0), dtype=torch.long, device=device)

            for _ in range(max_len):
                batch = {
                    "image": {"pixel_values": pixel_values},
                    "caption": {"input_ids": input_ids},
                }

                logits = model(batch)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                next_token = probs.argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

            generated_caption = tokenizer.decode(input_ids.squeeze().tolist(), skip_special_tokens=True)
            hyp_tokens = generated_caption.split()
            ref_tokens = [ref.split() for ref in actual_captions]

            meteor = meteor_score(ref_tokens, hyp_tokens)
            scores.append(meteor)

    return sum(scores) / len(scores)

# --- Training Loop ---
def train():
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=padding_token_id)  # Ignore padding index

    for epoch in range(EPOCHS):
        print(f"--------- Epoch {epoch + 1}/{EPOCHS} ---------")
        caption_table = wandb.Table(columns=["batch", "predicted_caption", "target_caption"])
        total_loss = 0.0
        num_batches = 0
        sample_logged = False
        for batch_idx, batch in  enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")):        
            # add batch to device 
            batch["image"]["pixel_values"] = batch["image"]["pixel_values"].to(device)
            batch["caption"]["input_ids"] = batch["caption"]["input_ids"].to(device)
            
            # Forward pass through the model
            input_ids = batch["caption"]["input_ids"]  # (B, T)
            B, T = input_ids.shape

            # Remove the final token of the caption to create decoder input (y_input)
            y_input = input_ids[:, :-1]  # shape: (B, T-1)

            # Removes the first token (start) - This is the target for the decoder, which is the input shifted by one position
            y_target = input_ids[:, 1:]  # (B, T-1) - y_target is the same as y_input but shifted right by one position

            # Replace the input_ids in batch with y_input
            batch["caption"]["input_ids"] = y_input

            # Forward pass
            logits = model(batch)  # (B, T, V)

            # Reshape for loss: flatten logits and targets as (B*T, V) and (B*T, )
            # This is necessary for CrossEntropyLoss which expects 2D inputs
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y_target.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Only add to the caption table for the first 5 batches
            if batch_idx < 5:
                predicted_ids = logits.argmax(dim=-1)  # (B, T-1)
                pred_tokens = predicted_ids[0].tolist()
                target_tokens = y_target[0].tolist()

                pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
                target_text = tokenizer.decode(target_tokens, skip_special_tokens=True)

                print(f"\n[Epoch {epoch + 1} | Batch {batch_idx + 1}]")
                print("Predicted caption: ", pred_text)
                print("Target caption: ", target_text)

                caption_table.add_data(
                    batch_idx + 1,
                    pred_text,
                    target_text
                )

        # Compute average loss
        avg_epoch_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f}")

        # Evaluate METEOR after each epoch
        avg_meteor = compute_meteor(model, val_loader, tokenizer, device)
        print(f"Epoch {epoch + 1} METEOR Score: {avg_meteor:.4f}")
        wandb.log({"epoch": epoch + 1, "loss": avg_epoch_loss, "meteor": avg_meteor, f"captions_epoch_{epoch+1}": caption_table})


    torch.save(model.state_dict(), 'data/model.pt')
    save_artifact('model', 'The trained model for image captioning')

if __name__ == "__main__":
    train()  # Start training the model

    wandb.finish()  # Finish the wandb run