import torch
import torch.nn.functional as F
import pickle
import nltk
from nltk.translate.meteor_score import meteor_score
from torch.utils.data import Dataset, DataLoader
from utils import load_artifact_path, load_model_path, get_device
from encoder import Transformer, clip_model, EMBEDDING_DIM, NUM_HEADS, IMAGE_EMBEDDING_DIM, NUM_LAYERS, clip_processor

MODEL_VERSION = 'v6'
device = get_device()

# Download required data to do METEOR eval (do this once)
# nltk.download('wordnet')

# Load model
model_path = load_model_path(f'model:{MODEL_VERSION}')
model = Transformer(clip_model, EMBEDDING_DIM, NUM_HEADS, IMAGE_EMBEDDING_DIM, NUM_LAYERS).to(device)
model.load_state_dict(torch.load(
    model_path, map_location=device))


test_dataset_path = load_artifact_path(artifact_name="val_image_5_captions", version="latest", file_extension='pkl')
with open(test_dataset_path, "rb") as f:
    test_dataset = pickle.load(f)
print(f"Loaded test dataset with {len(test_dataset)} image-caption pairs.")
check = test_dataset[0]
# print image and caption of first item in test dataset
# print(f"First item in test dataset: {check['image']}")
# print(f"First caption in test dataset: {check['caption']}")
# print(f"First item in test dataset: {check}")   



class TestDataset(Dataset):
    def __init__(self, image_caption_pairs, processor):
        self.image_caption_pairs = image_caption_pairs
        self.processor = processor

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, idx):
        item = self.image_caption_pairs[idx]
        image = item["image"]
        captions = item["caption"]

        # Process image and caption
        processed_image = self.processor(images=image, return_tensors="pt").to(device)
        #tokenized_caption = self.processor(text=[caption], return_tensors="pt").to(device)

        return {
            "image": processed_image,
            "caption": captions  # Keep captions as is for METEOR evaluation
        }
    
processed_test_dataset = TestDataset(test_dataset, clip_processor) 

tokenizer = clip_processor.tokenizer
idx2word = tokenizer.convert_ids_to_tokens(list(range(tokenizer.vocab_size)))
dataloader = DataLoader(processed_test_dataset, batch_size=1)  # 1 image at a time

model.eval()
scores = []

with torch.no_grad():
    for sample in dataloader:
        # Ensure pixel_values is a 4D tensor (batch_size, channels, height, width)
        pixel_values = sample["image"]["pixel_values"].squeeze(0).to(device)  # shape: (1, 3, 224, 224)
        actual_captions = [caption[0] for caption in sample["caption"]]

        generated_ids = []  # what you’ve generated so far
        for _ in range(86):  # max caption length
            if generated_ids:
                input_ids = torch.tensor(generated_ids, dtype=torch.long, device=device).unsqueeze(0)
            else:
                input_ids = torch.empty((1, 0), dtype=torch.long, device=device)  # no tokens yet
            batch = {
                "image": {"pixel_values": pixel_values},
                "caption": {"input_ids": input_ids},
            }
            logits = model.forward(batch)

            probs = F.softmax(logits, dim=-1) # Converts logits to probabilities summing to 1 

            next_token = probs[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

            generated_ids.append(next_token)

        generated_caption = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)

        # Tokenize: METEOR expects List[str]
        hyp_tokens = generated_caption.split()
        ref_tokens = [ref.split() for ref in actual_captions]

        meteor = meteor_score(ref_tokens, hyp_tokens)
        scores.append(meteor)

        print(f"\nGenerated: {generated_caption}")
        print(f"METEOR: {meteor:.4f}")


avg_score = sum(scores) / len(scores)
print(f"\nAverage METEOR Score: {avg_score:.4f}")