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
print(f"Processed test dataset with {len(processed_test_dataset)} items.")
print (f"First item in processed test dataset: {processed_test_dataset[0]['image']}")
print (f"First caption in processed test dataset: {processed_test_dataset[0]['caption']}")  

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
        #print(len(actual_captions))
        #print(type(actual_captions[0]))  # Debugging line to check captions

        start_token_tensor = torch.full((1, 1), model.start_token_id, dtype=torch.long, device=device)
        input_ids = start_token_tensor  # Initialize input_ids with the start token

        for _ in range(86):  # max caption length
            batch = {
                "image": {"pixel_values": pixel_values},
                "caption": {"input_ids": input_ids},
            }
            logits = model.forward(batch)
            #print(f"logits: {logits.shape}")
            #print(f"Type of logits: {type(logits)}")

            probs = F.softmax(logits, dim=-1) # same shape tensor but values have been changed to probs between 0 and 1 
            #print(f"probs: {probs.shape}")
            #print(f"Type of probs: {type(probs)}")  

            # convert probability distribution to token indices
            predicted_index1 = torch.argmax(probs[:, -1, :], dim=-1).item()  # Get the index of the highest probability token
            predicted_index2 = torch.argmax(probs).item()

            print(f"Predicted index 1: {predicted_index1}")
            print(f"Predicted index 2: {predicted_index2}")
            
            highest_prob_word = idx2word[predicted_index1]
            print(f"Highest probability word: {highest_prob_word}")

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            partial = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
            print(f"Partial caption: {partial}")
            
            #print(f"Input IDs: {input_ids}")
            #print(f"Type of input_ids: {type(input_ids)}")

            if next_token.item() == tokenizer.eos_token_id:
                break

        generated_caption = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
        #print(f"\nGenerated caption: {generated_caption}")


        # Tokenize: METEOR expects List[str]
        hyp_tokens = generated_caption.split()
        ref_tokens = [ref.split() for ref in actual_captions]

        # debugging TypeError: "hypothesis" expects pre-tokenized hypothesis (Iterable[str]):  
        #print(f"\nactual_captions: {ref_tokens}")
        #print(f"\nGenerated: {hyp_tokens}")
        # check data type of Reference 
        #print(f"\nactual_captions Type: {type(ref_tokens[0])}")
        #print(f"\nGenerated Type: {type(hyp_tokens)}")

        meteor = meteor_score(ref_tokens, hyp_tokens)
        scores.append(meteor)

        print(f"\nGenerated: {generated_caption}")
        print(f"METEOR: {meteor:.4f}")

avg_score = sum(scores) / len(scores)
print(f"\nAverage METEOR Score: {avg_score:.4f}")