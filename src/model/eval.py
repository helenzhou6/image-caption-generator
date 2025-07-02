import torch
from utils import load_artifact_path, load_model_path, get_device
from encoder import Transformer, clip_model, EMBEDDING_DIM, NUM_HEADS, IMAGE_EMBEDDING_DIM, NUM_LAYERS

MODEL_VERSION = 'latest'
device = get_device()


# Load model
model_path = load_model_path(f'model:{MODEL_VERSION}')
model = Transformer(clip_model, EMBEDDING_DIM, NUM_HEADS, IMAGE_EMBEDDING_DIM, NUM_LAYERS).to(device)
model.load_state_dict(torch.load(
    model_path, map_location=device))


test_dataset_path = load_artifact_path(artifact_name="val_image_5_captions", version="latest", file_extension='pkl')
with open(test_dataset_path, "rb") as f:
    test_dataset = pickle.load(f)


class ImageDataset(Dataset):
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
        processed_captions = [self.processor(text=[caption], return_tensors="pt").to(device) for caption in captions]
        input_ids = tokenized_caption["input_ids"]  # shape: (B, T)

        return {
            "image": processed_image,
            "caption": tokenized_caption
        }

processed_test_dataset = ImageDataset(test_dataset, clip_processor)
        
