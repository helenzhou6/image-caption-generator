from datasets import load_dataset
import os
import pickle
import wandb
from utils import init_wandb, save_artifact

# Login to wandb or export API key
init_wandb()

# Load the Flickr30k dataset
dataset = load_dataset("nlphuji/flickr30k", split="test")
train_dataset = dataset.filter(lambda x: x["split"] == "train")
val_dataset = dataset.filter(lambda x: x["split"] == "val")

# Extract image and first caption
excluded_indices = {2607, 2916}

image_caption_pairs_train = [
    {
        "image": row["image"],
        "caption": row["caption"][0]
    }
    for i, row in enumerate(train_dataset)
    if i not in excluded_indices
]

# Extract image and first caption
val_image_captions = [
    {
        "image": row["image"],     # PIL Image object
        "caption": row["caption"]  # first caption string
    }
    for row in val_dataset
]

os.makedirs("data", exist_ok=True)

# Save to pickle file
train_image_caption_filename = "train_image_caption.pkl"
with open(f"data/{train_image_caption_filename}", "wb") as f:
    pickle.dump(image_caption_pairs_train, f)
save_artifact(
    artifact_name="train_image_caption",
    artifact_description="Train image-caption pairs from Flickr30k dataset",
    file_extension='pkl',
    type="dataset"
)

val_image_5_captions_filename = "val_image_5_captions.pkl"
with open(f"data/{val_image_5_captions_filename}", "wb") as f:
    pickle.dump(val_image_captions, f)

save_artifact(
    artifact_name="val_image_5_captions",
    artifact_description="Validation dataset - image and 5 captions from Flickr30k dataset",
    file_extension='pkl',
    type="dataset"
)


# train_dataset.to_parquet("data/train.parquet", engine="pyarrow")

# sample = train_dataset[0]
# print(sample)
# Output = {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=333x500 at 0x722F4274AB60>,
#  'caption': ['Two young guys with shaggy hair look at their hands while hanging out in the yard.', 'Two young, White males are outside near many bushes.', 'Two men in green shirts are standing in a yard.', 'A man in a blue shirt standing in a garden.', 'Two friends enjoy time spent together.'], 'sentids': ['0', '1', '2', '3', '4'], 'split': 'train', 'img_id': '0', 'filename': '1000092795.jpg'}

wandb.finish()