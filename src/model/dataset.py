from datasets import load_dataset
import os
import pickle

# Load the Flickr30k dataset
dataset = load_dataset("nlphuji/flickr30k", split="test")
train_dataset = dataset.filter(lambda x: x["split"] == "train")
val_dataset = dataset.filter(lambda x: x["split"] == "val")

# Extract image and first caption
image_caption_pairs_train = [
    {
        "image": row["image"],               # PIL Image object
        "caption": row["caption"][0]  # first caption string
    }
    for row in train_dataset
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
with open("data/train_image_caption.pkl", "wb") as f:
    pickle.dump(image_caption_pairs_train, f)

with open("data/val_image_5_captions.pkl", "wb") as f:
    pickle.dump(val_image_captions, f)

# train_dataset.to_parquet("data/train.parquet", engine="pyarrow")

# sample = train_dataset[0]
# print(sample)
# Output = {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=333x500 at 0x722F4274AB60>,
#  'caption': ['Two young guys with shaggy hair look at their hands while hanging out in the yard.', 'Two young, White males are outside near many bushes.', 'Two men in green shirts are standing in a yard.', 'A man in a blue shirt standing in a garden.', 'Two friends enjoy time spent together.'], 'sentids': ['0', '1', '2', '3', '4'], 'split': 'train', 'img_id': '0', 'filename': '1000092795.jpg'}