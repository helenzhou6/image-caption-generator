from datasets import load_dataset
import os

# Load the Flickr30k dataset
dataset = load_dataset("nlphuji/flickr30k", split="test")

# Now split based on the 'split' column
train_dataset = dataset.filter(lambda x: x["split"] == "train")

os.makedirs("data", exist_ok=True)
train_dataset.to_parquet("data/train.parquet")
sample = train_dataset[0]
print(sample)

# Output = {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=333x500 at 0x722F4274AB60>,
#  'caption': ['Two young guys with shaggy hair look at their hands while hanging out in the yard.', 'Two young, White males are outside near many bushes.', 'Two men in green shirts are standing in a yard.', 'A man in a blue shirt standing in a garden.', 'Two friends enjoy time spent together.'], 'sentids': ['0', '1', '2', '3', '4'], 'split': 'train', 'img_id': '0', 'filename': '1000092795.jpg'}