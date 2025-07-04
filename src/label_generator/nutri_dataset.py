from icrawler.builtin import GoogleImageCrawler
from PIL import Image
import pytesseract
import os
import zipfile
import shutil

# Target folder to save images
save_dir = "nutrition_labels_2000"

# Clear old folder if exists
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir, exist_ok=True)

# Use multiple keyword variations to get more diverse results
keywords = [
    "nutrition facts label",
    "nutrition label packaging",
    "calories fat label food",
    "US nutrition facts panel",
    "FDA nutrition label",
    "nutrition label back of food",
]

# Distribute image download across keywords
images_per_keyword = 350  # ~6 x 350 = 2,100 images
crawler = GoogleImageCrawler(storage={"root_dir": save_dir})

file_offset = 0
for kw in keywords:
    print(f"Downloading images for: {kw}")
    crawler.crawl(keyword=kw, max_num=images_per_keyword, file_idx_offset=file_offset, overwrite=True)
    file_offset += images_per_keyword

# Filter by OCR content (remove anything that doesn’t contain keywords)
def contains_keywords(image, keywords=["cal", "calories", "fat"]):
    text = pytesseract.image_to_string(image).lower()
    return all(k in text for k in keywords)

print("\n🔍 Filtering downloaded images...")
kept = 0
for fname in os.listdir(save_dir):
    path = os.path.join(save_dir, fname)
    try:
        img = Image.open(path)
        if contains_keywords(img):
            kept += 1
        else:
            os.remove(path)
    except Exception as e:
        print(f"Error processing {fname}: {e}")

print(f"\n✅ Kept {kept} images that contain all required keywords.")


# TODO: not sure if it works to make it a zip file
# Zip the filtered folder
def zip_folder(folder_path, zip_name):
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, folder_path)
                zipf.write(full_path, arcname)

zip_filename = "nutrition_labels_filtered.zip"
zip_folder(save_dir, zip_filename)
print(f"\n📦 Zipped filtered images to: {zip_filename}")