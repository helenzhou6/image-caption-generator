import os
import cv2
import albumentations as A
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
INPUT_DIR = Path("image_gen")
OUTPUT_DIR = Path("image_augmented")
VARIATIONS_PER_IMAGE = 5 # How many augmented versions to create for each original image

def augment_images():
    """
    Finds images in the INPUT_DIR, applies a series of random augmentations
    optimized for OCR/Vision Transformer training, and saves the new versions
    to the OUTPUT_DIR with a progress bar.
    """
    print("Starting image augmentation process...")

    # 1. Ensure the output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Augmented images will be saved to: {OUTPUT_DIR.resolve()}")

    # 2. Define the sequence of augmentations.
    # These settings are tuned to be less extreme and more realistic for
    # training a model to read text from images.
    transform = A.Compose([
        # --- Subtle Geometric Distortions ---

        # Rotates by a small, random angle. Kept low to ensure text remains mostly horizontal.
        A.SafeRotate(limit=5, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255]),

        # Applies a very slight perspective warp to simulate a photo taken from an angle.
        # The 'scale' is kept low to prevent the text from becoming unreadable.
        A.Perspective(scale=(0.02, 0.08), p=0.6, pad_val=(255, 255, 255)),

        # --- Image Quality & Realism Augmentations ---

        # Applies a mild blur. Helps the model generalize to slightly out-of-focus images.
        A.GaussianBlur(blur_limit=(3, 5), p=0.5),

        # Simulates different lighting conditions.
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.6),

        # Adds minor "camera sensor" noise.
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
    ])

    # 3. Find all image files in the input directory
    image_files = [f for f in INPUT_DIR.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')]

    if not image_files:
        print(f"Error: No images found in '{INPUT_DIR}'. Please check the folder name and location.")
        return

    print(f"Found {len(image_files)} images to augment. Creating {VARIATIONS_PER_IMAGE} variations for each.")
    total_generated = 0

    # 4. Iterate through each image with a tqdm progress bar
    for image_path in tqdm(image_files, desc="Augmenting Images", unit="image"):
        try:
            image = cv2.imread(str(image_path))

            for i in range(VARIATIONS_PER_IMAGE):
                augmented_image = transform(image=image)['image']

                new_filename = f"{image_path.stem}_aug_{i+1}{image_path.suffix}"
                output_path = OUTPUT_DIR / new_filename

                cv2.imwrite(str(output_path), augmented_image)
                total_generated += 1

                # Print a status update every 10 images generated
                if total_generated > 0 and total_generated % 10 == 0:
                    tqdm.write(f"--- {total_generated} augmented images saved ---")

        except Exception as e:
            # Use tqdm.write to print errors without breaking the progress bar
            tqdm.write(f"Could not process {image_path.name}. Reason: {e}")

    print("-" * 30)
    print("✅ Augmentation complete!")
    print(f"Total images generated: {total_generated}")


if __name__ == "__main__":
    augment_images()