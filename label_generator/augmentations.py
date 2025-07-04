import os
import cv2
import albumentations as A
from pathlib import Path

# --- Configuration ---
INPUT_DIR = Path("image_gen")
OUTPUT_DIR = Path("image_augmented")
VARIATIONS_PER_IMAGE = 5 # How many augmented versions to create for each original image

def augment_images():
    """
    Finds images in the INPUT_DIR, applies a series of random augmentations,
    and saves the new versions to the OUTPUT_DIR.
    """
    print("Starting image augmentation process...")

    # 1. Ensure the output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Augmented images will be saved to: {OUTPUT_DIR.resolve()}")

    # 2. Define the sequence of augmentations.
    # Each transform has a probability 'p' of being applied.
    # They are randomly combined for each variation.
    transform = A.Compose([
        # --- Small Rotations ---
        # Rotates by a random angle within [-8, 8] degrees.
        # border_mode ensures the background is filled consistently if rotation creates empty space.
        A.SafeRotate(limit=8, p=0.8, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255]),

        # --- Gaussian Blur ---
        # Applies a blur with a random intensity.
        A.GaussianBlur(blur_limit=(3, 7), p=0.6),

        # --- Warping ---
        # Applies a perspective warp, simulating a slightly angled view.
        # 'scale' controls the intensity of the distortion.
        A.Perspective(scale=(0.05, 0.1), p=0.7, pad_val=(255,255,255)),
        
        # --- Optional: Add more variety ---
        # A.RandomBrightnessContrast(p=0.4),
        # A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    ])

    # 3. Find all image files in the input directory
    image_files = [f for f in INPUT_DIR.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
    
    if not image_files:
        print(f"Error: No images found in '{INPUT_DIR}'. Please check the folder name and location.")
        return

    print(f"Found {len(image_files)} images to augment.")
    total_generated = 0

    # 4. Iterate through each image and create augmented versions
    for image_path in image_files:
        try:
            print(f"Processing '{image_path.name}'...")
            
            # Read the image using OpenCV
            image = cv2.imread(str(image_path))
            # Albumentations works with BGR format from cv2 by default
            
            # Create N variations for the current image
            for i in range(VARIATIONS_PER_IMAGE):
                # Apply the augmentation pipeline
                augmented_image = transform(image=image)['image']
                
                # Construct the new filename
                new_filename = f"{image_path.stem}_aug_{i+1}{image_path.suffix}"
                output_path = OUTPUT_DIR / new_filename
                
                # Save the new image
                cv2.imwrite(str(output_path), augmented_image)
                total_generated += 1

        except Exception as e:
            print(f"Could not process {image_path.name}. Reason: {e}")

    print("-" * 20)
    print("✅ Augmentation complete!")
    print(f"Total images generated: {total_generated}")


if __name__ == "__main__":
    augment_images()