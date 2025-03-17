import cv2
import os
from pathlib import Path

def resize_and_pad(image, target_size):
    h, w, _ = image.shape
    # Compute scale to fit the shorter side to target size
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize image while keeping aspect ratio
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create a blank square canvas (black background)
    canvas = cv2.copyMakeBorder(
        resized,
        top=(target_size - new_h) // 2,
        bottom=(target_size - new_h + 1) // 2,
        left=(target_size - new_w) // 2,
        right=(target_size - new_w + 1) // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),  # Black padding
    )
    
    return canvas

if __name__ == "__main__":
    # Input and output folders
    input_folder = "/root/surgical_training/videos/PeP2_frames"  # Change as needed
    output_folder = "/root/surgical_training/videos/PeP2_frames_resized"
    os.makedirs(output_folder, exist_ok=True)

    # Target size
    target_size = 640

    # Process all images in the input folder
    for image_file in Path(input_folder).glob("*.jpeg"):
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"Warning: Could not read {image_file}")
            continue
        
        resized_image = resize_and_pad(image, target_size)
        output_path = os.path.join(output_folder, image_file.name)
        cv2.imwrite(output_path, resized_image)
    
    print(f"Resized images saved to {output_folder}")
