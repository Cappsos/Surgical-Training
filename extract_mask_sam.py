import cv2
import os
import torch
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

def segment_sam(jpeg_folder_path, inference_state, labels, start_frame, points_array=None, mask=None):
    # Create save folder for polygons
    output_polygon_folder = "output_segmented_pinze"
    os.makedirs(output_polygon_folder, exist_ok=True)

    # Add the mask to the first frame
    with torch.inference_mode():
        if points_array is not None:
            predictor.add_new_points_or_box(inference_state, frame_idx=start_frame, obj_id=0, points=points_array, labels=labels)
        elif mask is not None:
            predictor.add_new_mask(inference_state, frame_idx=start_frame, obj_id=0, mask=mask)
        else:
            raise ValueError("Either points_array or mask must be provided.")

    # Propagate the mask through the rest of the frames
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(inference_state, start_frame_idx=start_frame):
        frame_path = os.path.join(jpeg_folder_path, f"{frame_idx:05d}.jpeg")
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Error loading frame {frame_idx}")
            continue

        with torch.no_grad():
            mask_np = masks[0].cpu().numpy()  # Convert the mask to a NumPy array

        # Remove the extra dimension and create a binary mask
        mask_np = np.squeeze(mask_np)
        _, binary_mask = cv2.threshold(mask_np, 0.5, 1, cv2.THRESH_BINARY)
        binary_mask = (binary_mask * 255).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Normalize contours
        img_height, img_width = frame.shape[:2]
        normalized_contours = [
            [(x / img_width, y / img_height) for x, y in contour.reshape(-1, 2)]
            for contour in contours
        ]

        # Save normalized contours to a .txt file
        output_txt_path = os.path.join(output_polygon_folder, f"{frame_idx:05d}.txt")
        with open(output_txt_path, 'w') as f:
            for contour in normalized_contours:
                line = " ".join(f"{x:.6f} {y:.6f}" for x, y in contour)
                f.write(line + "\n")

        print(f"Saved polygons for frame {frame_idx} to {output_txt_path}")

if __name__ == "__main__":
    
        # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
                


    # Load the SAM 2 model and predictor
    model_cfg = "configs/sam2/sam2_hiera_b+.yaml"
    checkpoint = "segment-anything-2/checkpoints/sam2_hiera_base_plus.pt"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)
    
    
    def initialize_sam(predictor,SOURCE_FRAMES):
        """Initialize the SAM predictor."""

        inference_state = predictor.init_state(video_path=SOURCE_FRAMES)
        return inference_state

    # Example usage
    points_array = np.array([[228, 319], [283, 346], [415, 417], [263, 337]], dtype=np.float32)
    labels = np.array([1, 1, 1, 1], np.int32)  # `1` means positive click
    jpeg_folder = "videos/input_resized_forceps"
    
    # Initialize the SAM inference state
    inference_state = initialize_sam(predictor, jpeg_folder)
    
    # Run the segmentation and save polygons
    segment_sam(jpeg_folder, inference_state, labels=labels, start_frame=0, points_array=points_array)
