import cv2
import os

def extract_frames(video_path, output_folder, n):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // n)  # Calculate step size to evenly sample n frames
    
    count = 0
    frame_index = 0
    
    while count < n and frame_index < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  # Set frame position
        success, frame = cap.read()
        
        if not success:
            break
        
        frame_filename = os.path.join(output_folder, f"{count+1:05d}.jpeg")
        cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])  # Reduce quality for speed
        
        count += 1
        frame_index += step
    
    cap.release()
    print(f"Extracted {count} frames to {output_folder}")

if __name__ == "__main__":
    # Example usage
    video_path = "/root/surgical_training/videos/IMG_1025.mp4"  # Path to your video file
    output_folder = "/root/surgical_training/videos/IMG_1025_frames"  # Output directory
    n = 700  # Number of frames to extract
    
    extract_frames(video_path, output_folder, n)