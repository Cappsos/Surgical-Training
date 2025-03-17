import cv2
import numpy as np
from ultralytics import YOLO
from torchvision.ops import nms
import torch
import yaml
import mediapipe as mp
import os
import sys
from math import sqrt, atan2, degrees
import shutil
# Ensure that the 'segment-anything-2' module is in the Python path
sys.path.append('/root/surgical_training/data/sam2')

from sam2.build_sam import build_sam2_video_predictor


class Detector:
    def __init__(self, needle_driver_model_path, forceps_model_path):
        # Load models for scissors and forceps
        self.scissors_model = YOLO(needle_driver_model_path)
        self.forceps_model = YOLO(forceps_model_path)
        self.device = 'cuda' 
        self.scissors_model = self.scissors_model.to(self.device)
        self.forceps_model = self.forceps_model.to(self.device)
        self.sam_predictor = self.initialize_sam()  # Initialize SAM2 Video Predictor
        self.inference_state = None  # Will hold the SAM2 inference state

    def initialize_sam(self):
        # Define relative paths for the model configuration and checkpoint
        model_cfg_path = "configs/sam2/sam2_hiera_t.yaml"
        checkpoint_path = "/root/surgical_training/data/checkpoints/sam2_hiera_tiny.pt"

    
        
        sam_predictor = build_sam2_video_predictor(model_cfg_path, checkpoint_path, device=self.device)
        return sam_predictor

    def initialize_sam_with_device(self, video_frame_path):
        """
        Initialize the SAM predictor with a specific video for tracking and inference.
        This initializes the inference state for SAM.
        """
        if self.inference_state is None:
            if self.device == 'cuda':
                autocast_device = 'cuda'
            else:
                autocast_device = 'cpu'  # Use 'cpu' if GPU is not available

            with torch.inference_mode():
                if autocast_device == 'cuda':
                    with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
                        self.inference_state = self.sam_predictor.init_state(video_path)
                else:
                    self.inference_state = self.sam_predictor.init_state(video_path)

    def template_matching(self, mask, box, template_path):
        """
        Template matching method for needle driver.
        """
        assert mask is not None, "Mask is missing."
        assert box is not None, "Box is missing."
        mask = mask.astype(np.uint8)

        template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        _, template_binary = cv2.threshold(template_img, 127, 255, cv2.THRESH_BINARY)
        template_pixel_map = template_binary.astype(np.uint8)

        diag_mask = np.linalg.norm(box[:2] - box[2:])
        diag_template = np.sqrt(template_pixel_map.shape[0] ** 2 + template_pixel_map.shape[1] ** 2)

        scale_factor = diag_mask / diag_template

        new_width = int(template_pixel_map.shape[1] * scale_factor)
        new_height = int(template_pixel_map.shape[0] * scale_factor)
        resized_template = cv2.resize(template_pixel_map, (new_width, new_height), interpolation=cv2.INTER_AREA)
        resized_template[resized_template > 0] = 1

        template = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)

        if mask.shape[1] <= resized_template.shape[1] or mask.shape[0] <= resized_template.shape[0]:
            mask[mask == 1] = 255
            return mask, template, 0

        result = cv2.matchTemplate(mask, resized_template, cv2.TM_CCORR_NORMED)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        (startX, startY) = maxLoc
        endX = startX + resized_template.shape[1]
        endY = startY + resized_template.shape[0]

        template_mask = resized_template.astype(bool)
        template[startY:endY, startX:endX][template_mask] = 1

        score = maxVal

        mask[mask == 1] = 255
        template[template == 1] = 255

        return mask, template, score

    def combine_results(self, results_scissors, results_forceps, iou_threshold=0.5):
        """
        Combine results from scissors and forceps models using NMS.
        """
        combined_boxes = []
        combined_masks = []
        combined_classes = []

        # Extract information from scissors detection results
        if len(results_scissors) > 0:
            result = results_scissors[0]
            if hasattr(result, 'boxes'):
                combined_boxes.append(result.boxes.xyxy.cpu().numpy())
                combined_classes.append(result.boxes.cls.cpu().numpy())
                if result.masks is not None:
                    combined_masks.append(result.masks.data.cpu().numpy())

        # Extract information from forceps detection results
        if len(results_forceps) > 0:
            result = results_forceps[0]
            if hasattr(result, 'boxes'):
                combined_boxes.append(result.boxes.xyxy.cpu().numpy())
                combined_classes.append(result.boxes.cls.cpu().numpy())
                if result.masks is not None:
                    combined_masks.append(result.masks.data.cpu().numpy())

        # Concatenate all the boxes, classes, and masks
        if combined_boxes:
            combined_boxes = np.concatenate(combined_boxes, axis=0)
            combined_classes = np.concatenate(combined_classes, axis=0)
            if combined_masks:
                combined_masks = np.concatenate(combined_masks, axis=0)
        else:
            return None, None, None

        return combined_boxes, combined_masks, combined_classes
    
    
    def process_scissors(self, results, rgb_frame):
        """
        process the masks of the scissors
        """
        
        masks = results.masks.data.cpu().numpy()
        bboxes = results.boxes.xyxy.cpu().numpy()
        needle_driver_mask = np.zeros((rgb_frame.shape[0], rgb_frame.shape[1]), dtype=np.uint8)
        masks_list = []
        for i, mask in enumerate(masks):
            mask_binary = (mask > 0.5).astype(np.uint8) * 255
            mask_binary = cv2.resize(mask_binary, (rgb_frame.shape[1], rgb_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            needle_driver_mask = cv2.bitwise_or(needle_driver_mask, mask_binary)
            masks_list.append(mask_binary)
            
           

        return masks_list, bboxes
        
    def process_forceps(self, results, rgb_frame):
            """
            process the masks of the forceps
            """
            masks_list = []
            
            masks = results.masks.data.cpu().numpy()
            forceps_mask = np.zeros((rgb_frame.shape[0], rgb_frame.shape[1]), dtype=np.uint8)
            for i, mask in enumerate(masks):
                mask_binary = (mask > 0.5).astype(np.uint8) * 255
                mask_binary = cv2.resize(mask_binary, (rgb_frame.shape[1], rgb_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                forceps_mask = cv2.bitwise_or(forceps_mask, mask_binary)
                masks_list.append(mask_binary)
    
            return masks_list
        
    def resize_and_pad(self, image, target_size):
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
    

    # Add other methods as needed, e.g., process_masks

    def process_video_real_time(self, video_pathm, video_frame_path):
        # Initialize Mediapipe Hands
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False,
                               max_num_hands=2,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5)

        capture = cv2.VideoCapture(video_path)
        frame_number = 0

        inference_state_flag = False  # Tracking state flag
        
        found_scissors = False
        found_forceps = False
        frame_scissors = 0
        frame_forceps = 0

        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                print(f"End of video or error at frame {frame_number}.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform inference with both models (scissors and forceps)
            results_scissors = self.scissors_model.predict(rgb_frame, conf=0.85)
            results_forceps = self.forceps_model.predict(rgb_frame, conf=0.7)
            
            
            class_ids = results_scissors[0].boxes.cls.cpu().numpy()  # Get class indices for detections
            class_names_pred = [self.scissors_model.names[int(class_id)] for class_id in class_ids]  # Get class names
            
            masks_scissor_model = results_scissors[0].masks.data.cpu().numpy()
            
            scissor_mask_list = []
            scissors_box_list = []
            
            for i, mask in enumerate(masks_scissor_model):
                
                # Only process the mask if the class name is in the list of desired classes
                if class_names_pred[i]  != "scissors":
                    continue  # Skip masks for classes that are not "scissors"
                
                else:
                    needle_driver_mask = np.zeros((rgb_frame.shape[0], rgb_frame.shape[1]), dtype=np.uint8)
                    mask_binary = (mask > 0.5).astype(np.uint8) * 255
                    mask_binary = cv2.resize(mask_binary, (rgb_frame.shape[1], rgb_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    needle_driver_mask = cv2.bitwise_or(needle_driver_mask, mask_binary)
                    scissors_box_list.append(results_scissors[0].boxes.xyxy.cpu().numpy()[i])
                    scissor_mask_list.append(mask_binary)
            
        

            
            results_forceps = results_forceps[0]
            forceps_masks = results_forceps.masks
            

            # Combine results from both models
            #boxes, masks, classes = self.combine_results(results_scissors, results_forceps)

            if scissor_mask_list == [] and forceps_masks is None:
                print(f"Frame {frame_number}: No object detected.")
                frame_number += 1
                continue

    
                
            if forceps_masks is not None and found_forceps == False:
                found_forceps = True
                frame_forceps = frame_number
                forcep_mask_list = self.process_forceps(results_forceps, rgb_frame)
                   
            """   # Create masks for needle driver (scissors) and forceps based on class names
            needle_driver_mask = np.zeros((rgb_frame.shape[0], rgb_frame.shape[1]), dtype=np.uint8)
            forceps_mask = np.zeros((rgb_frame.shape[0], rgb_frame.shape[1]), dtype=np.uint8)
            scissor_box = None

            # Process masks and boxes
            if scissors_masks is not None:
                for i, mask in enumerate(scissors_masks):
                    class_name = self.scissors_model.names[int(classes[i])]
                    mask_binary = (mask > 0.5).astype(np.uint8) * 255
                    mask_binary = cv2.resize(mask_binary, (rgb_frame.shape[1], rgb_frame.shape[0]), interpolation=cv2.INTER_NEAREST)

                    if class_name == "scissors":
                        scissor_box = boxes[i]
                        needle_driver_mask = cv2.bitwise_or(needle_driver_mask, mask_binary)
                    elif class_name == "forceps":
                        forceps_mask = cv2.bitwise_or(forceps_mask, mask_binary)
                """
            
            score = 0
            # TEMPLATE MATCHING for Needle Driver (scissors)
            if scissor_mask_list != []:
                _, temp, score = self.template_matching(scissor_mask_list[0], scissors_box_list[0], 'Template_4_cropped.jpg')
                

            # Evaluate finger positions only if needle driver mask is valid
            if score > 0.6:
                print("SCORE ----> ", score)

                # Perform Mediapipe hand detection
                results = hands.process(rgb_frame)
                if results.multi_hand_landmarks:
                    thumb_in_ring1, ring_finger_in_ring2, is_parallel = self.evaluate_finger_positions(results, temp, rgb_frame)

                    # Provide detailed feedback on which conditions are failing
                    if not thumb_in_ring1:
                        print(f"Frame {frame_number}: Thumb is NOT in the correct ring.")
                    else:
                        print(f"Frame {frame_number}: Thumb is in the correct ring.")

                    if not ring_finger_in_ring2:
                        print(f"Frame {frame_number}: Ring finger is NOT in the correct ring.")
                    else:
                        print(f"Frame {frame_number}: Ring finger is in the correct ring.")

                    if not is_parallel:
                        print(f"Frame {frame_number}: Index finger is NOT parallel.")
                    else:
                        print(f"Frame {frame_number}: Index finger is parallel.")

                    # Check conditions for correct finger positions before initializing tracking
                    if ((thumb_in_ring1 and ring_finger_in_ring2 and is_parallel) or True) and not found_scissors:
                        found_scissors = True
                        frame_scissors = frame_number
                        
                        print("Fingers are in the correct position. Initializing SAM tracking.")
                        
                    if found_scissors and found_forceps:
                        print("frame_scissors: ", frame_scissors)
                        print("frame_forceps: ", frame_forceps)
                        # Initialize SAM tracking for the needle driver and forceps
                        
                        
                        self.initialize_sam_with_device(video_frame_path)
                        frame_idx = frame_number

                        # Add needle driver mask as a prompt to SAM
                        needle_driver_obj_id = 0  # Object ID for needle driver
                        
                        for mask_scissors in scissor_mask_list:
                            mask_prompt = mask_scissors.astype(bool)
                            self.sam_predictor.add_new_mask(
                                self.inference_state,
                                frame_scissors ,
                                needle_driver_obj_id,
                                mask_prompt
                            )
                        
                        forceps_obj_id = 1  # Object ID for forceps
                        for mask_forceps in forcep_mask_list:
                        # Add forceps mask as a prompt to SAM if detected
                            forceps_mask_prompt = mask_forceps.astype(bool)
                            self.sam_predictor.add_new_mask(
                                self.inference_state,
                                frame_forceps ,
                                forceps_obj_id,
                                forceps_mask_prompt 
                            ) 
                        
                            

                        inference_state_flag = True
                        break  # Exit the loop to start SAM processing

            frame_number += 1

        if inference_state_flag and self.inference_state is not None:
            # Start SAM tracking from the current frame
            capture.set(cv2.CAP_PROP_POS_FRAMES, min(frame_scissors, frame_forceps))
            if self.device == 'cuda':
                autocast_device = 'cuda'
            else:
                autocast_device = 'cpu'

            with torch.inference_mode():
                if autocast_device == 'cuda':
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4 files
                    print(rgb_frame.shape[0], rgb_frame.shape[1])
                    print(rgb_frame.shape)
                    video_writer = cv2.VideoWriter('processed_video.mp4', fourcc, 20, (rgb_frame.shape[0], rgb_frame.shape[1]))
                    
                    with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
                        for frame_idx, obj_ids, video_res_masks in self.sam_predictor.propagate_in_video(
                            self.inference_state,
                            start_frame_idx=min(frame_scissors, frame_forceps),
                            reverse=False
                        ):
                            #print(f'obje_ids: {obj_ids}') 
                            #print(f'video_res_masks: {video_res_masks}')
                            
                            ret, frame = capture.read()
                            if not ret:
                                print(f"End of video or error at frame {frame_idx}.")
                                break
                           
                            txt_filename = f"polygons_coord/{frame_idx:05d}.txt"
                        
                            with open(txt_filename, 'w') as file:
                                # Process and visualize masks
                                for obj_id, mask in zip(obj_ids, video_res_masks):
                                    # Ensure mask is a numpy array on the CPU
                                    if mask is None:
                                        print(f"Frame {frame_idx}: Object {obj_id} mask is None.")
                                        continue

                                    # Move tensor to CPU if it's on GPU, and then convert to numpy array
                                    mask = mask.cpu().numpy()
                                    #print(mask.shape)
                                    
                                    
                                    
                                    mask = np.squeeze(mask, axis=0)

                                    # Check for unique values in the mask (debugging purposes)
                                    
                                    # Clip values to remove invalid values (e.g., negative values)
                                    binary_mask = (mask > 0.5).astype(np.uint8) * 255
                                
                                    
                                    # Find contours in the binary mask
                                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, )


                                    for contour in contours:
                                        # Draw the contour on the original frame
                                        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)  # Green color for contours
                                        
                                        # Flatten the contour to a list of coordinates
                                        contour_coords = contour.reshape(-1, 2)

                                        # Write obj_id followed by the coordinates to the file
                                        file.write(f"{obj_id} " + " ".join([f"{x/binary_mask.shape[0]} {y/binary_mask.shape[1]}" for y, x in contour_coords]) + "\n")
                                        
                                #frame_resized = self.resize_and_pad(frame, 640) #cv2.resize(frame, (640, 360))  # Resizing to 640x360 for faster processing and better viewing
                                    
                                    #colored_mask = np.zeros_like(frame)
                                    #colored_mask[:, :, 0] = 0    # Blue channel (set to 0 for no blue)
                                    #colored_mask[:, :, 1] = 0    # Green channel (set to 0 for no green)
                                    #colored_mask[:, :, 2] = binary_mask  # Red channel (set to 255 for red)
                                    #alpha = 0.5 
                                    #blended_frame = cv2.addWeighted(frame, 1 - alpha, colored_mask, alpha, 0)
                                    
                                cv2.imshow('Segmented Frame', frame)
                                video_writer.write(frame)
                                
                                    
                                # Wait for keypress and check for commands
                                key = cv2.waitKey(1) & 0xFF
                                
                                if key == ord('q'):  # Press 'q' to quit
                                    break
                                                                                                
                                                    
                else:
                    print("Autocast device is not 'cuda'.")
                    return
                
            video_writer.release()
            capture.release()
            cv2.destroyAllWindows()
            print("Finished processing video with SAM tracking.")

    def evaluate_finger_positions(self, results, temp, frame):
        contours, _ = cv2.findContours(temp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        if len(contours) >= 4:
            ring1_contour = contours[2]
            ring2_contour = contours[3]
            index_contour = contours[0]

            thumb_points = []
            ring_finger_points = []
            index_points = []

            for hand_landmarks in results.multi_hand_landmarks:
                height, width, _ = frame.shape

                # Thumb landmarks (Landmark 1 to 4)
                for idx in range(1, 5):
                    x = int(hand_landmarks.landmark[idx].x * width)
                    y = int(hand_landmarks.landmark[idx].y * height)
                    thumb_points.append((x, y))

                # Ring finger landmarks (Landmark 13 to 16)
                for idx in range(13, 17):
                    x = int(hand_landmarks.landmark[idx].x * width)
                    y = int(hand_landmarks.landmark[idx].y * height)
                    ring_finger_points.append((x, y))

                # Index finger landmarks (Landmark 5 to 8)
                for idx in range(5, 9):
                    x = int(hand_landmarks.landmark[idx].x * width)
                    y = int(hand_landmarks.landmark[idx].y * height)
                    index_points.append((x, y))

            thumb_in_ring1 = self.check_points_in_contour(thumb_points, ring1_contour)
            ring_finger_in_ring2 = self.check_points_in_contour(ring_finger_points, ring2_contour)
            is_parallel, _ = self.evaluate_index_alignment(index_points, index_contour)

            return thumb_in_ring1, ring_finger_in_ring2, is_parallel

        return False, False, False

    def check_points_in_contour(self, points, contour):
        for point in points:
            result = cv2.pointPolygonTest(contour, point, False)
            if result > 0:
                return True
        return False

    def evaluate_index_alignment(self, index_points, contour_0, parallel_threshold=25):
        if len(index_points) < 2:
            print("Not enough points for evaluating index alignment.")
            return False, None

        contour_0 = np.array(contour_0).reshape(-1, 1, 2).astype(np.float32)
        line0 = cv2.fitLine(contour_0, cv2.DIST_L2, 0, 0.01, 0.01)
        vx0, vy0, x0, y0 = line0.flatten()
        angle0 = atan2(vy0, vx0)

        index_points_array = np.array(index_points, dtype=np.float32).reshape(-1, 1, 2)
        line_index = cv2.fitLine(index_points_array, cv2.DIST_L2, 0, 0.01, 0.01)
        vx_index, vy_index, x_index, y_index = line_index.flatten()
        angle_index = atan2(vy_index, vx_index)

        angle_diff = abs(degrees(angle0 - angle_index))
        angle_diff = angle_diff % 180
        if angle_diff > 90:
            angle_diff = 180 - angle_diff

        is_parallel = angle_diff <= parallel_threshold

        return is_parallel, angle_diff
    
    
if __name__ == '__main__':
    
    video_path = "/root/surgical_training/data/videos/prova_resized_PeP3.mp4"
    needle_driver_model_path = '/root/surgical_training/data/yolo_models/yolov8x-seg.pt'
    forceps_model_path = '/root/surgical_training/data/yolo_models/yolov8-added-forceps-25-ep-extended.pt'
    video_frame_path = '/root/surgical_training/data/videos/resized_cutted_compressed_PeP3_frames'

    detector = Detector(needle_driver_model_path, forceps_model_path)
    detector.process_video_real_time(video_path, video_frame_path)