from ultralytics import YOLO
import torch

def train_yolo(model_path, data_path, epochs=5, img_size=640, batch_size=4, device=0, project='segmentation', name='yolov8-finetune', workers=2, augment=False, freeze=20):
    """
    Train a YOLOv8 segmentation model.

    Parameters:
    model_path (str): Path to the YOLO model weights (e.g., 'yolov8x-seg.pt').
    data_path (str): Path to dataset configuration file.
    epochs (int): Number of training epochs.
    img_size (int): Input image size.
    batch_size (int): Batch size.
    device (int or str): Device to use for training (0 for GPU, 'cpu' for CPU).
    project (str): Project folder name.
    name (str): Experiment name.
    workers (int): Number of data loader workers.
    augment (bool): Whether to use augmentation.
    freeze (int): Number of layers to freeze.
    """
    model = YOLO(model_path)

    print(data_path)

    # Train the model
    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        workers=workers,
        augment=augment,
        freeze=freeze
    )
    
    print("Training completed!")

if __name__ == "__main__":
    
    train_yolo(
        model_path='yolov8x-seg.pt',
        data_path='datasets/dataset.yaml',
        epochs=10,
        img_size=640,
        batch_size=2,
        device=0,
        project='segmentation',
        name='yolov8-finetune',
        workers=2,
        augment=True,
        freeze=20
    )