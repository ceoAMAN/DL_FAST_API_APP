import os
import argparse
import torch
import cv2
import time
import numpy as np
from pathlib import Path
from utils import (
    check_img_size, 
    load_image, 
    non_max_suppression, 
    log_results, 
    visualize_predictions,
    create_directories
)

# List of COCO class names (example - can be replaced with your specific classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

def load_model(model_path='weights/best.pt', device='cuda'):
    """
    Load the pre-trained model from the specified path.
    
    Args:
        model_path (str): Path to model weights
        device (str): Device to load model on ('cuda' or 'cpu')
        
    Returns:
        model: Loaded PyTorch model
    """
    print(f"Loading model from {model_path}...")
    try:
        # Attempt to load model using different methods
        try:
            # Try loading using torch.load
            model = torch.load(model_path, map_location=device)
            
            # Check if model is a state dict, if so, need to load it into structure
            if isinstance(model, dict) and 'model' in model:
                model = model['model']
                
        except Exception as e:
            print(f"Standard loading failed, trying alternate method: {e}")
            # Try loading using torch hub (YOLOv5 style)
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            
        # Move model to device and set to evaluation mode
        model.to(device)
        model.eval()
        print(f"Model loaded successfully on {device}")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def inference(model, image, device='cuda'):
    """
    Run inference on the input image.
    
    Args:
        model: The loaded PyTorch model
        image (tensor): Preprocessed image tensor
        device (str): Device to perform inference on
        
    Returns:
        tensor: Raw model predictions
    """
    print("Running inference...")
    start_time = time.time()
    
    with torch.no_grad():
        # Move image to device if not already
        image = image.to(device)
        
        # Run inference
        predictions = model(image)
        
        # Handle different model output formats
        if isinstance(predictions, tuple):
            predictions = predictions[0]  # Some models return additional info
        
    elapsed = time.time() - start_time
    print(f"Inference completed in {elapsed:.2f} seconds")
    
    return predictions

def process_images(model, image_paths, output_dir, conf_threshold=0.25, iou_threshold=0.45, device='cuda'):
    """
    Process multiple images with the model.
    
    Args:
        model: The loaded PyTorch model
        image_paths (list): List of image paths to process
        output_dir (str): Directory to save results
        conf_threshold (float): Confidence threshold for detections
        iou_threshold (float): IoU threshold for NMS
        device (str): Device to perform inference on
    """
    # Create output directory
    create_directories([output_dir])
    
    # Process each image
    for i, img_path in enumerate(image_paths):
        print(f"\nProcessing image {i+1}/{len(image_paths)}: {img_path}")
        
        try:
            # Load and preprocess image
            orig_img, img_tensor = load_image(img_path)
            
            # Run inference
            predictions = inference(model, img_tensor, device)
            
            # Apply NMS to get final detections
            results = non_max_suppression(
                predictions, 
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
            
            if len(results[0]):
                print(f"Found {len(results[0])} objects")
            else:
                print("No objects detected")
            
            # Visualize results
            output_img = visualize_predictions(orig_img, results[0], COCO_CLASSES)
            
            # Save output image
            output_path = os.path.join(output_dir, f"result_{Path(img_path).stem}.jpg")
            cv2.imwrite(output_path, output_img)
            print(f"Results saved to {output_path}")
            
            # Log results
            log_path = os.path.join(output_dir, "detection_results.log")
            log_results(results, log_path)
            
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Object Detection with PyTorch")
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='Model weights path')
    parser.add_argument('--source', type=str, default='data/images', help='Source directory with images')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--img-size', type=int, default=640, help='Inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    
    return parser.parse_args()

if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(model_path=args.weights, device=device)
    
    # Get image paths
    if os.path.isdir(args.source):
        image_paths = [os.path.join(args.source, f) for f in os.listdir(args.source) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    else:
        image_paths = [args.source]
    
    if not image_paths:
        print(f"No images found in {args.source}")
        exit(1)
    
    print(f"Found {len(image_paths)} images")
    
    # Process images
    process_images(
        model=model,
        image_paths=image_paths,
        output_dir=args.output,
        conf_threshold=args.conf_thres,
        iou_threshold=args.iou_thres,
        device=device
    )
    
    print("\nDetection completed!")