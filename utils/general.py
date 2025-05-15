import os
import cv2
import numpy as np
import torch
import logging
from pathlib import Path

def check_img_size(img_size, stride=32):
    """
    Verify that image size is compatible with model's stride.
    Adjusts size to ensure it's divisible by the stride.
    
    Args:
        img_size (tuple): Original image size (height, width)
        stride (int): Model stride
        
    Returns:
        tuple: Adjusted image size
    """
    new_size = (int(np.ceil(img_size[0] / stride) * stride), 
                int(np.ceil(img_size[1] / stride) * stride))
    return new_size

def load_image(image_path, img_size=(640, 640)):
    """
    Load and prepare an image for detection.
    
    Args:
        image_path (str): Path to the image file
        img_size (tuple): Target size to resize the image
        
    Returns:
        tuple: (original image, processed tensor)
    """
    # Load original image for later visualization
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Process image for model input
    img = cv2.resize(orig_img, img_size)
    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
    img = np.transpose(img, (2, 0, 1))    # HWC to CHW (height, width, channels) -> (channels, height, width)
    img = np.expand_dims(img, axis=0)     # Add batch dimension
    img = torch.from_numpy(img).float()
    
    return orig_img, img

def non_max_suppression(predictions, conf_threshold=0.25, iou_threshold=0.45, classes=None):
    """
    Apply Non-Maximum Suppression to prediction boxes.
    
    Args:
        predictions (tensor): Model predictions with shape [batch, num_predictions, 6]
                             where last dimension is [x1, y1, x2, y2, confidence, class]
        conf_threshold (float): Confidence threshold
        iou_threshold (float): IoU threshold for NMS
        classes (list): Filter by class
        
    Returns:
        list: List of detections, each in (x1, y1, x2, y2, conf, cls) format
    """
    # Extract predictions that meet confidence threshold
    nc = predictions.shape[2] - 5  # number of classes
    xc = predictions[..., 4] > conf_threshold  # candidates
    
    # Settings
    min_wh, max_wh = 2, 4096  # min and max box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum boxes into NMS
    time_limit = 10.0  # seconds to quit after
    
    outputs = [torch.zeros((0, 6), device=predictions.device)] * predictions.shape[0]
    for xi, x in enumerate(predictions):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence
        
        # If none remain, process next image
        if not x.shape[0]:
            continue
        
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        
        # Detections matrix nx6 (xyxy, conf, cls)
        if nc > 1:
            i, j = (x[:, 5:] > conf_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_threshold]
        
        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        
        # Apply finite constraint
        if not torch.isfinite(x).all():
            x = x[torch.isfinite(x).all(1)]
            
        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        
        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision_nms(boxes, scores, iou_threshold)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
            
        outputs[xi] = x[i]
        
    return outputs

def torchvision_nms(boxes, scores, iou_threshold):
    """
    Performs non-maximum suppression (NMS) on the boxes according to their IoU.
    
    Args:
        boxes (Tensor): Boxes to perform NMS on (x1, y1, x2, y2)
        scores (Tensor): Scores for each box
        iou_threshold (float): IoU threshold for NMS
        
    Returns:
        Tensor: Indices of boxes that were kept
    """
    try:
        import torchvision
        return torchvision.ops.nms(boxes, scores, iou_threshold)
    except:
        # Fallback to custom NMS if torchvision not available
        return custom_nms(boxes, scores, iou_threshold)

def custom_nms(boxes, scores, iou_threshold):
    """
    Custom implementation of NMS when torchvision is not available.
    
    Args:
        boxes (Tensor): Boxes to perform NMS on (x1, y1, x2, y2)
        scores (Tensor): Scores for each box
        iou_threshold (float): IoU threshold for NMS
        
    Returns:
        Tensor: Indices of boxes that were kept
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)
    
    keep = []
    while order.size(0) > 0:
        i = order[0].item()
        keep.append(i)
        
        if order.size(0) == 1:
            break
            
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])
        
        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = torch.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
        
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

def xywh2xyxy(x):
    """
    Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2].
    
    Args:
        x (tensor): Boxes in [x, y, w, h] format
        
    Returns:
        tensor: Boxes in [x1, y1, x2, y2] format
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def log_results(results, log_path='output/results.log'):
    """
    Save detection results to log file.
    
    Args:
        results (list): List of detection results
        log_path (str): Path to log file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    for i, det in enumerate(results):
        if len(det):
            logging.info(f"Image {i+1}: {len(det)} detections")
            for *xyxy, conf, cls in det:
                logging.info(f"  Box: [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}], " 
                             f"Confidence: {conf:.2f}, Class: {int(cls)}")
        else:
            logging.info(f"Image {i+1}: No detections")

def visualize_predictions(image, predictions, class_names=None):
    """
    Draw bounding boxes and labels on the image.
    
    Args:
        image (np.ndarray): Original image
        predictions (tensor): Model predictions after NMS
        class_names (list): List of class names
        
    Returns:
        np.ndarray: Image with bounding boxes and labels
    """
    # Make a copy of the image to avoid modifying the original
    img_copy = image.copy()
    
    # Define colors for different classes (HSV color space to get distinct colors)
    np.random.seed(42)  # for reproducibility
    colors = {i: tuple(map(int, np.random.randint(0, 255, size=3))) for i in range(80)}  # up to 80 classes
    
    # Draw each detection
    if len(predictions):
        for *xyxy, conf, cls_id in predictions:
            # Convert to integers
            x1, y1, x2, y2 = map(int, xyxy)
            
            # Get label and color
            label = f"{class_names[int(cls_id)] if class_names else int(cls_id)} {conf:.2f}"
            color = colors[int(cls_id)]
            
            # Draw rectangle
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(img_copy, (x1, y1-t_size[1]-3), (x1+t_size[0], y1), color, -1)
            
            # Put text
            cv2.putText(img_copy, label, (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return img_copy

def create_directories(directories):
    """
    Create necessary directories if they don't exist.
    
    Args:
        directories (list): List of directory paths
    """
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)