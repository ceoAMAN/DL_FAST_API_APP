# Computer Vision Object Detection Project

A comprehensive object detection system built with PyTorch. This project provides a modular framework for loading and using pre-trained object detection models like YOLOv5 to perform inference on images.

## Project Structure

```
computer_vision_1/
│
├── weights/
│   └── best.pt                # Pre-trained model weights
│
├── data/
│   └── images/                # Input images for detection
│       └── test/
│           └── image1.jpg     # Example test image
│
├── output/                    # Detection results are stored here
│   ├── result_*.jpg           # Images with detection visualizations
│   └── detection_results.log  # Log file with detection data
│
├── utils/
│   ├── datasets.py            # Dataset handling (loading, augmentation)
│   └── torch_utils.py         # PyTorch utilities (model loading, GPU handling)
│
├── scripts/
│   └── detect.py              # Main detection script
│
├── models/                    # Model architecture definitions (if needed)
│
├── utils.py                   # General utility functions
└── README.md                  # Project documentation
```

## Component Overview

### Core Modules

1. **detect.py**: Main script for performing object detection on images
   - Loads pre-trained models
   - Processes input images
   - Runs inference
   - Visualizes and saves results

2. **utils.py**: General utility functions
   - Image preprocessing
   - Non-Maximum Suppression (NMS)
   - Visualization
   - Logging
   - Directory creation

### Utility Modules

3. **utils/datasets.py**: Dataset handling
   - Image loading and preprocessing
   - Annotation handling
   - Dataset splitting (train/val/test)
   - Data augmentation

4. **utils/torch_utils.py**: PyTorch utilities
   - Model loading and saving
   - Device selection (GPU/CPU)
   - Mixed precision training
   - Random seed setting
   - Model information

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/computer_vision_1.git
cd computer_vision_1
```

2. Install requirements:
```bash
pip install torch torchvision numpy opencv-python scikit-learn
```

3. Download model weights:
   - Place your pre-trained weights file (e.g., YOLOv5 weights) in the `weights/` folder
   - Rename it to `best.pt` or specify its path when running the script

### Usage

#### Basic Object Detection

Run detection on images in the default directory:

```bash
python scripts/detect.py
```

#### Advanced Usage

```bash
python scripts/detect.py --weights weights/best.pt --source data/images --output output --conf-thres 0.25 --iou-thres 0.45
```

### Command Line Arguments

- `--weights`: Path to model weights file (default: 'weights/best.pt')
- `--source`: Source directory or single image path (default: 'data/images')
- `--output`: Output directory for results (default: 'output')
- `--img-size`: Input image size (default: 640)
- `--conf-thres`: Confidence threshold (default: 0.25)
- `--iou-thres`: IoU threshold for NMS (default: 0.45)
- `--device`: Device to use ('0', '0,1,2,3', 'cpu')

## Customizing

### Using Custom Classes

To use custom classes instead of the default COCO classes, modify the `COCO_CLASSES` list in `detect.py`.

### Using Different Model Architectures

The project is designed to work with YOLOv5 models by default, but can be adapted for other architectures:

1. If using a different architecture, you may need to adjust the loading mechanism in `load_model()` function.
2. Ensure the model's output format is compatible with the post-processing functions.

## Extending the Project

### Adding Training Capabilities

This project focuses on inference, but you can extend it for training:

1. Use `utils/datasets.py` to prepare your training data
2. Create a training script that utilizes the utilities in `utils/torch_utils.py`
3. Implement a training loop with your preferred loss function

### Integration with Tracking

To add tracking capabilities:

1. Add a tracking module that can use detections from frame to frame
2. Update the detection pipeline to maintain object IDs across frames

## Troubleshooting

### Common Issues

1. **Model loading errors**: Ensure your model format is compatible or adjust `load_model()` function.
2. **CUDA out of memory**: Reduce batch size or image size.
3. **Missing dependencies**: Install required packages from requirements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- YOLO object detection algorithms
- PyTorch framework
- OpenCV for image processing

