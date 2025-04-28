
Computer Vision Object Detection and Segmentation Project using
YOLOv8 
---

**Project Overview**
---
This project demonstrates advanced computer vision techniques using Ultralytics YOLOv8 for two distinct applications: trash object detection and plant disease segmentation. The research focuses on implementing state-of-the-art deep learning models for image analysis and detection.

---

**Key Features**

1. Trash Object Detection
   - Utilizes YOLOv8 Nano (yolov8n.pt) model
   - Identifies and localizes trash items in images

2. Plant Disease Segmentation
   - Implements YOLOv8 X-Segment (yolov8x-seg.pt) model
   - Performs pixel-level segmentation of plant disease regions

---

**Technical Specifications**

1. Trash Dataset
   - Training Epochs: 5
   - Model: YOLOv8 Nano
   - Image Size: 640x640
   - Batch Size: 4
 
2. Plant Disease Dataset
   - Training Epochs: 10
   - Model: YOLOv8 X-Segment
   - Image Size: 640x640
   - Batch Size: 4

**Evaluation Metrics**

***Mean Average Precision (mAP)***
 - mAP50: Precision at Intersection over Union (IoU) 0.50
 - mAP50-95: Precision across multiple IoU thresholds

---

**Installation**

***Prerequisites***
- Python 3.8+
- CUDA-compatible GPU (recommended)

***Dependencies***

```
pip install ultralytics opencv-python-headless supervision wget matplotlib PyYAML tqdm
```
---
**Key Functions**

- **train_yolo_model()**: Train object detection model
- **evaluate_model()**: Compute performance metrics
- **visualize_predictions()**: Visualize model predictions
- **train_segmentation_model()**: Train image segmentation model
- **visualize_segmentation()**: Visualize segmentation results

---
***The project includes visualization functions that:***
- Display bounding boxes for object detection
- Show segmentation masks for plant diseases
- Provide confidence threshold filtering
---
**Future Improvements**
- Experiment with different YOLO model architectures
- Increase training epochs
- Implement data augmentation techniques
- Fine-tune hyperparameters
- Explore transfer learning approaches

**Performance Considerations**
- Requires sufficient computational resources
- GPU acceleration recommended
- Model performance varies with dataset complexity

**Troubleshooting**
- Ensure all dependencies are correctly installed
- Check CUDA and GPU compatibility
- Verify dataset structure and image formats

Acknowledgments

Ultralytics for YOLO framework
Dataset providers and contributors
