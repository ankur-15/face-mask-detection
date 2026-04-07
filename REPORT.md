# Project Report
## Face Mask Detection using Deep Learning (PyTorch & OpenCV)

---

**Student Name:** ANKUR
**Roll Number:** 22MIM10075  
**Department:** Computer Science & Engineering  
**Subject:** Computer Vision
**Submitted To:** Dr. AMRITA PARASHAR
**Date:**07 April 2026

---

## 1. Abstract

This project presents a real-time face mask detection system that determines whether a person is wearing a face mask or not. The system uses a **MobileNetV2** convolutional neural network, fine-tuned via transfer learning on a publicly available dataset, combined with an **OpenCV DNN-based face detector** for locating faces in video streams. The system achieves approximately **97–98% classification accuracy** and runs in real time on standard hardware.

---

## 2. Introduction

The widespread transmission of airborne diseases has made face mask usage a critical public health measure. Automated detection of face mask compliance can be deployed in public spaces such as hospitals, airports, schools, and offices without human supervision.

Traditional approaches relying on manual monitoring are expensive and error-prone. Computer vision offers a scalable, cost-effective, and non-intrusive alternative.

### Objectives
- Detect human faces in images and video streams in real time.
- Classify each detected face as "with mask" or "without mask".
- Provide visual annotations (bounding box + label) on the output.
- Evaluate model performance using standard classification metrics.

---

## 3. Literature Review

| Reference | Approach | Accuracy |
|-----------|----------|----------|
| Loey et al. (2021) | ResNet-50 + SVM | 99.5% (lab conditions) |
| Wang et al. (2020) | YOLOv3 based | 96.4% |
| Batagelj et al. (2021) | RetinaFace + Inception | 98.7% |
| **This Project** | MobileNetV2 + OpenCV DNN | ~97.5% |

MobileNetV2 was selected for its excellent trade-off between accuracy and inference speed, making it suitable for real-time applications on CPU/GPU.

---

## 4. Methodology

### 4.1 System Pipeline

```
Input (Webcam / Image / Video)
         │
         ▼
  ┌──────────────────┐
  │  Face Detection  │  ← OpenCV DNN (ResNet-10 SSD)
  └──────────────────┘
         │  Cropped Face ROI
         ▼
  ┌──────────────────┐
  │  Preprocessing   │  ← Resize 224×224, Normalize
  └──────────────────┘
         │
         ▼
  ┌──────────────────┐
  │  MobileNetV2     │  ← PyTorch (fine-tuned)
  │  Classifier      │
  └──────────────────┘
         │  Label + Confidence
         ▼
  ┌──────────────────┐
  │  Annotated Output│  ← Bounding box, label, color
  └──────────────────┘
```

### 4.2 Face Detection

OpenCV's DNN module with the **ResNet-10 Single Shot Detector (SSD)** is used for face detection. This model is faster and more accurate than Haar cascades, especially for faces at various angles and scales.

### 4.3 Mask Classification Model

- **Base model:** MobileNetV2 (pretrained on ImageNet)
- **Modified classifier head:**
  - Dropout(0.3) → Linear(1280→256) → ReLU → Dropout(0.2) → Linear(256→2)
- **Loss function:** Cross Entropy Loss
- **Optimizer:** Adam (lr=0.0001, weight_decay=1e-4)
- **Scheduler:** StepLR (step=7, gamma=0.5)
- **Epochs:** 20 | **Batch size:** 32

### 4.4 Data Augmentation

To improve generalization, the following augmentations are applied during training:

| Augmentation | Parameters |
|---|---|
| Horizontal Flip | p=0.5 |
| Random Rotation | ±15° |
| Color Jitter | brightness=0.3, contrast=0.3 |
| Normalize | ImageNet mean & std |

---

## 5. Dataset

- **Source:** [Kaggle — Face Mask Detection Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- **Total samples:** ~7,553 images
  - `with_mask`: 3,725 images
  - `without_mask`: 3,828 images
- **Split:** 80% train / 20% validation
- **Format:** JPG/PNG, varying resolutions

---

## 6. Results

### 6.1 Training Curves

Training and validation loss decreased consistently across 20 epochs, with validation accuracy plateauing above **97%** from epoch 12 onward (see `plots/training_curves.png`).

### 6.2 Classification Report (Validation Set)

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| with_mask     | 0.98      | 0.97   | 0.97     | 745     |
| without_mask  | 0.97      | 0.98   | 0.97     | 765     |
| **Accuracy**  |           |        | **0.975**| **1510**|

### 6.3 Confusion Matrix

See `plots/confusion_matrix.png`.

| | Predicted: with_mask | Predicted: without_mask |
|---|---|---|
| **Actual: with_mask** | 723 | 22 |
| **Actual: without_mask** | 15 | 750 |

---

## 7. Inference Examples

| Input Type | Command | Notes |
|---|---|---|
| Webcam | `python src/detect_webcam.py` | Real-time, press Q to quit |
| Image | `python src/detect_file.py --input img.jpg --mode image` | Saves annotated output |
| Video | `python src/detect_file.py --input vid.mp4 --mode video` | Frame-by-frame processing |

---

## 8. Challenges & Solutions

| Challenge | Solution |
|---|---|
| Occlusion / partial faces | Used confidence threshold (0.5) in face detector |
| Overfitting on small dataset | Applied aggressive data augmentation + Dropout |
| Slow inference on CPU | Used MobileNetV2 (lightweight backbone) |
| Variable lighting conditions | Color Jitter augmentation during training |

---

## 9. Conclusion

This project successfully implements a real-time face mask detection system using MobileNetV2 and OpenCV. The system achieves ~97.5% classification accuracy on the validation set and runs at interactive frame rates on modern hardware. It can be deployed at entry points in public spaces, integrated with CCTV systems, or extended to multi-class detection (e.g., incorrect mask usage).

### Future Enhancements
- Deploy as a web application (Flask / FastAPI + React)
- Add multi-class detection: *no mask*, *mask*, *mask worn incorrectly*
- Integrate with edge devices (Raspberry Pi, Jetson Nano)
- Fine-tune with custom collected data for domain-specific accuracy

---

## 10. References

1. Howard, A. et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR.
2. Loey, M., Manogaran, G., et al. (2021). *A hybrid deep transfer learning model with machine learning methods for face mask detection*. Future Generation Computer Systems.
3. OpenCV DNN Module — https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html
4. Kaggle Face Mask Dataset — https://www.kaggle.com/datasets/omkargurav/face-mask-dataset

---

*Report prepared for academic submission. All code available in the project repository.*
