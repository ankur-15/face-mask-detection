# 😷 Face Mask Detection using PyTorch & OpenCV

A real-time computer vision system that detects whether a person is wearing a face mask or not — using a fine-tuned **MobileNetV2** classifier and OpenCV's **DNN-based face detector**.

---

## 📁 Project Structure

```
face_mask_detection/
├── data/
│   ├── with_mask/          ← face images wearing masks
│   └── without_mask/       ← face images without masks
├── face_detector/          ← auto-downloaded OpenCV DNN model files
├── models/
│   └── mask_detector.pth   ← saved best model weights
├── plots/
│   ├── training_curves.png ← loss & accuracy plots
│   └── confusion_matrix.png
├── src/
│   ├── model.py            ← MobileNetV2 classifier
│   ├── dataset.py          ← Dataset loader & augmentations
│   ├── face_detector.py    ← OpenCV DNN face detection utility
│   ├── train.py            ← Training script
│   ├── detect_webcam.py    ← Real-time webcam detection
│   └── detect_file.py      ← Image / video file detection
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/face-mask-detection.git
cd face-mask-detection
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📦 Dataset

This project is compatible with the popular **Kaggle Face Mask Dataset**:

- 🔗 [Face Mask Detection Dataset on Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)

After downloading, place images as:
```
data/
├── with_mask/      ← ~3725 images
└── without_mask/   ← ~3828 images
```

---

## 🏋️ Training

```bash
python src/train.py \
    --data_dir  data/ \
    --epochs    20 \
    --batch_size 32 \
    --lr        0.0001 \
    --save_dir  models/ \
    --plot_dir  plots/
```

After training you will find:
- `models/mask_detector.pth` — best model weights
- `plots/training_curves.png` — loss & accuracy graph
- `plots/confusion_matrix.png` — confusion matrix

---

## 🎥 Real-Time Webcam Detection

```bash
python src/detect_webcam.py --weights models/mask_detector.pth
```

| Key | Action |
|-----|--------|
| `q` | Quit   |

---

## 🖼️ Detect on Image / Video File

**Image:**
```bash
python src/detect_file.py --input photo.jpg --mode image
# Save output:
python src/detect_file.py --input photo.jpg --mode image --output result.jpg
```

**Video:**
```bash
python src/detect_file.py --input video.mp4 --mode video --output result.mp4
```

---

## 🧠 Model Architecture

| Component          | Detail                          |
|--------------------|---------------------------------|
| **Backbone**       | MobileNetV2 (pretrained ImageNet) |
| **Classifier Head** | Dropout → Linear(256) → ReLU → Dropout → Linear(2) |
| **Face Detector**  | OpenCV DNN — ResNet-10 SSD      |
| **Input Size**     | 224 × 224 RGB                   |
| **Classes**        | `with_mask`, `without_mask`     |

---

## 📊 Results (Example)

| Metric     | With Mask | Without Mask |
|------------|-----------|--------------|
| Precision  | 0.98      | 0.97         |
| Recall     | 0.97      | 0.98         |
| F1-Score   | 0.97      | 0.97         |
| **Overall Accuracy** | **~97.5%** | |

> Results may vary depending on dataset and hardware. Train on GPU for best speed.

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **PyTorch** — model training & inference
- **Torchvision** — pretrained MobileNetV2 + image transforms
- **OpenCV** — face detection (DNN) + video I/O
- **scikit-learn** — metrics & confusion matrix
- **Matplotlib / Seaborn** — plots

---

## 👤 Author

**Your Name**  
B.Tech Computer Science | [Your College Name]  
GitHub: [@your-username](https://github.com/your-username)

---

## 📜 License

This project is licensed under the MIT License.
