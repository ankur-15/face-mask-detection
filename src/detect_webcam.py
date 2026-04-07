"""
Real-Time Webcam Detection — Face Mask Detector
Usage:
    python src/detect_webcam.py --weights models/mask_detector.pth
"""

import cv2
import torch
import numpy as np
import argparse
from PIL import Image

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from model import load_model
from dataset import INFERENCE_TRANSFORMS, CLASS_NAMES
from face_detector import load_face_detector, detect_faces, download_face_detector


# Label colors: green = with mask, red = without mask
LABEL_COLOR = {
    "with_mask":    (0, 200, 0),
    "without_mask": (0, 0, 220),
}


def predict_face(model, face_roi, device):
    """Run mask classifier on a cropped face ROI (BGR numpy array)."""
    pil_img = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
    tensor  = INFERENCE_TRANSFORMS(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]
        pred    = torch.argmax(probs).item()
    label = CLASS_NAMES[pred]
    confidence = probs[pred].item()
    return label, confidence


def run_webcam(weights_path, face_detector_dir="face_detector", cam_index=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Load models
    proto, caffemodel = download_face_detector(face_detector_dir)
    face_net  = load_face_detector(proto, caffemodel)
    mask_model = load_model(weights_path, device)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("[INFO] Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detect_faces(face_net, frame)

        for (startX, startY, endX, endY) in boxes:
            face_roi = frame[startY:endY, startX:endX]
            if face_roi.size == 0:
                continue

            label, confidence = predict_face(mask_model, face_roi, device)
            color = LABEL_COLOR[label]
            display = f"{label.replace('_', ' ').title()} ({confidence*100:.1f}%)"

            # Draw box & label
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            y = startY - 10 if startY > 20 else startY + 20
            cv2.putText(frame, display, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # FPS overlay
        cv2.putText(frame, "Press Q to quit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Face Mask Detection — Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time mask detection")
    parser.add_argument("--weights",    default="models/mask_detector.pth")
    parser.add_argument("--face_dir",  default="face_detector")
    parser.add_argument("--cam",       type=int, default=0, help="Camera index")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_webcam(args.weights, args.face_dir, args.cam)
