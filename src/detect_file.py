"""
Image & Video File Detection — Face Mask Detector
Usage:
    # Single image
    python src/detect_file.py --input photo.jpg --mode image

    # Video file
    python src/detect_file.py --input video.mp4 --mode video --output output.mp4
"""

import cv2
import torch
import argparse
import os
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(__file__))

from model import load_model
from dataset import INFERENCE_TRANSFORMS, CLASS_NAMES
from face_detector import load_face_detector, detect_faces, download_face_detector


LABEL_COLOR = {
    "with_mask":    (0, 200, 0),
    "without_mask": (0, 0, 220),
}


def predict_face(model, face_roi, device):
    pil_img = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
    tensor  = INFERENCE_TRANSFORMS(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]
        pred    = torch.argmax(probs).item()
    return CLASS_NAMES[pred], probs[pred].item()


def annotate_frame(frame, face_net, mask_model, device):
    """Detect faces, classify each, draw annotations. Returns annotated frame."""
    boxes = detect_faces(face_net, frame)
    for (startX, startY, endX, endY) in boxes:
        face_roi = frame[startY:endY, startX:endX]
        if face_roi.size == 0:
            continue
        label, conf = predict_face(mask_model, face_roi, device)
        color = LABEL_COLOR[label]
        text  = f"{label.replace('_', ' ').title()} {conf*100:.1f}%"
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        y = startY - 10 if startY > 20 else startY + 20
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return frame, len(boxes)


# ── Image Mode ────────────────────────────────────────────────────────────────

def detect_image(input_path, output_path, face_net, mask_model, device):
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {input_path}")
        return

    annotated, n_faces = annotate_frame(frame, face_net, mask_model, device)
    print(f"[Image] Detected {n_faces} face(s) in '{input_path}'")

    if output_path:
        cv2.imwrite(output_path, annotated)
        print(f"[Image] Saved → {output_path}")
    else:
        cv2.imshow("Mask Detection", annotated)
        print("[Image] Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ── Video Mode ────────────────────────────────────────────────────────────────

def detect_video(input_path, output_path, face_net, mask_model, device):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    print(f"[Video] Processing {total} frames …")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, _ = annotate_frame(frame, face_net, mask_model, device)
        frame_idx += 1

        if writer:
            writer.write(annotated)
        else:
            cv2.imshow("Mask Detection — Video", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_idx % 50 == 0:
            print(f"  Progress: {frame_idx}/{total}")

    cap.release()
    if writer:
        writer.release()
        print(f"[Video] Saved → {output_path}")
    cv2.destroyAllWindows()
    print("[Video] Done.")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Mask detection on image/video files")
    parser.add_argument("--input",    required=True, help="Path to input image or video")
    parser.add_argument("--mode",     choices=["image", "video"], default="image")
    parser.add_argument("--output",   default=None,  help="Output file path (optional)")
    parser.add_argument("--weights",  default="models/mask_detector.pth")
    parser.add_argument("--face_dir", default="face_detector")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    proto, caffemodel = download_face_detector(args.face_dir)
    face_net   = load_face_detector(proto, caffemodel)
    mask_model = load_model(args.weights, device)

    if args.mode == "image":
        detect_image(args.input, args.output, face_net, mask_model, device)
    else:
        detect_video(args.input, args.output, face_net, mask_model, device)


if __name__ == "__main__":
    main()
