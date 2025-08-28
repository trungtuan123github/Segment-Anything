"""
yolo_clip_realtime.py
YOLOv8 (detection) + CLIP (zero-shot) realtime recognition.
- Default YOLO model: yolov8n (very small, fast).
- Default CLIP: ViT-B/32 (can run on CPU with --clip_on_cpu).
"""

import cv2
import torch
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from time import time

# YOLOv8 from ultralytics
from ultralytics import YOLO

# CLIP
import clip

def preprocess_crops(crops_pil, preprocess, device):
    # crops_pil: list of PIL images
    if len(crops_pil) == 0:
        return None
    batch = torch.stack([preprocess(im) for im in crops_pil], dim=0)
    return batch.to(device)

def draw_box_label(img, box, label, score, color=(0,255,0), thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)
    txt = f"{label} {score:.2f}"
    # put filled background for text for readability
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (x1, y1-20), (x1+tw+6, y1), color, -1)
    cv2.putText(img, txt, (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

def main(args):
    # device for torch (models)
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print("[INFO] Device:", device)

    # 1) Load YOLOv8
    print("[INFO] Loading YOLO:", args.yolo_model)
    yolo = YOLO(args.yolo_model)  # ultralytics loads weights automatically
    # Set yolo to use device
    yolo.model = yolo.model.to(device)

    # 2) Load CLIP (we recommend running CLIP on CPU to save GPU memory)
    clip_device = "cpu" if args.clip_on_cpu or device=="cpu" else device
    print(f"[INFO] Loading CLIP on {clip_device} ...")
    clip_model, preprocess = clip.load(args.clip_model, device=clip_device)
    clip_model.eval()

    # prepare text tokens
    if args.labels:
        labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    else:
        labels = ["person","car","cat","dog","bottle","chair","table","phone","laptop","cup","book","bicycle"]
    with torch.no_grad():
        text_tokens = clip.tokenize(labels).to(clip_device)
        text_features = clip_model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # open camera / video
    cap = cv2.VideoCapture(0 if args.camera else args.video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video source")

    prev = time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        H, W = frame.shape[:2]

        # optional resize for speed
        if args.max_side and max(H, W) > args.max_side:
            scale = args.max_side / max(H, W)
            small = cv2.resize(frame, (int(W*scale), int(H*scale)))
        else:
            scale = 1.0
            small = frame

        # 1) YOLO detect (returns list of boxes)
        # ultralytics predict returns results; we set conf threshold via args
        results = yolo.predict(source=small, conf=args.yolo_conf, iou=args.yolo_iou, verbose=False)[0]
        # boxes in xyxy relative to small image
        boxes_xyxy = results.boxes.xyxy.cpu().numpy() if len(results.boxes) else np.array([])  # (N,4)
        scores = results.boxes.conf.cpu().numpy() if len(results.boxes) else np.array([])
        # if class labels from YOLO are desired you can also get results.boxes.cls

        # 2) Crop boxes & prepare CLIP batch
        crops = []
        crop_boxes_original = []  # store in original frame coordinates
        for (box, sc) in zip(boxes_xyxy, scores):
            x1, y1, x2, y2 = box
            # scale back to original coords
            if scale != 1.0:
                inv = 1.0/scale
                x1o, y1o, x2o, y2o = x1*inv, y1*inv, x2*inv, y2*inv
            else:
                x1o, y1o, x2o, y2o = x1, y1, x2, y2
            # enforce integer & clamp
            x1i = max(0, int(round(x1o))); y1i = max(0, int(round(y1o)))
            x2i = min(frame.shape[1], int(round(x2o))); y2i = min(frame.shape[0], int(round(y2o)))
            if x2i <= x1i or y2i <= y1i:
                continue
            crop = frame[y1i:y2i, x1i:x2i]
            # optional resize small crops to limit CLIP input
            # convert to PIL
            crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            crop_boxes_original.append((x1i, y1i, x2i, y2i))

        labels_out = []
        scores_out = []
        # run CLIP in a single batch if any crops
        if len(crops) > 0:
            # batch preprocess -> move to clip_device
            batch = torch.stack([preprocess(im) for im in crops]).to(clip_device)  # (N,3,224,224)
            with torch.no_grad():
                img_feats = clip_model.encode_image(batch)
                img_feats /= img_feats.norm(dim=-1, keepdim=True)  # normalize
                # similarity: (N, num_text)
                sims = (img_feats @ text_features.T).softmax(dim=-1)
                best_idx = sims.argmax(dim=-1)  # per crop
                best_scores = sims.max(dim=-1).values.cpu().numpy()
                best_idx = best_idx.cpu().numpy()
            for bi, sc, bidx in zip(batch, best_scores, best_idx):
                labels_out.append(labels[bidx])
                scores_out.append(float(sc))
        # draw boxes+labels on original frame
        for (box, lbl, sc) in zip(crop_boxes_original, labels_out, scores_out):
            draw_box_label(frame, box, lbl, sc)

        # show FPS
        now = time()
        fps = 1.0 / (now - prev) if now != prev else 0.0
        prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

        cv2.imshow("YOLOv8 + CLIP Realtime", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    p.add_argument("--camera", action="store_true", default=True)
    p.add_argument("--video_path", default=0)
    p.add_argument("--yolo_model", default="yolov8n.pt", help="YOLOv8 weights (defaults to yolov8n.pt)")
    p.add_argument("--yolo_conf", type=float, default=0.25)
    p.add_argument("--yolo_iou", type=float, default=0.45)
    p.add_argument("--clip_model", default="ViT-B/32")
    p.add_argument("--clip_on_cpu", action="store_true", help="run CLIP on CPU to save GPU VRAM (recommended)")
    p.add_argument("--labels", default="", help="comma-separated labels for CLIP (if empty use default list)")
    p.add_argument("--max_side", type=int, default=640, help="resize long side for speed (0 = disable)")
    args = p.parse_args()
    main(args)
