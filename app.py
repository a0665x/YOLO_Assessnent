import base64
import json
import math
import os
import random
import threading
import time
import urllib.request
import uuid
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None

try:
    from tflite_runtime.interpreter import Interpreter
except Exception:  # pragma: no cover
    Interpreter = None


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
VIDEO_DIR = ROOT / "videos"
IMAGE_DIR = ROOT / "images"
REPORT_DIR = ROOT / "reports"
OUTPUT_DIR = ROOT / "outputs"
DOWNLOAD_DIR = Path(os.getenv("DOWNLOAD_DIR", str(ROOT / "downloads")))

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
TOP_K = 300
NMS_IOU_THRESHOLD = 0.45
DEFAULT_CONF_THRESHOLD = 0.25
PERSON_CLASS_ID = 0
MAX_EXAMPLES = 24
MAX_BIN_SAMPLES = 6
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SIZE_BINS = [
    {"key": "micro", "label": "<0.15%", "min_pct": 0.0, "max_pct": 0.15},
    {"key": "tiny", "label": "0.15-0.25%", "min_pct": 0.15, "max_pct": 0.25},
    {"key": "xs", "label": "0.25-0.5%", "min_pct": 0.25, "max_pct": 0.5},
    {"key": "s", "label": "0.5-1%", "min_pct": 0.5, "max_pct": 1.0},
    {"key": "m", "label": "1-2%", "min_pct": 1.0, "max_pct": 2.0},
    {"key": "l", "label": "2-4%", "min_pct": 2.0, "max_pct": 4.0},
    {"key": "xl", "label": "4-8%", "min_pct": 4.0, "max_pct": 8.0},
    {"key": "xxl", "label": ">=8%", "min_pct": 8.0, "max_pct": 10_000.0},
]
CONF_BINS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]

CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
]


def list_files(directory: Path, suffixes):
    if not directory.exists():
        return []
    return sorted(
        [p.name for p in directory.iterdir() if p.is_file() and p.suffix.lower() in suffixes]
    )


def safe_name(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    return "".join(ch if ch in allowed else "_" for ch in str(value).strip())[:80] or "report"


def size_bin_label(area_pct):
    for item in SIZE_BINS:
        if item["min_pct"] <= area_pct < item["max_pct"]:
            return item["label"]
    return "unknown"


def conf_bin_label(conf):
    for lo, hi in CONF_BINS:
        if lo <= conf < hi:
            return f"{lo:.2f}-{hi:.2f}"
    return "unknown"


def jpeg_data_uri(image, max_width=220):
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        image = cv2.resize(image, (max_width, max(1, int(h * scale))))
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def edge_debug_images(crop):
    if crop is None or crop.size == 0:
        return {}
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(gray, 50, 150)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_norm = cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.normalize(np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return {
        "crop": jpeg_data_uri(crop, max_width=360),
        "gray": jpeg_data_uri(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), max_width=360),
        "canny": jpeg_data_uri(cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR), max_width=360),
        "laplacian": jpeg_data_uri(cv2.cvtColor(lap_norm, cv2.COLOR_GRAY2BGR), max_width=360),
        "sobel": jpeg_data_uri(cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR), max_width=360),
        "blur": jpeg_data_uri(cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR), max_width=360),
    }


def draw_overlay(frame, objects):
    canvas = frame.copy()
    for obj in objects:
        x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
        track_id = obj.get("track_id")
        conf = obj.get("conf", 0.0)
        sharp = obj.get("sharpness_score", 0.0)
        color = (38, 132, 255) if sharp >= 55 else (0, 153, 217) if sharp >= 35 else (31, 95, 190)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"ID {track_id} conf {conf:.2f} sharp {sharp:.0f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(canvas, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 4, max(14, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def nms(boxes, scores, threshold):
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter + 1e-6
        order = order[1:][inter / union < threshold]
    return keep


def iou(box_a, box_b):
    xa = max(float(box_a[0]), float(box_b[0]))
    ya = max(float(box_a[1]), float(box_b[1]))
    xb = min(float(box_a[2]), float(box_b[2]))
    yb = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    return inter / (area_a + area_b - inter + 1e-6)


class SimpleTracker:
    def __init__(self, threshold=0.35, max_misses=45):
        self.threshold = threshold
        self.max_misses = max_misses
        self.next_id = 1
        self.tracks = []

    def update(self, detections):
        for track in self.tracks:
            track["misses"] += 1

        unmatched = set(range(len(detections)))
        for track in list(self.tracks):
            best_idx = None
            best_score = 0.0
            for idx in list(unmatched):
                if int(detections[idx]["class_id"]) != int(track["class_id"]):
                    continue
                score = iou(track["bbox"], detections[idx]["bbox"])
                if score > best_score:
                    best_idx = idx
                    best_score = score
            if best_idx is not None and best_score >= self.threshold:
                det = detections[best_idx]
                track.update(
                    bbox=det["bbox"],
                    conf=det["conf"],
                    class_id=det["class_id"],
                    class_name=det["class_name"],
                    misses=0,
                    hits=track["hits"] + 1,
                )
                unmatched.discard(best_idx)

        for idx in sorted(unmatched):
            det = detections[idx]
            self.tracks.append(
                {
                    "track_id": self.next_id,
                    "bbox": det["bbox"],
                    "conf": det["conf"],
                    "class_id": det["class_id"],
                    "class_name": det["class_name"],
                    "hits": 1,
                    "misses": 0,
                }
            )
            self.next_id += 1

        self.tracks = [track for track in self.tracks if track["misses"] <= self.max_misses]
        return [dict(track) for track in self.tracks if track["misses"] == 0]


class YoloTFLiteRunner:
    def __init__(self):
        self.model_name = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None

    def load(self, model_name):
        model_path = (MODEL_DIR / model_name).resolve()
        if MODEL_DIR.resolve() not in model_path.parents or model_path.suffix != ".tflite":
            raise ValueError("invalid model path")
        if not model_path.exists():
            raise FileNotFoundError(str(model_path))
        if tf is not None:
            self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        elif Interpreter is not None:
            self.interpreter = Interpreter(model_path=str(model_path))
        else:
            raise RuntimeError("tensorflow or tflite_runtime is required")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.model_name = model_name

    def infer(self, frame, conf_threshold):
        if self.interpreter is None:
            raise RuntimeError("model is not loaded")
        frame_h, frame_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_info = self.input_details[0]
        _, in_h, in_w, _ = input_info["shape"]
        resized = cv2.resize(rgb, (int(in_w), int(in_h)))
        if input_info["dtype"] == np.uint8:
            input_data = np.expand_dims(resized.astype(np.uint8), axis=0)
        else:
            input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

        self.interpreter.set_tensor(input_info["index"], input_data)
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.output_details[0]["index"])
        return self._postprocess(raw, frame_w, frame_h, conf_threshold)

    def _postprocess(self, raw_output, frame_w, frame_h, conf_threshold):
        output = np.squeeze(raw_output)
        if output.ndim != 2:
            return []
        if output.shape[0] < output.shape[1] and output.shape[0] <= 128:
            output = output.T
        if output.shape[1] < 6:
            return []

        boxes, scores, class_ids = [], [], []
        coords = output[:, :4]
        class_scores = output[:, 4:]
        best_conf = np.max(class_scores, axis=1)
        best_class = np.argmax(class_scores, axis=1)
        for idx in np.argsort(best_conf)[::-1][:TOP_K]:
            conf = float(best_conf[idx])
            class_id = int(best_class[idx])
            if conf < conf_threshold:
                continue
            cx, cy, bw, bh = [float(v) for v in coords[idx]]
            if max(cx, cy, bw, bh) <= 1.5:
                cx *= frame_w
                bw *= frame_w
                cy *= frame_h
                bh *= frame_h
            x1 = max(0.0, cx - bw / 2.0)
            y1 = max(0.0, cy - bh / 2.0)
            x2 = min(float(frame_w - 1), cx + bw / 2.0)
            y2 = min(float(frame_h - 1), cy + bh / 2.0)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            class_ids.append(class_id)

        keep = nms(boxes, scores, NMS_IOU_THRESHOLD)
        return [
            {
                "bbox": [int(round(v)) for v in boxes[i]],
                "conf": round(float(scores[i]), 4),
                "class_id": int(class_ids[i]),
                "class_name": CLASS_NAMES[class_ids[i]] if class_ids[i] < len(CLASS_NAMES) else f"class_{class_ids[i]}",
            }
            for i in keep
        ]


def quality_metrics(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return {}, crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    gradient_mean = float(np.mean(gradient_mag))
    tenengrad = float(np.mean(gradient_mag * gradient_mag))
    edge_pixels = gray[edges > 0]
    edge_contrast = float(np.std(edge_pixels)) if edge_pixels.size else 0.0
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = float(np.mean(hsv[:, :, 1]))
    exposure_penalty = abs(brightness - 128.0) / 128.0
    lap_score = min(math.log1p(max(lap_var, 0.0)) / math.log1p(3000.0), 1.0) * 28.0
    tenengrad_score = min(math.log1p(max(tenengrad, 0.0)) / math.log1p(16_000.0), 1.0) * 24.0
    gradient_score = min(math.log1p(max(gradient_mean, 0.0)) / math.log1p(130.0), 1.0) * 16.0
    edge_density_score = min(edge_density / 0.18, 1.0) * 14.0
    edge_contrast_score = min(edge_contrast / 75.0, 1.0) * 12.0
    contrast_score = min(contrast / 85.0, 1.0) * 6.0
    sharpness_score = max(
        0.0,
        min(
            100.0,
            lap_score
            + tenengrad_score
            + gradient_score
            + edge_density_score
            + edge_contrast_score
            + contrast_score
            - exposure_penalty * 10.0,
        ),
    )
    contour_clarity_score = max(
        0.0,
        min(
            100.0,
            sharpness_score * 0.42
            + min(edge_density / 0.16, 1.0) * 26.0
            + min(edge_contrast / 70.0, 1.0) * 18.0
            + min(contrast / 80.0, 1.0) * 14.0,
        ),
    )
    return (
        {
            "laplacian_var": round(lap_var, 3),
            "laplacian_score": round(lap_score, 2),
            "tenengrad": round(tenengrad, 3),
            "tenengrad_score": round(tenengrad_score, 2),
            "edge_density": round(edge_density, 5),
            "edge_density_score": round(edge_density_score, 2),
            "edge_contrast": round(edge_contrast, 3),
            "edge_contrast_score": round(edge_contrast_score, 2),
            "gradient_mean": round(gradient_mean, 3),
            "gradient_score": round(gradient_score, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "saturation": round(saturation, 2),
            "sharpness_score": round(sharpness_score, 2),
            "contour_clarity_score": round(contour_clarity_score, 2),
            "blur_score": round(max(0.0, 100.0 - sharpness_score), 2),
        },
        crop,
    )


def frame_quality_metrics(frame, previous_gray=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)
    exposure_error = abs(brightness - 128.0) / 128.0
    exposure_score = max(0.0, min(100.0, 100.0 - exposure_error * 100.0))
    sharpness_score = max(0.0, min(100.0, math.log1p(max(lap_var, 0.0)) / math.log1p(1_200.0) * 100.0))
    temporal_delta = 0.0
    if previous_gray is not None and previous_gray.shape == gray.shape:
        temporal_delta = float(np.mean(cv2.absdiff(gray, previous_gray)))
    motion_blur = max(
        0.0,
        min(
            100.0,
            (100.0 - sharpness_score) * 0.65
            + min(temporal_delta / 35.0, 1.0) * 35.0
            - min(edge_density / 0.08, 1.0) * 12.0,
        ),
    )
    lux_proxy = max(0.0, min(100.0, (brightness / 255.0) * 70.0 + min(contrast / 80.0, 1.0) * 30.0))
    quality_score = (
        sharpness_score * 0.26
        + exposure_score * 0.22
        + min(contrast / 70.0, 1.0) * 100.0 * 0.18
        + max(0.0, 100.0 - motion_blur) * 0.18
        + lux_proxy * 0.16
    )
    return {
        "frame_sharpness": round(sharpness_score, 2),
        "frame_laplacian_var": round(lap_var, 3),
        "frame_brightness": round(brightness, 2),
        "frame_exposure_score": round(exposure_score, 2),
        "frame_motion_blur": round(motion_blur, 2),
        "frame_contrast": round(contrast, 2),
        "frame_lux_proxy": round(lux_proxy, 2),
        "frame_edge_density": round(edge_density, 5),
        "frame_quality_score": round(quality_score, 2),
    }, gray


@dataclass
class AssessmentState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    runner: YoloTFLiteRunner = field(default_factory=YoloTFLiteRunner)
    tracker: SimpleTracker = field(default_factory=SimpleTracker)
    job_id: str | None = None
    running: bool = False
    stop_requested: bool = False
    error: str | None = None
    model_name: str | None = None
    source: dict = field(default_factory=dict)
    analysis_mode: str = "idle"
    frame_index: int = 0
    frame_count: int | None = None
    fps: float = 0.0
    started_at: float | None = None
    last_frame_jpeg: bytes | None = None
    detections: list = field(default_factory=list)
    events: list = field(default_factory=list)
    examples: list = field(default_factory=list)
    bin_samples: dict = field(default_factory=dict)
    area_conf_counts: Counter = field(default_factory=Counter)
    class_counts: Counter = field(default_factory=Counter)
    area_counts: Counter = field(default_factory=Counter)
    conf_counts: Counter = field(default_factory=Counter)
    quality_history: deque = field(default_factory=lambda: deque(maxlen=5000))
    frame_quality_history: deque = field(default_factory=lambda: deque(maxlen=5000))
    timing_history: deque = field(default_factory=lambda: deque(maxlen=600))
    track_history: dict = field(default_factory=dict)
    previous_gray: np.ndarray | None = None

    def reset_job(self, job_id, model_name, source):
        self.tracker = SimpleTracker()
        self.job_id = job_id
        self.running = True
        self.stop_requested = False
        self.error = None
        self.model_name = model_name
        self.source = source
        self.analysis_mode = "tracking" if source.get("type") in {"video", "webcam"} else "image_batch"
        self.frame_index = 0
        self.frame_count = None
        self.fps = 0.0
        self.started_at = time.time()
        self.last_frame_jpeg = None
        self.detections = []
        self.events = []
        self.examples = []
        self.bin_samples = {}
        self.area_conf_counts = Counter()
        self.class_counts = Counter()
        self.area_counts = Counter()
        self.conf_counts = Counter()
        self.quality_history = deque(maxlen=5000)
        self.frame_quality_history = deque(maxlen=5000)
        self.timing_history = deque(maxlen=600)
        self.track_history = {}
        self.previous_gray = None


app = Flask(__name__)
state = AssessmentState()


def add_example(frame, obj, crop, stage, condition):
    if len(state.examples) >= MAX_EXAMPLES:
        return
    item = {
        "stage": stage,
        "condition": condition,
        "frame_index": state.frame_index,
        "track_id": obj.get("track_id"),
        "conf": obj.get("conf"),
        "area": obj.get("area"),
        "bbox_area_pct": obj.get("bbox_area_pct"),
        "perception_grade": obj.get("perception_grade"),
        "sharpness_score": obj.get("sharpness_score"),
        "contour_clarity_score": obj.get("contour_clarity_score"),
        "track_stability_score": obj.get("track_stability_score"),
        "track_smoothness_score": obj.get("track_smoothness_score"),
        "track_continuity_score": obj.get("track_continuity_score"),
        "bbox_w": obj.get("bbox_w"),
        "bbox_h": obj.get("bbox_h"),
        "size_bin": obj.get("size_bin"),
        "metrics": obj.get("metrics", {}),
        "debug_images": obj.get("debug_images", {}),
        "image": jpeg_data_uri(crop),
        "context": jpeg_data_uri(draw_overlay(frame, [obj]), max_width=360),
    }
    state.examples.append(item)


def add_bin_sample(frame, obj):
    key = f"{obj.get('area_bin')}|{obj.get('conf_bin')}"
    samples = state.bin_samples.setdefault(key, [])
    history = list(state.track_history.get(obj.get("track_id"), []))
    x1, y1, x2, y2 = [float(v) for v in obj.get("bbox", [0, 0, 0, 0])]
    diag = max(1.0, math.hypot(float(frame.shape[1]), float(frame.shape[0])))
    current_center = {
        "frame": state.frame_index,
        "cx": ((x1 + x2) / 2.0 / diag),
        "cy": ((y1 + y2) / 2.0 / diag),
    }
    trail = history[-12:]
    if not trail or trail[-1].get("frame") != state.frame_index:
        trail = trail + [current_center]
    sample = {
        "area_bin": obj.get("area_bin"),
        "conf_bin": obj.get("conf_bin"),
        "frame_index": state.frame_index,
        "track_id": obj.get("track_id"),
        "bbox": obj.get("bbox"),
        "bbox_w": obj.get("bbox_w"),
        "bbox_h": obj.get("bbox_h"),
        "bbox_area_pct": obj.get("bbox_area_pct"),
        "conf": obj.get("conf"),
        "perception_grade": obj.get("perception_grade"),
        "sharpness_score": obj.get("sharpness_score"),
        "contour_clarity_score": obj.get("contour_clarity_score"),
        "track_stability_score": obj.get("track_stability_score"),
        "track_smoothness_score": obj.get("track_smoothness_score"),
        "track_continuity_score": obj.get("track_continuity_score"),
        "track_trail": trail,
        "frame_w": int(frame.shape[1]),
        "frame_h": int(frame.shape[0]),
        "image": jpeg_data_uri(frame, max_width=960),
    }
    if len(samples) < MAX_BIN_SAMPLES:
        samples.append(sample)
        return
    # Keep a small rolling reservoir so long videos still refresh representative samples.
    if random.random() < 0.18:
        samples[random.randrange(len(samples))] = sample


def tracking_quality_metrics(item, frame_w, frame_h):
    if state.analysis_mode != "tracking" or item.get("track_id") is None:
        return {
            "tracking_available": False,
            "track_stability_score": None,
            "track_continuity_score": None,
            "track_smoothness_score": None,
            "track_age_score": None,
            "track_center_jump_pct": None,
        }

    track_id = item.get("track_id")
    x1, y1, x2, y2 = [float(v) for v in item.get("bbox", [0, 0, 0, 0])]
    diag = max(1.0, math.hypot(float(frame_w), float(frame_h)))
    center = ((x1 + x2) / 2.0 / diag, (y1 + y2) / 2.0 / diag)
    history = state.track_history.setdefault(track_id, deque(maxlen=24))
    points = list(history) + [{"frame": state.frame_index, "cx": center[0], "cy": center[1]}]
    hits = float(item.get("hits") or len(points) or 1)
    age_score = min(hits / 12.0, 1.0) * 100.0

    continuity_score = 55.0 if len(points) < 3 else 100.0
    smoothness_score = 55.0 if len(points) < 3 else 100.0
    jump_pct = 0.0
    if len(points) >= 2:
        gaps = [max(1.0, points[i]["frame"] - points[i - 1]["frame"]) for i in range(1, len(points))]
        median_gap = float(np.median(gaps)) if gaps else 1.0
        max_gap = max(gaps) if gaps else 1.0
        continuity_score = max(0.0, min(100.0, 100.0 - max(0.0, max_gap - median_gap) * 18.0))
        velocities = []
        for i in range(1, len(points)):
            gap = max(1.0, points[i]["frame"] - points[i - 1]["frame"])
            velocities.append(
                (
                    (points[i]["cx"] - points[i - 1]["cx"]) / gap,
                    (points[i]["cy"] - points[i - 1]["cy"]) / gap,
                )
            )
        jump_pct = math.hypot(points[-1]["cx"] - points[-2]["cx"], points[-1]["cy"] - points[-2]["cy"]) * 100.0
        if len(velocities) >= 2:
            accel = [
                math.hypot(velocities[i][0] - velocities[i - 1][0], velocities[i][1] - velocities[i - 1][1])
                for i in range(1, len(velocities))
            ]
            accel_p90 = float(np.percentile(accel, 90)) if accel else 0.0
            smoothness_score = max(0.0, min(100.0, 100.0 - accel_p90 * 9500.0))
        else:
            smoothness_score = max(0.0, min(100.0, 100.0 - jump_pct * 3.0))

    stability = age_score * 0.30 + continuity_score * 0.35 + smoothness_score * 0.35
    history.append({"frame": state.frame_index, "cx": center[0], "cy": center[1]})
    return {
        "tracking_available": True,
        "track_stability_score": round(stability, 2),
        "track_continuity_score": round(continuity_score, 2),
        "track_smoothness_score": round(smoothness_score, 2),
        "track_age_score": round(age_score, 2),
        "track_center_jump_pct": round(jump_pct, 4),
    }


def update_stats(frame, tracked, elapsed_ms, temporal=True):
    previous = state.previous_gray if temporal else None
    frame_metrics, gray = frame_quality_metrics(frame, previous)
    state.previous_gray = gray if temporal else None
    state.frame_quality_history.append({"frame": state.frame_index, **frame_metrics})
    enriched = []
    frame_h, frame_w = frame.shape[:2]
    frame_area = max(1, frame_h * frame_w)
    for obj in tracked:
        x1, y1, x2, y2 = obj["bbox"]
        bbox_w = max(0, x2 - x1)
        bbox_h = max(0, y2 - y1)
        area = bbox_w * bbox_h
        bbox_area_pct = (float(area) / float(frame_area)) * 100.0
        metrics, crop = quality_metrics(frame, obj["bbox"])
        item = dict(obj)
        item.update(metrics)
        item.update(frame_metrics)
        item["metrics"] = {**metrics, **frame_metrics}
        item["area"] = int(area)
        item["bbox_area_pct"] = round(bbox_area_pct, 4)
        item["bbox_w"] = int(bbox_w)
        item["bbox_h"] = int(bbox_h)
        item["size_bin"] = size_bin_label(bbox_area_pct)
        item["area_bin"] = item["size_bin"]
        item["conf_bin"] = conf_bin_label(float(item.get("conf", 0.0)))
        track_metrics = tracking_quality_metrics(item, frame_w, frame_h)
        item.update(track_metrics)
        item["perception_grade"] = perception_grade(item)
        item["image"] = jpeg_data_uri(crop)
        item["debug_images"] = edge_debug_images(crop)
        item["context"] = jpeg_data_uri(draw_overlay(frame, [item]), max_width=420)
        enriched.append(item)

        state.area_conf_counts[(item["area_bin"], item["conf_bin"])] += 1
        state.area_counts[item["area_bin"]] += 1
        state.conf_counts[item["conf_bin"]] += 1
        state.class_counts[item["class_name"]] += 1
        add_bin_sample(frame, item)
        state.quality_history.append(
            {
                "frame": state.frame_index,
                "track_id": item["track_id"],
                "class_name": item["class_name"],
                "conf": item["conf"],
                "area": item["area"],
                "bbox_area_pct": item["bbox_area_pct"],
                "bbox_w": item["bbox_w"],
                "bbox_h": item["bbox_h"],
                "size_bin": item["size_bin"],
                "sharpness_score": item["sharpness_score"],
                "contour_clarity_score": item["contour_clarity_score"],
                "tenengrad_score": item["tenengrad_score"],
                "edge_contrast_score": item["edge_contrast_score"],
                "edge_density_score": item["edge_density_score"],
                "edge_density": item["edge_density"],
                "brightness": item["brightness"],
                "contrast": item["contrast"],
                "blur_score": item["blur_score"],
                "track_stability_score": item.get("track_stability_score"),
                "track_continuity_score": item.get("track_continuity_score"),
                "track_smoothness_score": item.get("track_smoothness_score"),
                "track_age_score": item.get("track_age_score"),
                "track_center_jump_pct": item.get("track_center_jump_pct"),
                "perception_grade": item["perception_grade"],
                **frame_metrics,
            }
        )

        if item["sharpness_score"] < 35:
            add_example(frame, item, crop, "Low contour clarity", "sharpness_score < 35")
        elif item["bbox_area_pct"] < 0.5:
            add_example(frame, item, crop, "Small target", "bbox area < 0.5% of frame")
        elif item["conf"] >= 0.7 and item["sharpness_score"] >= 55:
            add_example(frame, item, crop, "Recommended quality", "conf >= 0.70 and sharpness_score >= 55")

    state.detections = enriched
    state.timing_history.append(elapsed_ms)


def perception_grade(item):
    conf = float(item.get("conf", 0.0))
    area_pct = float(item.get("bbox_area_pct", 0.0))
    sharp = float(item.get("sharpness_score", 0.0))
    area_score = min(area_pct / 2.0, 1.0) * 100.0
    track_score = item.get("track_stability_score")
    if track_score is not None:
        score = 0.40 * conf * 100.0 + 0.30 * sharp + 0.20 * float(track_score) + 0.10 * area_score
    else:
        score = 0.45 * conf * 100.0 + 0.40 * sharp + 0.15 * area_score
    if score >= 75:
        return "A"
    if score >= 60:
        return "B"
    if score >= 45:
        return "C"
    return "D"


def avg_metric(rows, key):
    values = [float(row.get(key, 0.0) or 0.0) for row in rows if row.get(key) is not None]
    return round(float(np.mean(values)), 2) if values else 0.0


def numeric_values(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def distribution_summary(rows, metrics):
    output = []
    for key, label in metrics:
        values = numeric_values(rows, key)
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        output.append(
            {
                "key": key,
                "label": label,
                "mean": round(float(np.mean(arr)), 3),
                "std": round(float(np.std(arr)), 3),
                "min": round(float(np.min(arr)), 3),
                "p10": round(float(np.percentile(arr, 10)), 3),
                "p50": round(float(np.percentile(arr, 50)), 3),
                "p90": round(float(np.percentile(arr, 90)), 3),
                "max": round(float(np.max(arr)), 3),
            }
        )
    return output


def grade_distribution(qualities):
    counts = Counter(row.get("perception_grade", "D") for row in qualities)
    total = max(1, len(qualities))
    return [
        {"grade": grade, "count": int(counts.get(grade, 0)), "ratio": round(counts.get(grade, 0) / total, 4)}
        for grade in ["A", "B", "C", "D"]
    ]


def product_spec_indicators(qualities, frame_qualities, source, frame_index):
    by_frame = Counter(row.get("frame") for row in qualities if row.get("frame") is not None)
    by_track = defaultdict(list)
    for row in qualities:
        if row.get("track_id") is not None:
            by_track[row.get("track_id")].append(row)
    stable_tracks = [rows for rows in by_track.values() if len(rows) >= 5]
    stable_samples = [row for rows in stable_tracks for row in rows]
    usable = [
        row
        for row in stable_samples or qualities
        if float(row.get("conf", 0.0) or 0.0) >= 0.5
        and row.get("perception_grade") in {"A", "B"}
        and float(row.get("sharpness_score", 0.0) or 0.0) >= 45
    ]
    bbox_pct_values = numeric_values(usable, "bbox_area_pct")
    stable_ratio = (len(stable_samples) / len(qualities) * 100.0) if qualities and source.get("type") in {"video", "webcam"} else 0.0
    min_trackable_pct = float(np.percentile(bbox_pct_values, 10)) if bbox_pct_values else 0.0
    return {
        "tracking_available": source.get("type") in {"video", "webcam"},
        "stable_track_definition": "track_id with >= 5 sampled bbox hits",
        "stable_track_count": len(stable_tracks) if source.get("type") in {"video", "webcam"} else 0,
        "stable_bbox_sample_count": len(stable_samples) if source.get("type") in {"video", "webcam"} else 0,
        "tracking_stability_score": round(stable_ratio, 2),
        "max_simultaneous_bbox": int(max(by_frame.values())) if by_frame else 0,
        "avg_simultaneous_bbox": round(float(np.mean(list(by_frame.values()))), 2) if by_frame else 0.0,
        "frames_analyzed": int(frame_index or 0),
        "recommended_min_bbox_area_pct": round(min_trackable_pct, 3),
        "avg_confidence": avg_metric(qualities, "conf"),
        "p10_confidence": round(float(np.percentile(numeric_values(qualities, "conf"), 10)), 3) if qualities else 0.0,
        "avg_bbox_area_pct": avg_metric(qualities, "bbox_area_pct"),
        "avg_bbox_clarity": avg_metric(qualities, "sharpness_score"),
        "avg_contour_clarity": avg_metric(qualities, "contour_clarity_score"),
        "avg_track_stability": avg_metric(stable_samples or qualities, "track_stability_score") if source.get("type") in {"video", "webcam"} else 0.0,
        "avg_track_smoothness": avg_metric(stable_samples or qualities, "track_smoothness_score") if source.get("type") in {"video", "webcam"} else 0.0,
        "avg_track_continuity": avg_metric(stable_samples or qualities, "track_continuity_score") if source.get("type") in {"video", "webcam"} else 0.0,
        "avg_frame_lux_proxy": avg_metric(frame_qualities, "frame_lux_proxy"),
        "avg_frame_exposure_score": avg_metric(frame_qualities, "frame_exposure_score"),
        "dynamic_clarity_score": round(max(0.0, 100.0 - avg_metric(frame_qualities, "frame_motion_blur")), 2),
        "spec_note": "Use recommended_min_bbox_area_pct with confidence, contour clarity, exposure, lux proxy, and motion clarity as model operating conditions.",
    }


def build_recommendations(object_samples, qualities, frame_qualities):
    recs = []
    if object_samples < 20:
        recs.append(
            "人物 bbox 樣本數低於 20，統計信心不足；建議改用人物更多、距離更多層級或更長的影片重新評估。"
        )
    elif object_samples < 80:
        recs.append("人物 bbox 樣本數偏少，建議至少累積 80 個 bbox samples 以上再作為客戶報告主結論。")

    avg_conf = avg_metric(qualities, "conf")
    avg_sharp = avg_metric(qualities, "sharpness_score")
    avg_frame_blur = avg_metric(frame_qualities, "frame_motion_blur")
    avg_exposure = avg_metric(frame_qualities, "frame_exposure_score")
    tiny_count = sum(1 for row in qualities if float(row.get("bbox_area_pct", 0.0) or 0.0) < 0.5)
    if qualities and tiny_count / len(qualities) > 0.55:
        recs.append("多數人物 bbox 面積低於 frame 0.5%，模型感知主要受遠距離小目標限制；建議拉近焦段或提高輸入解析度。")
    if qualities and avg_conf < 0.45:
        recs.append("平均 confidence 偏低，建議檢查模型類別定義、場景 domain gap 或降低影像壓縮。")
    if qualities and avg_sharp < 45:
        recs.append("bbox 內輪廓清晰度偏低，建議改善對焦、快門速度或減少壓縮造成的邊緣損失。")
    if frame_qualities and avg_frame_blur > 55:
        recs.append("整體畫面動態模糊偏高，建議提升快門速度、穩定器或降低攝影機移動速度。")
    if frame_qualities and avg_exposure < 55:
        recs.append("曝光條件偏離中間值，建議改善照明或調整 AE，以提高模型可感知的紋理與輪廓。")
    if not recs:
        recs.append("目前樣本量與影像品質足以形成初步模型能力結論；建議再用不同天候/距離/光照條件補強覆蓋率。")
    return recs


def build_tracking_recommendations(source, object_samples):
    if source.get("type") == "images":
        return [
            "目前來源是離散 image files，只評估 detection 與影像品質；不建議解讀 tracking continuity、ID switch 或 ByteTrack 穩定度。",
            "若要評估 ByteTrack 品質，請改用 video 或 webcam 連續影像，並確保同一人物跨多幀出現。",
        ]
    if object_samples < 20:
        return ["連續影像中的人物樣本偏少，ByteTrack 穩定度與 ID 維持品質仍不足以作為結論。"]
    return ["此來源為連續影像，可用 bbox 連續性、track_id 穩定度與 miss/hit 狀態作為 tracking 品質輔助判斷。"]


def image_paths_from_source(source):
    selected = source.get("images") or []
    if not selected:
        selected = list_files(IMAGE_DIR, IMAGE_SUFFIXES)
    paths = []
    for name in selected:
        path = (IMAGE_DIR / str(name)).resolve()
        if IMAGE_DIR.resolve() in path.parents and path.suffix.lower() in IMAGE_SUFFIXES and path.exists():
            paths.append(path)
    return paths


def radar_summary(qualities, frame_qualities):
    avg_conf = avg_metric(qualities, "conf") * 100.0
    avg_bbox_sharp = avg_metric(qualities, "sharpness_score")
    avg_frame_sharp = avg_metric(frame_qualities, "frame_sharpness")
    avg_exposure = avg_metric(frame_qualities, "frame_exposure_score")
    avg_contrast = min(avg_metric(frame_qualities, "frame_contrast") / 70.0, 1.0) * 100.0
    avg_lux = avg_metric(frame_qualities, "frame_lux_proxy")
    motion_stability = max(0.0, 100.0 - avg_metric(frame_qualities, "frame_motion_blur"))
    bbox_size = min(avg_metric(qualities, "bbox_area_pct") / 2.0, 1.0) * 100.0
    return {
        "model_confidence": round(avg_conf, 2),
        "bbox_contour": round(avg_bbox_sharp, 2),
        "bbox_size": round(bbox_size, 2),
        "frame_clarity": round(avg_frame_sharp, 2),
        "exposure": round(avg_exposure, 2),
        "contrast": round(avg_contrast, 2),
        "lux_proxy": round(avg_lux, 2),
        "motion_stability": round(motion_stability, 2),
    }


def quality_radar_summary(frame_qualities):
    avg_sharp = avg_metric(frame_qualities, "frame_sharpness")
    avg_exposure = avg_metric(frame_qualities, "frame_exposure_score")
    avg_contrast = min(avg_metric(frame_qualities, "frame_contrast") / 70.0, 1.0) * 100.0
    avg_lux = avg_metric(frame_qualities, "frame_lux_proxy")
    motion_stability = max(0.0, 100.0 - avg_metric(frame_qualities, "frame_motion_blur"))
    edge_structure = min(avg_metric(frame_qualities, "frame_edge_density") / 0.08, 1.0) * 100.0
    return {
        "frame_clarity": round(avg_sharp, 2),
        "exposure": round(avg_exposure, 2),
        "contrast": round(avg_contrast, 2),
        "lux_proxy": round(avg_lux, 2),
        "motion_stability": round(motion_stability, 2),
        "edge_structure": round(edge_structure, 2),
    }


def model_radar_summary(qualities):
    avg_conf = avg_metric(qualities, "conf") * 100.0
    avg_bbox_sharp = avg_metric(qualities, "sharpness_score")
    bbox_size = min(avg_metric(qualities, "bbox_area_pct") / 2.0, 1.0) * 100.0
    conf_coverage = 0.0
    grade_coverage = 0.0
    low_blur_coverage = 0.0
    edge_coverage = 0.0
    tracking_stability = avg_metric(qualities, "track_stability_score")
    if qualities:
        conf_coverage = sum(1 for row in qualities if float(row.get("conf", 0.0) or 0.0) >= 0.5) / len(qualities) * 100.0
        grade_coverage = sum(1 for row in qualities if row.get("perception_grade") in {"A", "B"}) / len(qualities) * 100.0
        low_blur_coverage = sum(1 for row in qualities if float(row.get("blur_score", 100.0) or 100.0) <= 45) / len(qualities) * 100.0
        edge_coverage = min(avg_metric(qualities, "edge_density") / 0.08, 1.0) * 100.0
    return {
        "confidence_mean": round(avg_conf, 2),
        "confidence_coverage": round(conf_coverage, 2),
        "bbox_contour": round(avg_bbox_sharp, 2),
        "bbox_size": round(bbox_size, 2),
        "low_blur_coverage": round(low_blur_coverage, 2),
        "edge_coverage": round(edge_coverage, 2),
        "ab_grade_coverage": round(grade_coverage, 2),
        "tracking_stability": round(tracking_stability, 2),
    }


def process_image_source(source, conf_threshold):
    paths = image_paths_from_source(source)
    with state.lock:
        state.frame_count = len(paths)
        state.fps = 0.0
    if not paths:
        raise RuntimeError("no image files found in images/")

    for path in paths:
        if state.stop_requested:
            break
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        with state.lock:
            state.frame_index += 1
        t0 = time.time()
        detections = state.runner.infer(frame, conf_threshold)
        detections = [det for det in detections if int(det.get("class_id", -1)) == PERSON_CLASS_ID]
        tracked = []
        for idx, det in enumerate(detections, start=1):
            tracked.append(
                {
                    **det,
                    "track_id": idx,
                    "hits": None,
                    "misses": None,
                    "tracking_available": False,
                    "image_name": path.name,
                }
            )
        elapsed_ms = round((time.time() - t0) * 1000.0, 2)
        overlay = draw_overlay(frame, tracked)
        ok, encoded = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        with state.lock:
            update_stats(frame, tracked, elapsed_ms, temporal=False)
            if ok:
                state.last_frame_jpeg = encoded.tobytes()


def process_source(source, conf_threshold, sample_stride):
    if source["type"] == "images":
        process_image_source(source, conf_threshold)
        return
    if source["type"] == "webcam":
        capture = cv2.VideoCapture(int(source.get("index", 0)))
    else:
        video_path = (VIDEO_DIR / source["name"]).resolve()
        capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("cannot open video source")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    with state.lock:
        state.frame_count = frame_count if frame_count > 0 else None
        state.fps = fps

    while not state.stop_requested:
        ok, frame = capture.read()
        if not ok:
            break
        with state.lock:
            state.frame_index += 1
            frame_index = state.frame_index
        if sample_stride > 1 and frame_index % sample_stride != 0:
            continue

        t0 = time.time()
        detections = state.runner.infer(frame, conf_threshold)
        detections = [det for det in detections if int(det.get("class_id", -1)) == PERSON_CLASS_ID]
        tracked = state.tracker.update(detections)
        elapsed_ms = round((time.time() - t0) * 1000.0, 2)
        overlay = draw_overlay(frame, tracked)
        ok, encoded = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        with state.lock:
            update_stats(frame, tracked, elapsed_ms, temporal=True)
            if ok:
                state.last_frame_jpeg = encoded.tobytes()
        if source["type"] == "webcam":
            time.sleep(0.001)

    capture.release()


def worker(model_name, source, conf_threshold, sample_stride):
    try:
        with state.lock:
            state.reset_job(str(uuid.uuid4()), model_name, source)
        state.runner.load(model_name)
        process_source(source, conf_threshold, sample_stride)
    except Exception as exc:
        with state.lock:
            state.error = str(exc)
    finally:
        with state.lock:
            state.running = False


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/assets")
def assets():
    return jsonify(
        {
            "models": list_files(MODEL_DIR, {".tflite"}),
            "videos": list_files(VIDEO_DIR, {".mkv", ".webm", ".mp4"}),
            "images": list_files(IMAGE_DIR, IMAGE_SUFFIXES),
        }
    )


@app.post("/api/images/download")
def download_image():
    payload = request.json or {}
    url = str(payload.get("url") or "").strip()
    filename = safe_name(payload.get("filename") or Path(url).name or "dataset_image")
    ext = Path(filename).suffix.lower()
    if ext not in IMAGE_SUFFIXES:
        ext = Path(url).suffix.lower()
    if ext not in IMAGE_SUFFIXES:
        ext = ".jpg"
    if not url.startswith(("http://", "https://")):
        return jsonify({"ok": False, "error": "url must start with http:// or https://"}), 400
    IMAGE_DIR.mkdir(exist_ok=True)
    target = (IMAGE_DIR / f"{Path(filename).stem}{ext}").resolve()
    if IMAGE_DIR.resolve() not in target.parents:
        return jsonify({"ok": False, "error": "invalid filename"}), 400
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ModelAssessment/1.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read(20 * 1024 * 1024 + 1)
        if len(data) > 20 * 1024 * 1024:
            return jsonify({"ok": False, "error": "image is larger than 20MB"}), 400
        target.write_bytes(data)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, "file": target.name})


@app.post("/api/load_model")
def load_model():
    model_name = request.json.get("model")
    state.runner.load(model_name)
    with state.lock:
        state.model_name = model_name
    return jsonify({"ok": True, "model": model_name})


@app.post("/api/start")
def start():
    payload = request.json or {}
    model_name = payload.get("model")
    source = payload.get("source") or {}
    conf_threshold = float(payload.get("conf_threshold", DEFAULT_CONF_THRESHOLD))
    sample_stride = max(1, int(payload.get("sample_stride", 1)))
    if not model_name:
        return jsonify({"ok": False, "error": "model is required"}), 400
    if source.get("type") == "video" and source.get("name") not in list_files(VIDEO_DIR, {".mkv", ".webm", ".mp4"}):
        return jsonify({"ok": False, "error": "invalid video"}), 400
    if source.get("type") == "images" and not image_paths_from_source(source):
        return jsonify({"ok": False, "error": "no valid image files selected"}), 400
    if source.get("type") not in {"video", "webcam", "images"}:
        return jsonify({"ok": False, "error": "source type must be video, webcam, or images"}), 400
    with state.lock:
        if state.running:
            return jsonify({"ok": False, "error": "analysis is already running"}), 409
    thread = threading.Thread(target=worker, args=(model_name, source, conf_threshold, sample_stride), daemon=True)
    thread.start()
    return jsonify({"ok": True})


@app.post("/api/stop")
def stop():
    with state.lock:
        state.stop_requested = True
    return jsonify({"ok": True})


@app.get("/api/status")
def status():
    with state.lock:
        elapsed = max(0.0, time.time() - state.started_at) if state.started_at else 0.0
        qualities = list(state.quality_history)
        frame_qualities = list(state.frame_quality_history)
        sharp_values = [x["sharpness_score"] for x in qualities]
        conf_values = [x["conf"] for x in qualities]
        area_values = [x["area"] for x in qualities]
        area_pct_values = [x.get("bbox_area_pct", 0.0) for x in qualities]
        timing = list(state.timing_history)
        matrix = [
            {
                "area_bin": area_label,
                "conf_bin": conf_label,
                "count": count,
            }
            for (area_label, conf_label), count in state.area_conf_counts.items()
        ]
        return jsonify(
            {
                "job_id": state.job_id,
                "running": state.running,
                "error": state.error,
                "model": state.model_name,
                "source": state.source,
                "analysis_mode": state.analysis_mode,
                "tracking_available": state.analysis_mode == "tracking",
                "frame_index": state.frame_index,
                "frame_count": state.frame_count,
                "fps": state.fps,
                "elapsed_sec": round(elapsed, 2),
                "object_samples": len(qualities),
                "current_objects": state.detections,
                "quality_samples": qualities,
                "frame_quality_samples": frame_qualities,
                "area_counts": dict(state.area_counts),
                "conf_counts": dict(state.conf_counts),
                "class_counts": dict(state.class_counts),
                "area_conf_matrix": matrix,
                "examples": state.examples,
                "recommendations": build_recommendations(len(qualities), qualities, frame_qualities),
                "tracking_recommendations": build_tracking_recommendations(state.source, len(qualities)),
                "grade_distribution": grade_distribution(qualities),
                "product_spec": product_spec_indicators(qualities, frame_qualities, state.source, state.frame_index),
                "metric_distribution": distribution_summary(
                    qualities,
                    [
                        ("conf", "Confidence"),
                        ("bbox_area_pct", "BBox frame area %"),
                        ("sharpness_score", "BBox sharpness"),
                        ("contour_clarity_score", "Contour clarity"),
                        ("tenengrad_score", "Tenengrad"),
                        ("edge_contrast_score", "Edge contrast"),
                        ("track_stability_score", "Track stability"),
                        ("track_smoothness_score", "Track smoothness"),
                        ("track_continuity_score", "Track continuity"),
                        ("blur_score", "BBox blur"),
                        ("frame_motion_blur", "Dynamic blur"),
                        ("frame_lux_proxy", "Lux proxy"),
                    ],
                ),
                "radar": radar_summary(qualities, frame_qualities),
                "quality_radar": quality_radar_summary(frame_qualities),
                "model_radar": model_radar_summary(qualities),
                "frame_quality": {
                    "frame_sharpness": avg_metric(frame_qualities, "frame_sharpness"),
                    "frame_brightness": avg_metric(frame_qualities, "frame_brightness"),
                    "frame_exposure_score": avg_metric(frame_qualities, "frame_exposure_score"),
                    "frame_motion_blur": avg_metric(frame_qualities, "frame_motion_blur"),
                    "frame_contrast": avg_metric(frame_qualities, "frame_contrast"),
                    "frame_lux_proxy": avg_metric(frame_qualities, "frame_lux_proxy"),
                    "frame_quality_score": avg_metric(frame_qualities, "frame_quality_score"),
                },
                "summary": {
                    "avg_conf": round(float(np.mean(conf_values)), 4) if conf_values else 0.0,
                    "avg_area": round(float(np.mean(area_values)), 2) if area_values else 0.0,
                    "avg_bbox_area_pct": round(float(np.mean(area_pct_values)), 4) if area_pct_values else 0.0,
                    "avg_sharpness": round(float(np.mean(sharp_values)), 2) if sharp_values else 0.0,
                    "p10_sharpness": round(float(np.percentile(sharp_values, 10)), 2) if sharp_values else 0.0,
                    "avg_inference_ms": round(float(np.mean(timing)), 2) if timing else 0.0,
                    "recommended_ratio": round(
                        sum(1 for x in qualities if x["perception_grade"] in {"A", "B"}) / len(qualities),
                        4,
                    )
                    if qualities
                    else 0.0,
                },
            }
        )


@app.post("/api/bin_sample")
def bin_sample():
    payload = request.json or {}
    area_bin = str(payload.get("area_bin") or "").strip()
    conf_bin = str(payload.get("conf_bin") or "").strip()
    key = f"{area_bin}|{conf_bin}"
    with state.lock:
        samples = list(state.bin_samples.get(key, []))
    if not samples:
        return jsonify({"ok": False, "error": "no sample for selected bin"}), 404
    return jsonify({"ok": True, "sample": random.choice(samples)})


@app.get("/stream")
def stream():
    def generate():
        while True:
            with state.lock:
                frame = state.last_frame_jpeg
                running = state.running
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            if not running and frame is None:
                time.sleep(0.2)
            time.sleep(0.08)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


def pyplot_modules():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = [
        "Noto Sans CJK TC",
        "Noto Sans CJK JP",
        "Noto Sans CJK KR",
        "Noto Sans CJK SC",
        "Noto Sans CJK HK",
        "DejaVu Sans",
        "Arial Unicode MS",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    try:
        import pandas as pd
        import seaborn as sns
    except Exception:
        pd = None
        sns = None
    return plt, pd, sns


def fig_to_data_uri(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    buffer.seek(0)
    return "data:image/png;base64," + base64.b64encode(buffer.read()).decode("ascii")


REPORT_CHART_TEXT = {
    "zh": {
        "heatmap_title": "BBox Frame 佔比 / Confidence 矩陣",
        "heatmap_x": "BBox 面積佔整張 frame 比例",
        "heatmap_y": "Confidence 區間",
        "count": "bbox 數量",
        "quality_radar": "影像品質雷達圖",
        "model_radar": "模型感知雷達圖",
        "grade_chart": "A/B/C/D 感知評級分佈",
        "violin_chart": "指標分佈 Violin 圖",
        "score_percent": "分數 / 百分比",
        "quality_labels": ["清晰", "曝光", "對比", "Lux", "穩定", "邊緣"],
        "model_labels": ["Conf", "Conf覆蓋", "輪廓", "Tracking", "BBox %", "低模糊", "邊緣", "A/B"],
        "no_samples": "目前沒有 bbox 樣本",
    },
    "en": {
        "heatmap_title": "BBox Frame Area % / Confidence Matrix",
        "heatmap_x": "BBox area ratio of frame",
        "heatmap_y": "Confidence bin",
        "count": "bbox count",
        "quality_radar": "Image Quality Radar",
        "model_radar": "Model Perception Radar",
        "grade_chart": "A/B/C/D Perception Grade Distribution",
        "violin_chart": "Metric Distribution Violin",
        "score_percent": "Score / percent",
        "quality_labels": ["Clarity", "Exposure", "Contrast", "Lux", "Stability", "Edges"],
        "model_labels": ["Conf", "Conf cover", "Contour", "Tracking", "BBox %", "Low blur", "Edges", "A/B"],
        "no_samples": "No bbox samples",
    },
    "ja": {
        "heatmap_title": "BBox Frame 比率 / Confidence 行列",
        "heatmap_x": "BBox 面積の frame 全体比率",
        "heatmap_y": "Confidence 区間",
        "count": "bbox 数",
        "quality_radar": "画質レーダー",
        "model_radar": "モデル認識レーダー",
        "grade_chart": "A/B/C/D 評価分布",
        "violin_chart": "指標分布 Violin",
        "score_percent": "スコア / パーセント",
        "quality_labels": ["明瞭", "露出", "対比", "Lux", "安定", "輪郭"],
        "model_labels": ["Conf", "Conf範囲", "輪郭", "Tracking", "BBox %", "低ぼけ", "輪郭", "A/B"],
        "no_samples": "bbox サンプルがありません",
    },
    "ko": {
        "heatmap_title": "BBox Frame 비율 / Confidence 매트릭스",
        "heatmap_x": "BBox 면적의 전체 frame 대비 비율",
        "heatmap_y": "Confidence 구간",
        "count": "bbox 수",
        "quality_radar": "이미지 품질 레이더",
        "model_radar": "모델 인식 레이더",
        "grade_chart": "A/B/C/D 인식 등급 분포",
        "violin_chart": "지표 분포 Violin",
        "score_percent": "점수 / 퍼센트",
        "quality_labels": ["선명", "노출", "대비", "Lux", "안정", "엣지"],
        "model_labels": ["Conf", "Conf범위", "윤곽", "Tracking", "BBox %", "저블러", "엣지", "A/B"],
        "no_samples": "bbox 샘플이 없습니다",
    },
}


def chart_text(language):
    return REPORT_CHART_TEXT.get(language, REPORT_CHART_TEXT["en"])


def render_heatmap_chart(snapshot, language="en"):
    text = chart_text(language)
    plt, pd, sns = pyplot_modules()
    labels = [item["label"] for item in SIZE_BINS]
    conf_labels = [f"{lo:.2f}-{hi:.2f}" for lo, hi in CONF_BINS]
    matrix = np.zeros((len(conf_labels), len(labels)), dtype=np.float64)
    for item in snapshot.get("area_conf_matrix", []):
        if item.get("area_bin") in labels and item.get("conf_bin") in conf_labels:
            row = conf_labels.index(item["conf_bin"])
            col = labels.index(item["area_bin"])
            matrix[row, col] = float(item.get("count", 0) or 0)
    fig, ax = plt.subplots(figsize=(8.8, 4.8), facecolor="#0b1220")
    ax.set_facecolor("#0f172a")
    if sns is not None:
        sns.heatmap(
            matrix[::-1],
            annot=True,
            fmt=".0f",
            cmap="mako",
            cbar_kws={"label": text["count"]},
            xticklabels=labels,
            yticklabels=conf_labels[::-1],
            linewidths=0.8,
            linecolor="#1e293b",
            ax=ax,
        )
    else:
        image = ax.imshow(matrix[::-1], cmap="mako")
        fig.colorbar(image, ax=ax, label=text["count"])
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(conf_labels)), conf_labels[::-1])
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                ax.text(x, y, int(matrix[::-1][y, x]), ha="center", va="center", color="white")
    ax.set_title(text["heatmap_title"], color="#e5eefc", weight="bold")
    ax.set_xlabel(text["heatmap_x"], color="#cbd5e1")
    ax.set_ylabel(text["heatmap_y"], color="#cbd5e1")
    ax.tick_params(colors="#cbd5e1")
    fig.tight_layout()
    uri = fig_to_data_uri(fig)
    plt.close(fig)
    return uri


def render_radar_chart(title, labels, values):
    plt, _, _ = pyplot_modules()
    values = [max(0.0, min(100.0, float(v or 0.0))) for v in values]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values_closed = values + values[:1]
    angles_closed = angles + angles[:1]
    fig = plt.figure(figsize=(5.4, 5.0), facecolor="#0b1220")
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor("#0f172a")
    ax.plot(angles_closed, values_closed, color="#f59e0b", linewidth=2)
    ax.fill(angles_closed, values_closed, color="#2563eb", alpha=0.32)
    ax.set_xticks(angles, labels)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], color="#94a3b8")
    ax.tick_params(colors="#dbeafe")
    ax.grid(color="#334155", alpha=0.72)
    ax.spines["polar"].set_color("#475569")
    ax.set_title(title, color="#e5eefc", pad=18, weight="bold")
    uri = fig_to_data_uri(fig)
    plt.close(fig)
    return uri


def render_grade_chart(snapshot, language="en"):
    text = chart_text(language)
    plt, _, _ = pyplot_modules()
    grades = snapshot.get("grade_distribution", [])
    labels = [row["grade"] for row in grades]
    counts = [row["count"] for row in grades]
    colors = ["#16a34a", "#2563eb", "#d97706", "#dc2626"]
    fig, ax = plt.subplots(figsize=(6.2, 3.8), facecolor="#0b1220")
    ax.set_facecolor("#0f172a")
    ax.bar(labels, counts, color=colors, width=0.62)
    for idx, row in enumerate(grades):
        ax.text(idx, row["count"] + 0.4, f'{row["count"]} / {row["ratio"]*100:.1f}%', ha="center", color="#e5eefc")
    ax.set_title(text["grade_chart"], color="#e5eefc", weight="bold")
    ax.set_ylabel(text["count"], color="#cbd5e1")
    ax.tick_params(colors="#cbd5e1")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")
    fig.tight_layout()
    uri = fig_to_data_uri(fig)
    plt.close(fig)
    return uri


def render_violin_chart(snapshot, language="en"):
    text = chart_text(language)
    plt, pd, sns = pyplot_modules()
    rows = snapshot.get("quality_samples", []) or []
    keys = [
        ("conf", "Conf"),
        ("bbox_area_pct", "BBox %"),
        ("sharpness_score", "Sharp"),
        ("contour_clarity_score", "Contour"),
        ("frame_lux_proxy", "Lux"),
        ("frame_motion_blur", "Motion blur"),
    ]
    long_rows = []
    for key, label in keys:
        values = numeric_values(rows, key)
        if key == "conf":
            values = [v * 100.0 for v in values]
        for value in values:
            long_rows.append({"metric": label, "value": value})
    fig, ax = plt.subplots(figsize=(8.8, 4.8), facecolor="#0b1220")
    ax.set_facecolor("#0f172a")
    if long_rows and sns is not None and pd is not None:
        df = pd.DataFrame(long_rows)
        sns.violinplot(data=df, x="metric", y="value", inner="quartile", color="#3b82f6", linewidth=1, ax=ax)
    elif long_rows:
        grouped = [[row["value"] for row in long_rows if row["metric"] == label] for _, label in keys]
        ax.violinplot(grouped, showmeans=True, showmedians=True)
        ax.set_xticks(range(1, len(keys) + 1), [label for _, label in keys])
    else:
        ax.text(0.5, 0.5, text["no_samples"], ha="center", va="center", color="#e5eefc", transform=ax.transAxes)
    ax.set_title(text["violin_chart"], color="#e5eefc", weight="bold")
    ax.set_ylabel(text["score_percent"], color="#cbd5e1")
    ax.tick_params(colors="#cbd5e1")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")
    fig.tight_layout()
    uri = fig_to_data_uri(fig)
    plt.close(fig)
    return uri


def build_report_charts(snapshot, language="en"):
    text = chart_text(language)
    quality = snapshot.get("quality_radar", {})
    model = snapshot.get("model_radar", {})
    return {
        "heatmap": render_heatmap_chart(snapshot, language),
        "quality_radar": render_radar_chart(
            text["quality_radar"],
            text["quality_labels"],
            [
                quality.get("frame_clarity"),
                quality.get("exposure"),
                quality.get("contrast"),
                quality.get("lux_proxy"),
                quality.get("motion_stability"),
                quality.get("edge_structure"),
            ],
        ),
        "model_radar": render_radar_chart(
            text["model_radar"],
            text["model_labels"],
            [
                model.get("confidence_mean"),
                model.get("confidence_coverage"),
                model.get("bbox_contour"),
                model.get("tracking_stability"),
                model.get("bbox_size"),
                model.get("low_blur_coverage"),
                model.get("edge_coverage"),
                model.get("ab_grade_coverage"),
            ],
        ),
        "grades": render_grade_chart(snapshot, language),
        "violin": render_violin_chart(snapshot, language),
    }


@app.post("/api/report")
def report():
    payload = request.json or {}
    name = safe_name(payload.get("name") or "yolo_assessment")
    tags = safe_name(payload.get("tags") or "")
    language = str(payload.get("language") or "zh").strip().lower()
    if language not in {"zh", "en", "ja", "ko"}:
        language = "zh"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_id = f"{stamp}_{name}" + (f"_{tags}" if tags else "")
    data_path = REPORT_DIR / f"{report_id}.json"
    html_path = REPORT_DIR / f"{report_id}.html"
    REPORT_DIR.mkdir(exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = status().get_json()
    snapshot["report"] = {
        "id": report_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "language": language,
        "metric_definitions": metric_definitions(language),
    }
    snapshot["report"]["charts"] = build_report_charts(snapshot, language)
    data_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    html = render_report_html(snapshot, language)
    html_path.write_text(html, encoding="utf-8")
    download_json = DOWNLOAD_DIR / data_path.name
    download_html = DOWNLOAD_DIR / html_path.name
    download_json.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    download_html.write_text(html, encoding="utf-8")
    return jsonify(
        {
            "ok": True,
            "json": data_path.name,
            "html": html_path.name,
            "download_json": str(download_json),
            "download_html": str(download_html),
        }
    )


@app.get("/reports/<path:name>")
def get_report(name):
    path = (REPORT_DIR / name).resolve()
    if REPORT_DIR.resolve() not in path.parents or not path.exists():
        return jsonify({"error": "not found"}), 404
    return send_file(path, as_attachment=request.args.get("download") == "1")


def metric_definitions(language="en"):
    definitions = {
        "zh": {
            "bbox_area_pct": "bbox 面積除以整張 frame 面積後的百分比，用於跨解析度比較。",
            "bbox_size_bin": "bbox frame 佔比分級：<0.15%, 0.15-0.25%, 0.25-0.5%, 0.5-1%, 1-2%, 2-4%, 4-8%, >=8%。",
            "sharpness_score": "嚴格 0-100 輪廓清晰度，綜合 Laplacian、Tenengrad、Sobel、Canny、edge contrast、bbox contrast 與曝光懲罰。",
            "contour_clarity_score": "bbox 內輪廓品質輔助分數，結合 sharpness、edge density、edge contrast 與局部對比。",
            "track_stability_score": "僅 video/webcam 使用，整合 track age、frame continuity、中心點移動平滑度；ID 斷掉或軌跡突跳會降低分數。",
            "track_smoothness_score": "追蹤中心點速度/加速度的平滑度，越高代表軌跡越連續。",
            "frame_quality": "整張畫面的清晰度、亮度、曝光、動態模糊、對比與 lux proxy。",
            "lux_proxy": "相對照度 proxy，非校正後的物理 lux。",
            "perception_grade": "Video/Webcam: 40% confidence + 30% bbox 清晰度 + 20% tracking 穩定 + 10% bbox size。Images: 45% confidence + 40% bbox 清晰度 + 15% bbox size。",
            "recommended_quality": "A/B 等級或 conf >= 0.70 且 strict sharpness_score >= 55 可視為推薦品質樣本。",
        },
        "en": {
            "bbox_area_pct": "bbox area divided by whole frame area, expressed as percent for cross-resolution comparison.",
            "bbox_size_bin": "bbox frame-ratio bins: <0.15%, 0.15-0.25%, 0.25-0.5%, 0.5-1%, 1-2%, 2-4%, 4-8%, >=8%.",
            "sharpness_score": "Strict 0-100 contour score from Laplacian, Tenengrad, Sobel, Canny, edge contrast, bbox contrast, and exposure penalty.",
            "contour_clarity_score": "Secondary bbox contour quality score combining sharpness, edge density, edge contrast, and local contrast.",
            "track_stability_score": "Video/webcam only: combines track age, frame continuity, and center-path smoothness. ID drops or path jumps reduce the score.",
            "track_smoothness_score": "Smoothness of tracked center velocity/acceleration. Higher means a more continuous trajectory.",
            "frame_quality": "Whole-frame clarity, brightness, exposure, dynamic blur, contrast, and lux proxy.",
            "lux_proxy": "Relative brightness/contrast proxy, not calibrated physical lux.",
            "perception_grade": "Video/Webcam: 40% confidence + 30% bbox clarity + 20% tracking stability + 10% bbox size. Images: 45% confidence + 40% bbox clarity + 15% bbox size.",
            "recommended_quality": "Recommended when grade is A/B, or conf >= 0.70 and strict sharpness_score >= 55.",
        },
        "ja": {
            "bbox_area_pct": "bbox 面積を frame 全体面積で割った百分率。解像度をまたいだ比較に使います。",
            "bbox_size_bin": "bbox frame 比率区間：<0.15%, 0.15-0.25%, 0.25-0.5%, 0.5-1%, 1-2%, 2-4%, 4-8%, >=8%。",
            "sharpness_score": "Laplacian、Tenengrad、Sobel、Canny、edge contrast、bbox contrast、露出ペナルティから算出する厳格な 0-100 輪郭明瞭度。",
            "contour_clarity_score": "sharpness、edge density、edge contrast、局所対比を組み合わせた bbox 輪郭品質補助スコア。",
            "track_stability_score": "video/webcam のみ。track age、frame continuity、中心点軌跡の滑らかさを統合し、ID 途切れや急な折れ線で低下します。",
            "track_smoothness_score": "追跡中心点の速度/加速度の滑らかさ。高いほど軌跡が連続的です。",
            "frame_quality": "フレーム全体の明瞭度、明るさ、露出、動的ぼけ、対比、lux proxy。",
            "lux_proxy": "相対照度 proxy。校正済み物理 lux ではありません。",
            "perception_grade": "Video/Webcam: 40% confidence + 30% bbox 明瞭度 + 20% tracking 安定性 + 10% bbox サイズ。Images: 45% confidence + 40% bbox 明瞭度 + 15% bbox サイズ。",
            "recommended_quality": "A/B 評価、または conf >= 0.70 かつ strict sharpness_score >= 55 を推奨品質サンプルとします。",
        },
        "ko": {
            "bbox_area_pct": "bbox 면적을 전체 frame 면적으로 나눈 백분율이며 해상도 간 비교에 사용합니다.",
            "bbox_size_bin": "bbox frame 비율 구간: <0.15%, 0.15-0.25%, 0.25-0.5%, 0.5-1%, 1-2%, 2-4%, 4-8%, >=8%.",
            "sharpness_score": "Laplacian, Tenengrad, Sobel, Canny, edge contrast, bbox contrast, 노출 패널티를 결합한 엄격한 0-100 윤곽 선명도.",
            "contour_clarity_score": "sharpness, edge density, edge contrast, local contrast를 결합한 bbox 윤곽 품질 보조 점수.",
            "track_stability_score": "video/webcam 전용. track age, frame continuity, 중심점 경로 smoothness를 결합하며 ID 끊김이나 경로 급변 시 낮아집니다.",
            "track_smoothness_score": "추적 중심점 속도/가속도의 부드러움. 높을수록 궤적이 연속적입니다.",
            "frame_quality": "전체 frame 선명도, 밝기, 노출, 동적 blur, 대비, lux proxy.",
            "lux_proxy": "상대 조도 proxy이며 보정된 물리 lux가 아닙니다.",
            "perception_grade": "Video/Webcam: 40% confidence + 30% bbox 선명도 + 20% tracking 안정성 + 10% bbox 크기. Images: 45% confidence + 40% bbox 선명도 + 15% bbox 크기.",
            "recommended_quality": "A/B 등급 또는 conf >= 0.70 및 strict sharpness_score >= 55이면 권장 품질 샘플입니다.",
        },
    }
    return definitions.get(language, definitions["en"])


REPORT_TEXT = {
    "zh": {
        "title": "YOLO 推論評估報告",
        "model": "模型",
        "source": "來源",
        "object_samples": "物件樣本",
        "avg_confidence": "平均 Confidence",
        "avg_bbox_area": "平均 BBox 面積",
        "avg_sharpness": "平均清晰度",
        "metric_definitions": "指標定義",
        "current_bbox_samples": "目前 BBox 樣本",
        "track": "Track",
        "conf": "Conf",
        "area": "Area",
        "sharpness": "Sharpness",
        "grade": "評級",
    },
    "en": {
        "title": "YOLO Inference Assessment Report",
        "model": "Model",
        "source": "Source",
        "object_samples": "Object Samples",
        "avg_confidence": "Avg Confidence",
        "avg_bbox_area": "Avg BBox Area",
        "avg_sharpness": "Avg Sharpness",
        "metric_definitions": "Metric Definitions",
        "current_bbox_samples": "Current BBox Samples",
        "track": "Track",
        "conf": "Conf",
        "area": "Area",
        "sharpness": "Sharpness",
        "grade": "Grade",
    },
    "ja": {
        "title": "YOLO 推論評価レポート",
        "model": "モデル",
        "source": "ソース",
        "object_samples": "物体サンプル",
        "avg_confidence": "平均 Confidence",
        "avg_bbox_area": "平均 BBox 面積",
        "avg_sharpness": "平均明瞭度",
        "metric_definitions": "指標定義",
        "current_bbox_samples": "現在の BBox サンプル",
        "track": "Track",
        "conf": "Conf",
        "area": "Area",
        "sharpness": "Sharpness",
        "grade": "評価",
    },
    "ko": {
        "title": "YOLO 추론 평가 보고서",
        "model": "모델",
        "source": "소스",
        "object_samples": "객체 샘플",
        "avg_confidence": "평균 Confidence",
        "avg_bbox_area": "평균 BBox 면적",
        "avg_sharpness": "평균 선명도",
        "metric_definitions": "지표 정의",
        "current_bbox_samples": "현재 BBox 샘플",
        "track": "Track",
        "conf": "Conf",
        "area": "Area",
        "sharpness": "Sharpness",
        "grade": "등급",
    },
}


REPORT_LABELS = {
    "zh": {
        "avg_bbox_frame": "平均 BBox Frame %",
        "product_spec": "產品模型規格指標",
        "product_spec_note": "以下數值可用於產品/模型操作規格：穩定 bbox 能力、tracking 穩定性、環境光源、動態清晰度與最低可用 bbox frame 佔比。",
        "charts": "圖表",
        "representative_sample": "代表性真實樣本",
        "condition": "條件",
        "metrics": "指標",
        "metric_stats": "指標平均 / 標準差 / 百分位",
        "metric": "指標",
        "mean": "平均",
        "std": "標準差",
        "grade_ratio": "A/B/C/D 數量比例",
        "count": "數量",
        "ratio": "比例",
        "bbox_frame_pct": "BBox frame %",
        "contour": "輪廓",
    },
    "en": {
        "avg_bbox_frame": "Avg BBox Frame %",
        "product_spec": "Product Model Spec Indicators",
        "product_spec_note": "These values are intended for product/model operating spec: stable bbox capacity, tracking stability, environmental light, dynamic clarity, and minimum bbox frame ratio.",
        "charts": "Charts",
        "representative_sample": "Representative Real Sample",
        "condition": "Condition",
        "metrics": "Metrics",
        "metric_stats": "Metric Average / Std / Percentiles",
        "metric": "Metric",
        "mean": "Mean",
        "std": "Std",
        "grade_ratio": "A/B/C/D Count Ratio",
        "count": "Count",
        "ratio": "Ratio",
        "bbox_frame_pct": "BBox frame %",
        "contour": "Contour",
    },
    "ja": {
        "avg_bbox_frame": "平均 BBox Frame %",
        "product_spec": "製品モデル仕様指標",
        "product_spec_note": "以下は製品/モデル運用仕様向けの値です：安定 bbox 能力、tracking 安定性、環境光、動的明瞭度、最低 bbox frame 比率。",
        "charts": "図表",
        "representative_sample": "代表実サンプル",
        "condition": "条件",
        "metrics": "指標",
        "metric_stats": "指標平均 / 標準偏差 / 百分位",
        "metric": "指標",
        "mean": "平均",
        "std": "標準偏差",
        "grade_ratio": "A/B/C/D 数量比率",
        "count": "数",
        "ratio": "比率",
        "bbox_frame_pct": "BBox frame %",
        "contour": "輪郭",
    },
    "ko": {
        "avg_bbox_frame": "평균 BBox Frame %",
        "product_spec": "제품 모델 사양 지표",
        "product_spec_note": "아래 값은 제품/모델 운용 사양용입니다: 안정 bbox 용량, tracking 안정성, 환경 광원, 동적 선명도, 최소 bbox frame 비율.",
        "charts": "차트",
        "representative_sample": "대표 실제 샘플",
        "condition": "조건",
        "metrics": "지표",
        "metric_stats": "지표 평균 / 표준편차 / 분위",
        "metric": "지표",
        "mean": "평균",
        "std": "표준편차",
        "grade_ratio": "A/B/C/D 수량 비율",
        "count": "수량",
        "ratio": "비율",
        "bbox_frame_pct": "BBox frame %",
        "contour": "윤곽",
    },
}


PRODUCT_SPEC_LABELS = {
    "tracking_available": {"zh": "是否可評估 tracking", "en": "Tracking available", "ja": "tracking 評価可否", "ko": "tracking 평가 가능"},
    "stable_track_definition": {"zh": "穩定 track 定義", "en": "Stable track definition", "ja": "安定 track 定義", "ko": "안정 track 정의"},
    "stable_track_count": {"zh": "穩定 track 數量", "en": "Stable track count", "ja": "安定 track 数", "ko": "안정 track 수"},
    "stable_bbox_sample_count": {"zh": "穩定 bbox 樣本數", "en": "Stable bbox samples", "ja": "安定 bbox サンプル数", "ko": "안정 bbox 샘플 수"},
    "tracking_stability_score": {"zh": "整體 tracking 穩定分數", "en": "Overall tracking stability", "ja": "全体 tracking 安定度", "ko": "전체 tracking 안정도"},
    "avg_track_stability": {"zh": "平均 track 穩定度", "en": "Avg track stability", "ja": "平均 track 安定度", "ko": "평균 track 안정도"},
    "avg_track_smoothness": {"zh": "平均軌跡平滑度", "en": "Avg track smoothness", "ja": "平均軌跡滑らかさ", "ko": "평균 궤적 smoothness"},
    "avg_track_continuity": {"zh": "平均 track 連續度", "en": "Avg track continuity", "ja": "平均 track 連続度", "ko": "평균 track 연속도"},
    "max_simultaneous_bbox": {"zh": "最大同時 bbox 數", "en": "Max simultaneous bbox", "ja": "最大同時 bbox 数", "ko": "최대 동시 bbox 수"},
    "avg_simultaneous_bbox": {"zh": "平均同時 bbox 數", "en": "Avg simultaneous bbox", "ja": "平均同時 bbox 数", "ko": "평균 동시 bbox 수"},
    "recommended_min_bbox_area_pct": {"zh": "建議最低 bbox frame %", "en": "Recommended minimum bbox frame %", "ja": "推奨最低 bbox frame %", "ko": "권장 최소 bbox frame %"},
    "avg_frame_lux_proxy": {"zh": "平均 lux proxy", "en": "Avg lux proxy", "ja": "平均 lux proxy", "ko": "평균 lux proxy"},
    "avg_frame_exposure_score": {"zh": "平均曝光分數", "en": "Avg exposure score", "ja": "平均露出スコア", "ko": "평균 노출 점수"},
    "dynamic_clarity_score": {"zh": "動態清晰度", "en": "Dynamic clarity", "ja": "動的明瞭度", "ko": "동적 선명도"},
    "spec_note": {"zh": "規格備註", "en": "Spec note", "ja": "仕様メモ", "ko": "사양 메모"},
}


def report_label(language, key):
    return REPORT_LABELS.get(language, REPORT_LABELS["en"]).get(key, key)


def product_spec_label(language, key):
    return PRODUCT_SPEC_LABELS.get(key, {}).get(language, key)


METRIC_LABELS = {
    "conf": {"zh": "Confidence", "en": "Confidence", "ja": "Confidence", "ko": "Confidence"},
    "bbox_area_pct": {"zh": "BBox frame %", "en": "BBox frame %", "ja": "BBox frame %", "ko": "BBox frame %"},
    "sharpness_score": {"zh": "BBox 清晰度", "en": "BBox sharpness", "ja": "BBox 明瞭度", "ko": "BBox 선명도"},
    "contour_clarity_score": {"zh": "輪廓清晰度", "en": "Contour clarity", "ja": "輪郭明瞭度", "ko": "윤곽 선명도"},
    "tenengrad_score": {"zh": "Tenengrad", "en": "Tenengrad", "ja": "Tenengrad", "ko": "Tenengrad"},
    "edge_contrast_score": {"zh": "邊緣對比", "en": "Edge contrast", "ja": "エッジ対比", "ko": "엣지 대비"},
    "track_stability_score": {"zh": "Tracking 穩定度", "en": "Track stability", "ja": "Tracking 安定度", "ko": "Tracking 안정도"},
    "track_smoothness_score": {"zh": "軌跡平滑度", "en": "Track smoothness", "ja": "軌跡滑らかさ", "ko": "궤적 smoothness"},
    "track_continuity_score": {"zh": "Track 連續度", "en": "Track continuity", "ja": "Track 連続度", "ko": "Track 연속도"},
    "blur_score": {"zh": "BBox 模糊", "en": "BBox blur", "ja": "BBox ぼけ", "ko": "BBox blur"},
    "frame_motion_blur": {"zh": "動態模糊", "en": "Dynamic blur", "ja": "動的ぼけ", "ko": "동적 blur"},
    "frame_lux_proxy": {"zh": "Lux proxy", "en": "Lux proxy", "ja": "Lux proxy", "ko": "Lux proxy"},
}


def metric_label(language, row):
    key = row.get("key")
    return METRIC_LABELS.get(key, {}).get(language, row.get("label", key))


def render_report_html(snapshot, language="zh"):
    text = REPORT_TEXT.get(language, REPORT_TEXT["zh"])
    labels = REPORT_LABELS.get(language, REPORT_LABELS["en"])
    summary = snapshot["summary"]
    spec = snapshot.get("product_spec", {})
    charts = snapshot.get("report", {}).get("charts", {})
    sample = (snapshot.get("examples") or snapshot.get("current_objects") or [{}])[0]
    metric_rows = "\n".join(
        f"<tr><td>{metric_label(language, row)}</td><td>{row['mean']}</td><td>{row['std']}</td><td>{row['p10']}</td><td>{row['p50']}</td><td>{row['p90']}</td></tr>"
        for row in snapshot.get("metric_distribution", [])
    )
    grade_rows = "\n".join(
        f"<tr><td>{row['grade']}</td><td>{row['count']}</td><td>{row['ratio'] * 100:.1f}%</td></tr>"
        for row in snapshot.get("grade_distribution", [])
    )
    spec_rows = "\n".join(
        f"<tr><td>{product_spec_label(language, key)}</td><td>{value}</td></tr>"
        for key, value in spec.items()
    )
    rows = "\n".join(
        f"<tr><td>{obj.get('track_id')}</td><td>{obj.get('conf')}</td><td>{obj.get('bbox_area_pct')}%</td>"
        f"<td>{obj.get('sharpness_score')}</td><td>{obj.get('contour_clarity_score')}</td><td>{obj.get('perception_grade')}</td></tr>"
        for obj in snapshot.get("current_objects", [])
    )
    return f"""<!doctype html>
<html lang="{language}">
<head>
  <meta charset="utf-8">
  <title>{text["title"]}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; color: #e5eefc; background: #08111f; line-height: 1.55; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 30px; }}
    h1, h2 {{ margin: 0 0 10px; }}
    h1 {{ font-size: 30px; }}
    h2 {{ font-size: 19px; margin-top: 26px; color: #f8fafc; }}
    .meta {{ color: #9fb0cc; margin-bottom: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
    .card {{ background: #101b2e; border: 1px solid rgba(148,163,184,.24); border-radius: 8px; padding: 14px; }}
    .value {{ font-size: 28px; font-weight: 700; color: #f59e0b; }}
    .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .wide {{ grid-column: 1 / -1; }}
    img.chart {{ width: 100%; border: 1px solid rgba(148,163,184,.24); border-radius: 8px; background: #0b1220; }}
    .sample {{ display: grid; grid-template-columns: minmax(260px, 420px) 1fr; gap: 16px; align-items: start; }}
    .sample img {{ width: 100%; border-radius: 8px; border: 1px solid rgba(148,163,184,.24); }}
    table {{ width: 100%; border-collapse: collapse; background: #101b2e; margin-top: 12px; font-size: 13px; }}
    th, td {{ border: 1px solid rgba(148,163,184,.22); padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #13233a; color: #c7d8f6; }}
    pre {{ white-space: pre-wrap; background: #101b2e; border: 1px solid rgba(148,163,184,.22); border-radius: 8px; padding: 12px; }}
    .note {{ color: #cbd5e1; background: rgba(217,119,6,.08); border: 1px solid rgba(217,119,6,.32); padding: 12px; border-radius: 8px; }}
    @media print {{ body {{ background: white; color: #0f172a; }} .card, table, pre {{ background: white; color: #0f172a; }} }}
  </style>
</head>
<body>
<main>
  <h1>{text["title"]}</h1>
  <p class="meta">{text["model"]}: {snapshot.get('model')} | {text["source"]}: {json.dumps(snapshot.get('source'), ensure_ascii=False)} | Report ID: {snapshot.get('report', {}).get('id')}</p>
  <div class="grid">
    <div class="card"><div>{text["object_samples"]}</div><div class="value">{snapshot.get('object_samples')}</div></div>
    <div class="card"><div>{text["avg_confidence"]}</div><div class="value">{summary.get('avg_conf')}</div></div>
    <div class="card"><div>{labels["avg_bbox_frame"]}</div><div class="value">{summary.get('avg_bbox_area_pct')}%</div></div>
    <div class="card"><div>{text["avg_sharpness"]}</div><div class="value">{summary.get('avg_sharpness')}</div></div>
  </div>
  <h2>{labels["product_spec"]}</h2>
  <p class="note">{labels["product_spec_note"]}</p>
  <table><tbody>{spec_rows}</tbody></table>
  <h2>{labels["charts"]}</h2>
  <div class="charts">
    <img class="chart" src="{charts.get('quality_radar', '')}" alt="Image quality radar">
    <img class="chart" src="{charts.get('model_radar', '')}" alt="Model perception radar">
    <img class="chart wide" src="{charts.get('heatmap', '')}" alt="BBox confidence matrix">
    <img class="chart" src="{charts.get('grades', '')}" alt="ABCD grade distribution">
    <img class="chart" src="{charts.get('violin', '')}" alt="Metric violin distribution">
  </div>
  <h2>{labels["representative_sample"]}</h2>
  <div class="sample">
    <img src="{sample.get('context') or sample.get('image') or ''}" alt="Representative bbox sample">
    <table><tbody>
      <tr><td>{labels["condition"]}</td><td>{sample.get('condition') or sample.get('stage') or '-'}</td></tr>
      <tr><td>Track</td><td>{sample.get('track_id')}</td></tr>
      <tr><td>Conf</td><td>{sample.get('conf')}</td></tr>
      <tr><td>{labels["bbox_frame_pct"]}</td><td>{sample.get('bbox_area_pct')}%</td></tr>
      <tr><td>Sharpness</td><td>{sample.get('sharpness_score')}</td></tr>
      <tr><td>{labels["metrics"]}</td><td><pre>{json.dumps(sample.get('metrics', {}), ensure_ascii=False, indent=2)}</pre></td></tr>
    </tbody></table>
  </div>
  <h2>{labels["metric_stats"]}</h2>
  <table><thead><tr><th>{labels["metric"]}</th><th>{labels["mean"]}</th><th>{labels["std"]}</th><th>P10</th><th>P50</th><th>P90</th></tr></thead><tbody>{metric_rows}</tbody></table>
  <h2>{labels["grade_ratio"]}</h2>
  <table><thead><tr><th>{text["grade"]}</th><th>{labels["count"]}</th><th>{labels["ratio"]}</th></tr></thead><tbody>{grade_rows}</tbody></table>
  <h2>{text["metric_definitions"]}</h2>
  <pre>{json.dumps(metric_definitions(), ensure_ascii=False, indent=2)}</pre>
  <h2>{text["current_bbox_samples"]}</h2>
  <table><thead><tr><th>{text["track"]}</th><th>{text["conf"]}</th><th>{labels["bbox_frame_pct"]}</th><th>{text["sharpness"]}</th><th>{labels["contour"]}</th><th>{text["grade"]}</th></tr></thead><tbody>{rows}</tbody></table>
</main>
</body>
</html>"""


if __name__ == "__main__":
    REPORT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "7860")), threaded=True)
