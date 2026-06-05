# Architecture

## Overview

The app is a single Flask service in `app.py` with a server-rendered HTML shell and static JavaScript/CSS. Long-running inference work runs in a background thread and writes into a shared `AssessmentState` guarded by a lock.

## Backend Components

- `YoloTFLiteRunner`: loads a selected `.tflite` model and performs TFLite inference.
- `_postprocess()`: interprets YOLO-style output tensors, converts boxes to frame coordinates, runs NMS, and emits detections.
- `SimpleTracker`: lightweight IoU-based tracking for video/webcam continuity.
- `AssessmentState`: shared runtime state for current job, detections, examples, bin samples, metric histories, and report data.
- `update_stats()`: central metric enrichment point for every detected/tracked bbox.
- `tracking_quality_metrics()`: computes per-track continuity and path smoothness for video/webcam Grade scoring.
- Report chart functions: generate heatmap, radar, grade, and violin charts as base64 PNGs.

## Source Flow

1. UI calls `/api/start` with model, source, threshold, and frame stride.
2. `worker()` resets state, loads the model, and delegates to `process_source()`.
3. Video/webcam sources use `cv2.VideoCapture`; image sources iterate selected files.
4. Person detections are enriched with bbox/image quality metrics.
5. Status, matrix, radar charts, samples, and reports read from `AssessmentState`.

## Concurrency

- The Flask server remains responsive while inference runs in a daemon thread.
- All shared state mutation and reads should happen under `state.lock`.
- Streaming uses `state.last_frame_jpeg` and returns multipart JPEG frames.

## Tracking Scope

- `analysis_mode = "tracking"` only for `video` and `webcam`.
- `analysis_mode = "image_batch"` for image files; tracking recommendations explicitly warn that ID continuity is not meaningful for discrete images.
