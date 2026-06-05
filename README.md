# YOLO Inference Assessment WebUI

<p align="center">
  <img src="./ai_assessment_720p.gif" alt="YOLO Inference Assessment WebUI demo" width="100%">
</p>

YOLO Inference Assessment WebUI is a Dockerized dashboard for quantifying how image quality affects object-detection perception on edge devices. It is designed to help engineers and customers turn video/image conditions into measurable model-operating indicators instead of relying only on subjective visual inspection.

The project focuses on YOLO/TFLite inference analysis for people detection. It measures detection confidence, bbox scale, bbox contour clarity, whole-frame quality, and tracking stability so a product team can define practical image-quality requirements for a deployed model.

## Why This Exists

Edge AI systems often fail because the camera environment does not match the model's usable perception range. A model may work well in a lab but become unstable with small targets, low light, motion blur, weak contrast, or unstable tracking IDs.

This tool helps answer product-spec questions such as:

- What bbox frame-area percentage is still usable for this model?
- What confidence range is typical under the current image quality?
- How sharp or blurry can a person bbox be before detection quality drops?
- How stable are tracking IDs across continuous video?
- What light, exposure, contrast, and motion conditions should be written into the product specification?
- Is the current video/image source statistically sufficient for model assessment?

## Core Features

- **TFLite model selector**: load `.tflite` models from `models/`.
- **Multiple input sources**:
  - video files from `videos/` (`.mp4`, `.mkv`, `.webm`)
  - webcam index
  - image batches from `images/`
- **Live inference stream** with bbox overlay, confidence, tracking ID, and per-bbox quality values.
- **BBox frame-area / confidence matrix** using percent-of-frame bins such as `XXS <0.15%`, `XS`, `S`, `M`, `L`, `XL`, `XXL`, and `XXXL`.
- **Interactive bin sampling**: click a matrix cell to inspect a real frame sample, view the original bbox, and drag custom rectangles to understand bbox scale in pixels and frame-area percentage.
- **Tracking visualization** for video/webcam sources, including per-ID stability metrics and center-point trail overlays.
- **Image quality radar** for whole-frame clarity, exposure, contrast, lux proxy, motion stability, and edge structure.
- **Model perception radar** for confidence coverage, bbox contour quality, tracking stability, bbox size, low-blur coverage, edge coverage, and A/B coverage.
- **A/B/C/D grade distribution** with hover definitions.
- **Metric distribution chart** with mean, standard deviation, and percentile spread for key perception metrics.
- **Resizable analytic columns** so charts and tables can be widened interactively; canvases redraw to fit the current column size.
- **Per-bbox analysis modal** with hierarchical metric groups:
  - final grade drivers
  - bbox image-processing diagnostics
  - tracking and whole-frame context
- **Multilingual UI and reports**: Chinese, English, Japanese, and Korean.
- **Self-contained report export** as HTML and JSON, with browser-native download behavior.
- **Docker-first operation** through `./run.sh`.

## Assessment Metrics

### Detection And BBox Metrics

- confidence score
- bbox width/height
- bbox frame-area percentage
- bbox area bin and size class
- edge density
- brightness and contrast inside bbox
- Laplacian variance
- Tenengrad score
- contour clarity score
- strict bbox sharpness score

### Whole-Frame Image Quality

- frame sharpness
- frame brightness
- exposure score
- motion blur estimate
- frame contrast
- lux proxy
- edge density
- overall frame quality score

`lux proxy` is a relative image-derived indicator. It is not a calibrated physical lux measurement.

### Tracking Metrics

Tracking metrics are evaluated for continuous video or webcam sources. Image batches are discrete samples, so tracking continuity is not treated as meaningful there.

- track age score
- track continuity score
- track smoothness score
- track center jump percentage
- track stability score
- stable track count
- stable bbox sample count
- maximum simultaneous bbox count
- average simultaneous bbox count

These metrics help detect ID drops, ID switches, sudden center-point jumps, and unstable trajectories.

## A/B/C/D Perception Grade

For video and webcam sources:

```text
score = 40% * confidence
      + 30% * bbox sharpness
      + 20% * tracking stability
      + 10% * bbox frame-area score
```

For image batches:

```text
score = 45% * confidence
      + 40% * bbox sharpness
      + 15% * bbox frame-area score
```

Grade thresholds:

```text
A >= 75
B >= 60
C >= 45
D < 45
```

The grade is intended as a practical product-facing perception indicator. It is not a replacement for labeled dataset accuracy metrics such as mAP, precision, or recall. Instead, it explains whether the current image conditions are inside the model's usable perception range.

## Quick Start

### 1. Prepare Input Files

Place files in these folders:

```text
models/   TFLite models
videos/   .mp4, .mkv, .webm videos
images/   image batch inputs
```

Example:

```text
models/yolov8n_float32.tflite
videos/park.webm
images/sample_001.jpg
```

### 2. Start The WebUI

```bash
./run.sh start
```

Open:

```text
http://127.0.0.1:7860
```

### 3. Common Docker Commands

```bash
./run.sh --help
./run.sh start
./run.sh stop
./run.sh restart
./run.sh rebuild
./run.sh down_up
./run.sh logs
./run.sh status
```

Use a different host port:

```bash
HOST_PORT=8080 ./run.sh start
```

By default, exported reports are also written to:

```text
~/Downloads
```

You can change that location:

```bash
DOWNLOAD_DIR=/path/to/downloads ./run.sh start
```

## Docker Mounts

The compose setup mounts local folders into the container:

```text
./models    -> /app/models
./videos    -> /app/videos
./images    -> /app/images
./reports   -> /app/reports
./outputs   -> /app/outputs
~/Downloads -> /app/downloads
```

The container is started with `/dev` mounted and `privileged: true` so webcam devices can be accessed when available.

## Report Export

The report button generates:

- self-contained HTML report
- JSON metrics report

Reports include:

- summary metrics
- bbox frame-area / confidence matrix
- image quality radar
- model perception radar
- A/B/C/D grade distribution
- metric distribution chart
- product-spec indicators
- representative bbox samples
- average and distribution statistics
- localized explanations based on the selected UI language

The frontend triggers `/reports/<file>?download=1`, so browsers such as Chrome show the export through the native download UI.

## Project Structure

```text
.
├── app.py                  Flask backend, inference loop, metrics, report generation
├── templates/index.html    WebUI shell
├── static/app.js           UI logic, charts, interactions, localization
├── static/styles.css       Dark dashboard styling and responsive layout
├── models/                 TFLite model files
├── videos/                 Video inputs
├── images/                 Image batch inputs
├── reports/                Generated HTML/JSON reports
├── outputs/                Reserved output artifacts
├── spec/                   Architecture and onboarding documentation
├── Dockerfile
├── docker-compose.yml
└── run.sh                  Docker operation helper
```

## API Overview

Main endpoints:

- `GET /` - WebUI
- `GET /api/assets` - available models, videos, and images
- `POST /api/load_model` - load a TFLite model
- `POST /api/start` - start analysis
- `POST /api/stop` - stop analysis
- `GET /api/status` - live metrics and chart data
- `POST /api/bin_sample` - random sample for a matrix bin
- `POST /api/report` - generate report
- `GET /reports/<name>?download=1` - browser-native report download
- `GET /stream` - live MJPEG inference stream

More implementation details are documented under `spec/`.

## Notes And Limitations

- The default target class is `person`.
- The tracker is a lightweight IoU-based continuity tracker used for product-level stability analysis.
- Image batches do not produce meaningful tracking continuity metrics.
- The quality metrics are image-derived proxies intended for operational assessment.
- The tool evaluates perception conditions and model behavior, not labeled-dataset accuracy by itself.

## Development Checks

```bash
python -m py_compile app.py
node --check static/app.js
./run.sh status
```

## License

No license has been assigned yet. Add a `LICENSE` file before public redistribution or commercial reuse.
