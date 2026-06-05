# Model Assessment WebUI Project Map

## Purpose

This project is a Dockerized Flask WebUI for evaluating YOLO/TFLite inference quality on people detections from videos, webcams, and image files. It quantifies how image quality, bbox scale, contour clarity, confidence, and tracking continuity affect model perception.

## Primary Entry Points

- Runtime backend: [../app.py](../app.py)
- UI template: [../templates/index.html](../templates/index.html)
- UI logic: [../static/app.js](../static/app.js)
- UI styling: [../static/styles.css](../static/styles.css)
- Docker runner: [../run.sh](../run.sh)
- Compose/Docker: [../docker-compose.yml](../docker-compose.yml), [../Dockerfile](../Dockerfile)

## Level 2 Specs

- [ARCHITECTURE.md](ARCHITECTURE.md): process model, backend responsibilities, and source flow.
- [DATA_MODEL.md](DATA_MODEL.md): metrics, bbox bins, grade calculation, tracking/spec indicators, report fields.
- [API.md](API.md): HTTP endpoints and payload expectations.
- [UI.md](UI.md): dashboard layout, matrix interaction, modal analysis, localization behavior.
- [OPERATIONS.md](OPERATIONS.md): Docker commands, mounted directories, report export behavior.
- [TESTING.md](TESTING.md): syntax checks, smoke tests, and manual verification checklist.

## Runtime Directories

- `models/`: `.tflite` files selectable in the UI.
- `videos/`: `.mkv`, `.webm`, `.mp4` video inputs.
- `images/`: image batch inputs and downloaded public image files.
- `reports/`: project-local exported HTML/JSON reports.
- `downloads/`: local fallback download export path when not running in Docker.
- `outputs/`: reserved for generated output artifacts.

## Important Design Notes

- Detection currently filters to `PERSON_CLASS_ID = 0`.
- Video/webcam sources use `SimpleTracker` IoU continuity as a ByteTrack-style fallback; image batches do not evaluate tracking continuity.
- BBox size bins are based on bbox area as a percent of the whole frame, not absolute pixels.
- Reports are self-contained HTML with embedded chart images.
