# API

## `GET /`

Returns the WebUI shell from `templates/index.html`.

## `GET /api/assets`

Returns selectable file lists:

- `models`: `.tflite` files from `models/`
- `videos`: `.mkv`, `.webm`, `.mp4` files from `videos/`
- `images`: supported image files from `images/`

## `POST /api/load_model`

Payload:

```json
{ "model": "model_name.tflite" }
```

Loads a TFLite model into the server process.

## `POST /api/start`

Payload:

```json
{
  "model": "model_name.tflite",
  "source": { "type": "video", "name": "clip.mp4" },
  "conf_threshold": 0.25,
  "sample_stride": 1
}
```

Source variants:

- `{ "type": "video", "name": "clip.mp4" }`
- `{ "type": "webcam", "index": 0 }`
- `{ "type": "images", "images": ["a.jpg", "b.png"] }`

Starts a background inference job.

## `POST /api/stop`

Requests the current worker to stop.

## `GET /api/status`

Returns current runtime state and summaries:

- `current_objects`
- `area_conf_matrix`
- `quality_radar`
- `model_radar`
- `grade_distribution`
- `metric_distribution`
- `product_spec`
- `recommendations`
- `tracking_recommendations`

Large per-bin sample images are intentionally not included here.

## `POST /api/bin_sample`

Payload:

```json
{
  "area_bin": "0.25-0.5%",
  "conf_bin": "0.50-0.70"
}
```

Returns a random retained frame sample for the selected matrix bin. The UI uses this for the interactive bbox size inspector. Samples can include:

- original frame image
- original bbox coordinates and bbox frame-area percent
- track ID and tracking stability metrics
- short `track_trail` center-point history for video/webcam sources

## `POST /api/report`

Payload:

```json
{
  "name": "customer_model_quality",
  "tags": "night,v1",
  "language": "zh"
}
```

Exports self-contained HTML and JSON reports to:

- project `reports/`
- Docker-mounted `/app/downloads`, mapped to host `~/Downloads` by default

## `GET /reports/<name>`

Serves a file from `reports/`.

Add `?download=1` to force attachment download:

```text
/reports/customer_model_quality.html?download=1
```

The frontend uses this path after `POST /api/report` so browser UIs such as Chrome show native download progress/history.

## `GET /stream`

Multipart JPEG stream of the latest inference overlay.
