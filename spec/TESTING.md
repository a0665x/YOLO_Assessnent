# Testing And Verification

## Static Checks

Run:

```bash
python -m py_compile app.py
node --check static/app.js
```

## Flask Smoke Test

Use Flask test client when Docker is not required:

```bash
python - <<'PY'
from app import app
client = app.test_client()
print(client.get('/api/status').status_code)
print(client.post('/api/report', json={'name':'smoke','tags':'test','language':'zh'}).status_code)
PY
```

## Docker Smoke Test

```bash
./run.sh start
./run.sh status
curl -sS http://127.0.0.1:7860/api/status
```

In restricted execution environments, localhost curl may require elevated permissions even when the service is running.

## Manual UI Checks

- Language selector changes UI copy.
- Webcam index is visible only when webcam source is selected.
- Image source warns that tracking quality is not evaluated.
- Matrix x-axis uses bbox frame-area percent bins.
- Clicking a populated matrix cell opens the Bin Sample Inspector.
- Dragging a rectangle on the sample image reports frame percent and pixel W,H.
- A/B/C/D and metric distribution charts update from `/api/status`.
- Grade chips show the formula tooltip on hover/focus.
- Classic Samples show representative quality conditions.
- Report export writes HTML/JSON to `reports/` and host `~/Downloads`.
