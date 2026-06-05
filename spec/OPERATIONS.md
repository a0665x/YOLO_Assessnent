# Operations

## Docker Runner

Use `./run.sh --help` for available commands.

Common commands:

```bash
./run.sh start
./run.sh stop
./run.sh restart
./run.sh rebuild
./run.sh down_up
./run.sh status
./run.sh logs
```

Default WebUI URL:

```text
http://127.0.0.1:7860
```

## Environment Variables

- `HOST_PORT`: host WebUI port, default `7860`
- `PORT`: container Flask port, default `7860`
- `PROJECT_NAME`: Docker compose project name, default `model-assessment`
- `DOWNLOAD_DIR`: host folder mounted to `/app/downloads`, default `~/Downloads`

## Docker Volumes

`docker-compose.yml` mounts:

- `./models:/app/models:ro`
- `./videos:/app/videos:ro`
- `./images:/app/images`
- `./reports:/app/reports`
- `./outputs:/app/outputs`
- `${DOWNLOAD_DIR:-/home/nvidia/Downloads}:/app/downloads`
- `/dev:/dev`

The service runs privileged so webcam devices can be accessed.

## Report Export

Reports are written to both:

- project-local `reports/`
- host Downloads through `/app/downloads`

The HTML report embeds charts as base64 PNGs and can be viewed without external network access.
