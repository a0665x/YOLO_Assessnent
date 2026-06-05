#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="${PROJECT_NAME:-model-assessment}"
SERVICE_NAME="${SERVICE_NAME:-yolo-assessment}"
HOST_PORT="${HOST_PORT:-7860}"
PORT="${PORT:-7860}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${HOME}/Downloads}"
export HOST_PORT PORT DOWNLOAD_DIR

usage() {
  cat <<EOF
YOLO Model Assessment WebUI Docker runner

Usage:
  ./run.sh [command]

Commands:
  up           Alias of start
  start        Build if needed and start the WebUI container in background
  stop         Stop the WebUI container without removing it
  restart      Restart the WebUI container
  rebuild      Rebuild the Docker image without cache, then start
  down         Stop and remove the container/network
  down_up      Run down, rebuild image, then start fresh
  down-up      Alias of down_up
  logs         Follow container logs
  status       Show docker compose service status
  shell        Open a shell inside the running container
  build        Build the Docker image
  help         Show this help
  --help       Show this help

Environment:
  HOST_PORT    Host port for WebUI, default 7860
  PORT         Container Flask port, default 7860
  PROJECT_NAME Docker compose project name, default model-assessment
  DOWNLOAD_DIR  Host folder for exported reports, default ~/Downloads

Examples:
  ./run.sh start
  HOST_PORT=8080 ./run.sh start
  ./run.sh rebuild
  ./run.sh down_up

WebUI:
  http://127.0.0.1:${HOST_PORT}
EOF
}

compose_cmd() {
  if docker compose version >/dev/null 2>&1; then
    docker compose -p "${PROJECT_NAME}" "$@"
  elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose -p "${PROJECT_NAME}" "$@"
  else
    echo "docker compose or docker-compose is required" >&2
    exit 1
  fi
}

ensure_dirs() {
  mkdir -p models videos images reports outputs
  mkdir -p "${DOWNLOAD_DIR}"
}

start() {
  ensure_dirs
  compose_cmd up -d --build
  echo "WebUI is starting at http://127.0.0.1:${HOST_PORT}"
}

stop() {
  compose_cmd stop
}

restart() {
  compose_cmd restart "${SERVICE_NAME}"
  echo "WebUI restarted at http://127.0.0.1:${HOST_PORT}"
}

build() {
  ensure_dirs
  compose_cmd build
}

rebuild() {
  ensure_dirs
  compose_cmd build --no-cache
  compose_cmd up -d
  echo "WebUI rebuilt and started at http://127.0.0.1:${HOST_PORT}"
}

down() {
  compose_cmd down
}

down_up() {
  down
  rebuild
}

logs() {
  compose_cmd logs -f --tail=200 "${SERVICE_NAME}"
}

status() {
  compose_cmd ps
}

shell_in() {
  compose_cmd exec "${SERVICE_NAME}" /bin/sh
}

cmd="${1:-start}"
case "${cmd}" in
  start|up) start ;;
  stop) stop ;;
  restart) restart ;;
  rebuild) rebuild ;;
  down) down ;;
  down_up|down-up) down_up ;;
  logs) logs ;;
  status|ps) status ;;
  shell|exec) shell_in ;;
  build) build ;;
  help|--help|-h) usage ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    echo >&2
    usage
    exit 2
    ;;
esac
