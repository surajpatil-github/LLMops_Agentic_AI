#!/bin/bash
# Build and push the LLMOps app image to Docker Hub (no Azure required).
#
# Usage:
#   export DOCKERHUB_USERNAME=yourname
#   export DOCKERHUB_TOKEN=your_access_token
#   bash scripts/build-and-push-docker-image.sh <tag>      # e.g. v1.0 or $(git rev-parse --short HEAD)
#
# If no tag is given, defaults to "latest".

set -euo pipefail

DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:?Set DOCKERHUB_USERNAME first}"
DOCKERHUB_TOKEN="${DOCKERHUB_TOKEN:?Set DOCKERHUB_TOKEN first}"

IMAGE_NAME="${DOCKERHUB_USERNAME}/llmops-app"
BUILD_TAG="${1:-latest}"

echo "Logging in to Docker Hub..."
echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin

echo "Building image: ${IMAGE_NAME}:${BUILD_TAG}..."
docker build \
  --platform linux/amd64 \
  -t "${IMAGE_NAME}:${BUILD_TAG}" \
  -t "${IMAGE_NAME}:latest" \
  -f Dockerfile .

echo "Pushing ${IMAGE_NAME}:${BUILD_TAG}..."
docker push "${IMAGE_NAME}:${BUILD_TAG}"
docker push "${IMAGE_NAME}:latest"

docker logout
echo "Done. Pushed ${IMAGE_NAME}:${BUILD_TAG} and :latest"
