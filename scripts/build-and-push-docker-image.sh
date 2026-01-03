#!/bin/bash
set -euo pipefail

# ===== CONFIG =====
APP_ACR_NAME="llmopsappacr25151"
IMAGE_NAME="llmops-app"
BUILD_TAG="${1:-latest}"

# ===== LOGIN TO ACR (via Jenkins credentials) =====
echo "🔐 Logging in to Azure Container Registry..."

echo "$ACR_PASSWORD" | docker login ${APP_ACR_NAME}.azurecr.io \
  --username "$ACR_USERNAME" \
  --password-stdin

# ===== BUILD IMAGE =====
echo "🐳 Building Docker image..."

docker build \
  --platform linux/amd64 \
  -t ${APP_ACR_NAME}.azurecr.io/${IMAGE_NAME}:${BUILD_TAG} \
  -t ${APP_ACR_NAME}.azurecr.io/${IMAGE_NAME}:latest \
  -f Dockerfile .

# ===== PUSH IMAGE =====
echo "📤 Pushing image to ACR..."

docker push ${APP_ACR_NAME}.azurecr.io/${IMAGE_NAME}:${BUILD_TAG}
docker push ${APP_ACR_NAME}.azurecr.io/${IMAGE_NAME}:latest

echo "✅ Build and push completed successfully!"
