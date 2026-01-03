#!/bin/bash
set -e

# ==========================================
# Configuration
# ==========================================
RESOURCE_GROUP="llmops-jenkins-rg"
LOCATION="centralindia"

STORAGE_ACCOUNT="llmopsjenkinsstore"
FILE_SHARE_NAME="jenkins-data"

ACR_NAME="llmopsacr959396117"
IMAGE_NAME="jenkins-python"
IMAGE_TAG="latest"

CONTAINER_NAME="jenkins-llmops"
DNS_NAME="jenkins-llmops-${RANDOM}"

# Jenkins persistent home (ACI-safe path)
JENKINS_HOME="/mnt/jenkins"

# ==========================================
# Colors
# ==========================================
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=== Azure Jenkins Deployment using ACR ===${NC}\n"

# ==========================================
# Step 0: Pre-checks
# ==========================================
az account show >/dev/null
docker info >/dev/null

: "${GROQ_API_KEY:?Set GROQ_API_KEY first}"
: "${GOOGLE_API_KEY:?Set GOOGLE_API_KEY first}"

# ==========================================
# Step 1: Create Resource Group
# ==========================================
echo -e "${GREEN}Step 1: Creating Resource Group...${NC}"
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION >/dev/null

# ==========================================
# Step 2: Create Storage Account
# ==========================================
echo -e "${GREEN}Step 2: Creating Storage Account...${NC}"
az storage account create \
  --name $STORAGE_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Standard_LRS >/dev/null

# ==========================================
# Step 3: Get Storage Key
# ==========================================
echo -e "${GREEN}Step 3: Getting Storage Key...${NC}"
STORAGE_KEY=$(az storage account keys list \
  --resource-group $RESOURCE_GROUP \
  --account-name $STORAGE_ACCOUNT \
  --query '[0].value' -o tsv)

# ==========================================
# Step 4: Create File Share
# ==========================================
echo -e "${GREEN}Step 4: Creating File Share...${NC}"
az storage share create \
  --name $FILE_SHARE_NAME \
  --account-name $STORAGE_ACCOUNT \
  --account-key $STORAGE_KEY \
  --quota 10 >/dev/null || true

# ==========================================
# Step 5: Enable ACR Admin
# ==========================================
echo -e "${GREEN}Step 5: Enabling ACR Admin...${NC}"
az acr update -n $ACR_NAME --admin-enabled true >/dev/null

# ==========================================
# Step 6: Get ACR Credentials
# ==========================================
echo -e "${GREEN}Step 6: Getting ACR Credentials...${NC}"
ACR_USERNAME=$(az acr credential show -n $ACR_NAME --query "username" -o tsv)
ACR_PASSWORD=$(az acr credential show -n $ACR_NAME --query "passwords[0].value" -o tsv)

# ==========================================
# Step 7: Build & Push Jenkins Image
# ==========================================
echo -e "${GREEN}Step 7: Building and Pushing Jenkins Image to ACR...${NC}"

az acr login -n $ACR_NAME

docker build --platform linux/amd64 \
  -f ../Dockerfile.jenkins \
  -t $ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG \
  ..

docker push $ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG

# ==========================================
# Step 8: Deploy Jenkins Container (ACI)
# ==========================================
echo -e "${GREEN}Step 8: Deploying Jenkins Container...${NC}"

az container create \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${CONTAINER_NAME}" \
  --image "${ACR_NAME}.azurecr.io/jenkins-python:latest" \
  --os-type Linux \
  --dns-name-label "${DNS_NAME}" \
  --ports 8080 50000 \
  --cpu 2 \
  --memory 4 \
  --registry-login-server "${ACR_NAME}.azurecr.io" \
  --registry-username "${ACR_USERNAME}" \
  --registry-password "${ACR_PASSWORD}" \
  --azure-file-volume-account-name "${STORAGE_ACCOUNT}" \
  --azure-file-volume-account-key "${STORAGE_KEY}" \
  --azure-file-volume-share-name "${FILE_SHARE_NAME}" \
  --azure-file-volume-mount-path "/mnt/jenkins" \
  --environment-variables \
    JENKINS_HOME="/mnt/jenkins" \
    GROQ_API_KEY="${GROQ_API_KEY}" \
    GOOGLE_API_KEY="${GOOGLE_API_KEY}" \
    LLM_PROVIDER="google"



# ==========================================
# Step 9: Jenkins URL
# ==========================================
echo -e "${GREEN}Step 9: Fetching Jenkins URL...${NC}"
JENKINS_URL=$(az container show \
  --resource-group $RESOURCE_GROUP \
  --name $CONTAINER_NAME \
  --query "ipAddress.fqdn" -o tsv)

echo -e "\n${GREEN}=== DEPLOYMENT COMPLETE ===${NC}"
echo -e "Jenkins URL: ${BLUE}http://$JENKINS_URL:8080${NC}"

echo -e "\n${GREEN}Get Jenkins Admin Password:${NC}"
echo -e "${BLUE}az container exec -g $RESOURCE_GROUP -n $CONTAINER_NAME --exec-command \"/bin/sh -c 'cat $JENKINS_HOME/secrets/initialAdminPassword'\"${NC}"
