#!/bin/bash
set -e

# ==============================
# Configuration
# ==============================
APP_RESOURCE_GROUP="llmops-app-rg"
LOCATION="centralindia"          # <-- changed from eastus (policy issue)
RAND=$RANDOM
APP_ACR_NAME="llmopsappacr${RAND}"  # <-- unique name
CONTAINER_APP_ENV="llmops-env"

# ==============================
# Colors for output
# ==============================
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo -e "${BLUE}┌──────────────────────────────────────────────────────────┐${NC}"
echo -e "${BLUE}│        Setting Up Application Infrastructure             │${NC}"
echo -e "${BLUE}└──────────────────────────────────────────────────────────┘${NC}\n"

# ==============================
# Step 1: Create Resource Group
# ==============================
echo -e "${GREEN}Step 1: Creating app resource group...${NC}"
az group create --name "$APP_RESOURCE_GROUP" --location "$LOCATION"

# ==============================
# Step 2: Create Container Registry
# ==============================
echo -e "${GREEN}Step 2: Creating app container registry...${NC}"
az acr create \
  --resource-group "$APP_RESOURCE_GROUP" \
  --name "$APP_ACR_NAME" \
  --sku Basic \
  --location "$LOCATION"

az acr update --name "$APP_ACR_NAME" --admin-enabled true

# ==============================
# Step 3: Create Container Apps environment
# ==============================
echo -e "${GREEN}Step 3: Creating Container Apps environment...${NC}"
az containerapp env create \
  --name "$CONTAINER_APP_ENV" \
  --resource-group "$APP_RESOURCE_GROUP" \
  --location "$LOCATION"

# ==============================
# Step 4: Get ACR credentials
# ==============================
echo -e "${GREEN}Step 4: Getting ACR credentials...${NC}"
ACR_USERNAME=$(az acr credential show --name "$APP_ACR_NAME" --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name "$APP_ACR_NAME" --query "passwords[0].value" -o tsv)

echo -e "\n${BLUE}┌──────────────────────────────────────────────────────────┐${NC}"
echo -e "${BLUE}│                     Setup Complete!                      │${NC}"
echo -e "${BLUE}└──────────────────────────────────────────────────────────┘${NC}\n"

echo -e "${YELLOW}ACR NAME (save this):${NC} ${GREEN}${APP_ACR_NAME}${NC}\n"

echo -e "${YELLOW}⚠ IMPORTANT: Add these credentials to Jenkins:${NC}\n"
echo -e "Credential ID: ${BLUE}acr-username${NC}"
echo -e "Value:         ${GREEN}${ACR_USERNAME}${NC}\n"
echo -e "Credential ID: ${BLUE}acr-password${NC}"
echo -e "Value:         ${GREEN}${ACR_PASSWORD}${NC}\n"

