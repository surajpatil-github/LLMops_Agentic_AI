/*
 * Jenkins Pipeline - Test + Deploy to Azure Container Apps
 *
 * Workflow:
 * 1) Build & push Docker image LOCALLY: scripts/build-and-push-docker-image.sh <tag>
 * 2) Jenkins pipeline:
 *    - Checkout
 *    - Setup Python env (uv)
 *    - Install deps
 *    - Run tests + coverage
 *    - Login to Azure (SP)
 *    - Verify image exists in ACR
 *    - Deploy to Azure Container Apps (create/update)
 *    - Verify deployment (URL + health + logs)
 *
 * Notes:
 * - Uses /tmp for Python/uv caches to avoid Azure Files limitations
 * - Designed to work in free/student subscriptions
 */

pipeline {
  agent any

  // ===== Global environment =====
  environment {
    // -----------------------------
    // Python settings
    // -----------------------------
    PYTHON_VERSION = '3.12'
    PYTHONPATH = "${WORKSPACE}:${WORKSPACE}/multi_doc_chat"

    // -----------------------------
    // Optional API keys
    // -----------------------------
    GROQ_API_KEY   = "${env.GROQ_API_KEY ?: ''}"
    GOOGLE_API_KEY = "${env.GOOGLE_API_KEY ?: ''}"
    LLM_PROVIDER   = "${env.LLM_PROVIDER ?: 'groq'}"

    // -----------------------------
    // Azure Container Registry (YOUR)
    // -----------------------------
    APP_ACR_NAME   = "llmopsappacr25151"
    APP_ACR_SERVER = "${APP_ACR_NAME}.azurecr.io"
    IMAGE_NAME     = "llmops-app"

    // -----------------------------
    // Azure Container Apps (YOUR)
    // -----------------------------
    APP_RESOURCE_GROUP = "llmops-app-rg"
    CONTAINER_APP_NAME = "llmops-app"
    CONTAINER_APP_ENV  = "llmops-env"
    APP_LOCATION       = "centralindia"

    // App port (from your screenshot it’s 8080)
    TARGET_PORT = "8080"

    // Misc
    MAX_RETRIES = "60"
    SLEEP_SECS  = "10"
  }

  // ===== Parameters =====
  parameters {
    booleanParam(name: 'RUN_DEPLOY', defaultValue: true, description: 'Run deploy stages (Azure Container Apps)')
    string(name: 'IMAGE_TAG', defaultValue: 'latest', description: 'Docker tag already pushed to ACR (e.g., test/latest)')
  }

  // ===== Trigger =====
  triggers {
    // Poll SCM approx every 2 minutes
    pollSCM('*/2 * * * *')
  }

  stages {

    // =========================================================
    // Stage: Checkout
    // =========================================================
    stage('Checkout') {
      steps {
        echo '📦 Checking out code from repository...'
        checkout scm
      }
    }

    // =========================================================
    // Stage: Setup Python Environment
    // =========================================================
    stage('Setup Python Environment') {
      steps {
        echo '🐍 Setting up Python virtual environment...'
        sh '''
          set -e

          echo "Installing uv..."
          curl -LsSf https://astral.sh/uv/install.sh | sh
          UV="$HOME/.local/bin/uv"

          echo "Configuring uv temp locations..."
          export UV_PYTHON_INSTALL_DIR=/tmp/uv/python
          export XDG_DATA_HOME=/tmp/.local/share
          export XDG_CACHE_HOME=/tmp/.cache

          echo "Installing Python ${PYTHON_VERSION}..."
          "$UV" python install ${PYTHON_VERSION}

          echo "Creating venv /tmp/venv-${BUILD_NUMBER}..."
          "$UV" venv --python ${PYTHON_VERSION} /tmp/venv-${BUILD_NUMBER}

          echo "Python version:"
          /tmp/venv-${BUILD_NUMBER}/bin/python --version

          echo "uv version:"
          "$UV" --version
        '''
      }
    }

    // =========================================================
    // Stage: Install Dependencies
    // =========================================================
    stage('Install Dependencies') {
      steps {
        echo '📦 Installing project dependencies...'
        sh '''
          set -e

          VENV_PY="/tmp/venv-${BUILD_NUMBER}/bin/python"
          UV="$HOME/.local/bin/uv"

          echo "Using temp storage for dependency resolution..."
          export XDG_DATA_HOME=/tmp/.local/share
          export XDG_CACHE_HOME=/tmp/.cache

          echo "Sanitizing requirements.txt..."
          SAN_REQ=$(mktemp)

          cat requirements.txt \
            | sed -E '/^[[:space:]]*llmops-series(==.*)?[[:space:]]*$/d' \
            | sed -E '/^[[:space:]]*pywin32(==.*)?[[:space:]]*$/d' \
            > "$SAN_REQ"

          echo "Installing third-party deps..."
          "$UV" pip install --python "$VENV_PY" -r "$SAN_REQ"

          echo "Installing test tooling..."
          "$UV" pip install --python "$VENV_PY" pytest pytest-cov

          echo "Using PYTHONPATH=${PYTHONPATH}"
        '''
      }
    }

    // =========================================================
    // Stage: Run Tests
    // =========================================================
    stage('Run Tests') {
      steps {
        echo '🧪 Running pytest tests...'
        sh '''
          set -e
          . /tmp/venv-${BUILD_NUMBER}/bin/activate

          mkdir -p test-results

          pytest tests/ \
            --verbose \
            --junit-xml=test-results/results.xml \
            --cov=multi_doc_chat \
            --cov-report=xml:coverage.xml \
            --cov-report=html:htmlcov \
            --cov-report=term || true
        '''
      }
      post {
        always {
          echo '📊 Archiving test results and coverage...'
          junit allowEmptyResults: true, testResults: 'test-results/*.xml'
          archiveArtifacts artifacts: 'coverage.xml,htmlcov/**,test-results/**', allowEmptyArchive: true
        }
      }
    }

    // =========================================================
    // Stage: Login to Azure
    // =========================================================
    stage('Login to Azure') {
      when { expression { params.RUN_DEPLOY } }
      steps {
        echo '🔐 Logging into Azure...'
        withCredentials([
          string(credentialsId: 'azure-client-id',       variable: 'AZURE_CLIENT_ID'),
          string(credentialsId: 'azure-client-secret',   variable: 'AZURE_CLIENT_SECRET'),
          string(credentialsId: 'azure-tenant-id',       variable: 'AZURE_TENANT_ID'),
          string(credentialsId: 'azure-subscription-id', variable: 'AZURE_SUBSCRIPTION_ID')
        ]) {
          sh '''
            set -e
            az login --service-principal \
              -u $AZURE_CLIENT_ID \
              -p $AZURE_CLIENT_SECRET \
              --tenant $AZURE_TENANT_ID

            az account set --subscription $AZURE_SUBSCRIPTION_ID
            az account show
          '''
        }
      }
    }

    // =========================================================
    // Stage: Verify Docker Image Exists
    // (more verbose checks like the long file)
    // =========================================================
    stage('Verify Docker Image Exists') {
      when { expression { params.RUN_DEPLOY } }
      steps {
        echo '🔎 Verifying Docker image exists in ACR...'
        sh """
          set -e
          echo "ACR: ${APP_ACR_NAME}"
          echo "Repo: ${IMAGE_NAME}"
          echo "Tag:  ${params.IMAGE_TAG}"

          echo "Listing tags (top 60 lines)..."
          az acr repository show-tags -n ${APP_ACR_NAME} --repository ${IMAGE_NAME} -o table | head -n 60

          echo "Checking manifests for tag match..."
          az acr repository show-manifests \
            -n ${APP_ACR_NAME} \
            --repository ${IMAGE_NAME} \
            --query \"[?tags[?@=='${params.IMAGE_TAG}']]\" \
            -o table | head -n 40
        """
      }
    }

    // =========================================================
    // Stage: Deploy to Azure Container Apps (BIG block)
    // Mirrors your screenshot logic: extension, RG, env health, recreate, create/update app
    // =========================================================
    stage('Deploy to Azure Container Apps') {
      when { expression { params.RUN_DEPLOY } }
      steps {
        echo '🚀 Deploying to Azure Container Apps...'
        withCredentials([
          string(credentialsId: 'azure-client-id',       variable: 'AZURE_CLIENT_ID'),
          string(credentialsId: 'azure-client-secret',   variable: 'AZURE_CLIENT_SECRET'),
          string(credentialsId: 'azure-tenant-id',       variable: 'AZURE_TENANT_ID'),
          string(credentialsId: 'azure-subscription-id', variable: 'AZURE_SUBSCRIPTION_ID'),
          string(credentialsId: 'acr-username',          variable: 'ACR_USERNAME'),
          string(credentialsId: 'acr-password',          variable: 'ACR_PASSWORD')
        ]) {
          sh """
            set -e

            echo "🔐 Login to Azure..."
            az login --service-principal \\
              -u "\$AZURE_CLIENT_ID" \\
              -p "\$AZURE_CLIENT_SECRET" \\
              --tenant "\$AZURE_TENANT_ID" >/dev/null

            az account set --subscription "\$AZURE_SUBSCRIPTION_ID"

            echo "🧩 Ensure containerapp extension..."
            az extension show -n containerapp >/dev/null 2>&1 || az extension add -n containerapp >/dev/null

            echo "📌 Ensure resource group exists..."
            if [ "\$(az group exists --name ${APP_RESOURCE_GROUP})" != "true" ]; then
              echo "Resource group not found. Creating ${APP_RESOURCE_GROUP} in ${APP_LOCATION}..."
              az group create --name ${APP_RESOURCE_GROUP} --location ${APP_LOCATION} >/dev/null
            fi

            echo "🌿 Ensure Container Apps environment exists and is healthy..."

            get_env_state() {
              az containerapp env show \\
                --name ${CONTAINER_APP_ENV} \\
                --resource-group ${APP_RESOURCE_GROUP} \\
                --query properties.provisioningState -o tsv 2>/dev/null || echo "NotFound"
            }

            recreate_env() {
              echo "Recreating Container Apps environment ${CONTAINER_APP_ENV}..."
              az containerapp env delete \\
                --name ${CONTAINER_APP_ENV} \\
                --resource-group ${APP_RESOURCE_GROUP} \\
                --yes --no-wait || true

              MAX_DELETE_RETRIES=60
              ATT=1
              until [ "\$(get_env_state)" = "NotFound" ]; do
                if [ \$ATT -ge \$MAX_DELETE_RETRIES ]; then
                  echo "❌ ERROR: Environment ${CONTAINER_APP_ENV} did not delete in time."
                  exit 1
                fi
                echo "Waiting for ${CONTAINER_APP_ENV} deletion... (\$ATT/\$MAX_DELETE_RETRIES)"
                sleep 10
                ATT=\$((ATT+1))
              done

              echo "Creating fresh environment ${CONTAINER_APP_ENV}..."
              az containerapp env create \\
                --name ${CONTAINER_APP_ENV} \\
                --resource-group ${APP_RESOURCE_GROUP} \\
                --location ${APP_LOCATION} >/dev/null
            }

            STATE="\$(get_env_state)"
            echo "Current env provisioningState: \$STATE"

            if [ "\$STATE" = "NotFound" ]; then
              echo "Environment not found. Creating..."
              az containerapp env create \\
                --name ${CONTAINER_APP_ENV} \\
                --resource-group ${APP_RESOURCE_GROUP} \\
                --location ${APP_LOCATION} >/dev/null
              STATE="\$(get_env_state)"
              echo "New env provisioningState: \$STATE"
            fi

            if [ "\$STATE" = "ScheduledForDelete" ] || [ "\$STATE" = "Failed" ]; then
              echo "Environment state is '\$STATE'. Will recreate."
              recreate_env
              STATE="\$(get_env_state)"
            fi

            echo "Waiting for environment ${CONTAINER_APP_ENV} to be 'Succeeded'... (current: \$STATE)"
            MAX_RETRIES=60
            SLEEP_SECS=10
            ATTEMPT=1

            while [ "\$(get_env_state)" != "Succeeded" ]; do
              if [ \$ATTEMPT -ge \$MAX_RETRIES ]; then
                echo "❌ ERROR: Environment ${CONTAINER_APP_ENV} is not ready after \$((MAX_RETRIES*SLEEP_SECS))s."
                az containerapp env show --name ${CONTAINER_APP_ENV} --resource-group ${APP_RESOURCE_GROUP} -o yaml || true
                exit 1
              fi

              CURR="\$(get_env_state)"
              echo "Attempt \$ATTEMPT/\$MAX_RETRIES: provisioningState=\$CURR. Waiting \$SLEEP_SECS sec..."
              sleep \$SLEEP_SECS
              ATTEMPT=\$((ATTEMPT+1))

              if [ "\$CURR" = "ScheduledForDelete" ] || [ "\$CURR" = "Failed" ]; then
                echo "Env flipped to '\$CURR' mid-wait. Recreating..."
                recreate_env
              fi
            done

            echo "✅ Environment is ready."

            IMAGE_URI="${APP_ACR_SERVER}/${IMAGE_NAME}:${params.IMAGE_TAG}"
            echo "Using image: \$IMAGE_URI"

            # Create the Container App if it does not exist; otherwise update
            if az containerapp show --name ${CONTAINER_APP_NAME} --resource-group ${APP_RESOURCE_GROUP} >/dev/null 2>&1; then
              echo "Container App exists. Updating image..."
              az containerapp update \\
                --name ${CONTAINER_APP_NAME} \\
                --resource-group ${APP_RESOURCE_GROUP} \\
                --image "\$IMAGE_URI" >/dev/null
            else
              echo "Container App not found. Creating it now..."
              az containerapp create \\
                --name ${CONTAINER_APP_NAME} \\
                --resource-group ${APP_RESOURCE_GROUP} \\
                --environment ${CONTAINER_APP_ENV} \\
                --image "\$IMAGE_URI" \\
                --ingress external \\
                --target-port ${TARGET_PORT} \\
                --min-replicas 1 \\
                --max-replicas 3 \\
                --registry-server ${APP_ACR_SERVER} \\
                --registry-username "\$ACR_USERNAME" \\
                --registry-password "\$ACR_PASSWORD" \\
                --env-vars \\
                  GROQ_API_KEY="\${GROQ_API_KEY}" \\
                  GOOGLE_API_KEY="\${GOOGLE_API_KEY}" \\
                  LLM_PROVIDER="\${LLM_PROVIDER}" \\
                >/dev/null
            fi

            echo "Waiting for deployment to stabilize..."
            sleep 30
          """
        }
      }
    }

    // =========================================================
    // Stage: Verify Deployment (URL + curl + logs) + cleanup
    // =========================================================
    stage('Verify Deployment') {
      steps {
        echo '✅ Verifying deployment...'
        withCredentials([
          string(credentialsId: 'azure-client-id',       variable: 'AZURE_CLIENT_ID'),
          string(credentialsId: 'azure-client-secret',   variable: 'AZURE_CLIENT_SECRET'),
          string(credentialsId: 'azure-tenant-id',       variable: 'AZURE_TENANT_ID'),
          string(credentialsId: 'azure-subscription-id', variable: 'AZURE_SUBSCRIPTION_ID')
        ]) {
          sh '''
            set -e

            az login --service-principal \
              -u $AZURE_CLIENT_ID \
              -p $AZURE_CLIENT_SECRET \
              --tenant $AZURE_TENANT_ID >/dev/null

            az account set --subscription $AZURE_SUBSCRIPTION_ID

            az extension show -n containerapp >/dev/null 2>&1 || az extension add -n containerapp >/dev/null

            APP_URL=$(az containerapp show \
              --name $CONTAINER_APP_NAME \
              --resource-group $APP_RESOURCE_GROUP \
              --query properties.configuration.ingress.fqdn -o tsv)

            echo "Application URL: https://$APP_URL"

            HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" https://$APP_URL || echo "000")

            if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "307" ]; then
              echo "✅ Deployment successful! App is responding."
            else
              echo "⚠️ Warning: App returned HTTP $HTTP_CODE"
            fi

            echo "Recent logs:"
            az containerapp logs show \
              --name $CONTAINER_APP_NAME \
              --resource-group $APP_RESOURCE_GROUP \
              --tail 50 || true

            echo "🧹 Cleaning up virtual environment..."
            rm -rf /tmp/venv-${BUILD_NUMBER} || true
          '''
        }
      }
    }
  }

  // =========================================================
  // Post (matches screenshot)
  // =========================================================
  post {
    success {
      echo '🎉 Pipeline completed successfully!'
      echo '📊 Test results and coverage reports have been archived.'
    }
    failure {
      echo '❌ Pipeline failed!'
      echo 'Check the console output above for error details.'
    }
    always {
      echo '🔚 Pipeline execution finished.'
      echo "🧼 Virtual environment /tmp/venv-${BUILD_NUMBER} will be cleaned up automatically."
    }
  }
}
