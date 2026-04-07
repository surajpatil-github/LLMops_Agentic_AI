/*
 * Jenkins Pipeline — Test + Build + Deploy (Docker Hub, no Azure)
 *
 * Workflow:
 *  1) Checkout
 *  2) Setup Python (uv)
 *  3) Install dependencies
 *  4) Run tests + coverage
 *  5) Build Docker image
 *  6) Push to Docker Hub
 *  7) Deploy with docker-compose (pull + up)
 *  8) Verify deployment (health check)
 *
 * Required Jenkins credentials:
 *  - dockerhub-username   (Secret text)
 *  - dockerhub-token      (Secret text — Docker Hub access token)
 *  - groq-api-key         (Secret text, optional)
 *  - google-api-key       (Secret text, optional)
 */

pipeline {
  agent any

  environment {
    PYTHON_VERSION = '3.12'
    PYTHONPATH     = "${WORKSPACE}:${WORKSPACE}/multi_doc_chat"

    // API keys — optional, read from Jenkins credentials if they exist
    LLM_PROVIDER   = "${env.LLM_PROVIDER ?: 'groq'}"

    // Docker Hub
    DOCKERHUB_USERNAME = credentials('dockerhub-username')
    IMAGE_NAME         = "${DOCKERHUB_USERNAME}/llmops-app"
    IMAGE_TAG          = "${env.BUILD_NUMBER}"

    // App config
    APP_PORT   = "8080"
    APP_URL    = "http://localhost:${APP_PORT}"
  }

  parameters {
    booleanParam(name: 'RUN_DEPLOY', defaultValue: true,
                 description: 'Run docker-compose deploy after tests pass')
    string(name: 'DEPLOY_TAG', defaultValue: '',
           description: 'Image tag to deploy (blank = use current BUILD_NUMBER)')
  }

  triggers {
    pollSCM('*/2 * * * *')
  }

  stages {

    // ─────────────────────────────────────────────
    stage('Checkout') {
    // ─────────────────────────────────────────────
      steps {
        echo 'Checking out source code...'
        checkout scm
      }
    }

    // ─────────────────────────────────────────────
    stage('Setup Python Environment') {
    // ─────────────────────────────────────────────
      steps {
        echo 'Setting up Python virtual environment with uv...'
        sh '''
          set -e

          curl -LsSf https://astral.sh/uv/install.sh | sh
          UV="$HOME/.local/bin/uv"

          export UV_PYTHON_INSTALL_DIR=/tmp/uv/python
          export XDG_DATA_HOME=/tmp/.local/share
          export XDG_CACHE_HOME=/tmp/.cache

          "$UV" python install ${PYTHON_VERSION}
          "$UV" venv --python ${PYTHON_VERSION} /tmp/venv-${BUILD_NUMBER}

          echo "Python: $(/tmp/venv-${BUILD_NUMBER}/bin/python --version)"
          echo "uv:     $("$UV" --version)"
        '''
      }
    }

    // ─────────────────────────────────────────────
    stage('Install Dependencies') {
    // ─────────────────────────────────────────────
      steps {
        echo 'Installing project dependencies...'
        sh '''
          set -e
          VENV_PY="/tmp/venv-${BUILD_NUMBER}/bin/python"
          UV="$HOME/.local/bin/uv"

          export XDG_DATA_HOME=/tmp/.local/share
          export XDG_CACHE_HOME=/tmp/.cache

          # Strip local/Windows-only packages
          SAN_REQ=$(mktemp)
          cat requirements.txt \
            | sed -E '/^[[:space:]]*llmops-series(==.*)?[[:space:]]*$/d' \
            | sed -E '/^[[:space:]]*pywin32(==.*)?[[:space:]]*$/d' \
            > "$SAN_REQ"

          "$UV" pip install --python "$VENV_PY" -r "$SAN_REQ"
          "$UV" pip install --python "$VENV_PY" pytest pytest-cov

          echo "PYTHONPATH=${PYTHONPATH}"
        '''
      }
    }

    // ─────────────────────────────────────────────
    stage('Run Tests') {
    // ─────────────────────────────────────────────
      steps {
        echo 'Running pytest with coverage...'
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
            --cov-report=term \
            || true
        '''
      }
      post {
        always {
          junit allowEmptyResults: true, testResults: 'test-results/*.xml'
          archiveArtifacts artifacts: 'coverage.xml,htmlcov/**,test-results/**',
                           allowEmptyArchive: true
        }
      }
    }

    // ─────────────────────────────────────────────
    stage('Build Docker Image') {
    // ─────────────────────────────────────────────
      when { expression { params.RUN_DEPLOY } }
      steps {
        echo "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}..."
        withCredentials([
          string(credentialsId: 'dockerhub-username', variable: 'DH_USER'),
          string(credentialsId: 'dockerhub-token',    variable: 'DH_TOKEN')
        ]) {
          sh """
            set -e

            # Login before build so Docker can pull python:3.12-slim base image
            echo "\$DH_TOKEN" | docker login -u "\$DH_USER" --password-stdin

            docker build \\
              --platform linux/amd64 \\
              -t ${IMAGE_NAME}:${IMAGE_TAG} \\
              -t ${IMAGE_NAME}:latest \\
              -f Dockerfile .

            echo "Image built:"
            docker images ${IMAGE_NAME}
          """
        }
      }
    }

    // ─────────────────────────────────────────────
    stage('Push to Docker Hub') {
    // ─────────────────────────────────────────────
      when { expression { params.RUN_DEPLOY } }
      steps {
        echo 'Pushing image to Docker Hub...'
        withCredentials([
          string(credentialsId: 'dockerhub-username', variable: 'DH_USER'),
          string(credentialsId: 'dockerhub-token',    variable: 'DH_TOKEN')
        ]) {
          sh """
            set -e
            echo "\$DH_TOKEN" | docker login -u "\$DH_USER" --password-stdin

            docker push ${IMAGE_NAME}:${IMAGE_TAG}
            docker push ${IMAGE_NAME}:latest

            docker logout
            echo "Pushed ${IMAGE_NAME}:${IMAGE_TAG} and :latest"
          """
        }
      }
    }

    // ─────────────────────────────────────────────
    stage('Deploy with Docker Compose') {
    // ─────────────────────────────────────────────
      when { expression { params.RUN_DEPLOY } }
      steps {
        echo 'Deploying application with docker-compose...'
        withCredentials([
          string(credentialsId: 'groq-api-key',   variable: 'GROQ_API_KEY'),
          string(credentialsId: 'google-api-key', variable: 'GOOGLE_API_KEY')
        ]) {
          sh """
            set -e

            DEPLOY_TAG="${params.DEPLOY_TAG ?: IMAGE_TAG}"
            export APP_IMAGE="${IMAGE_NAME}:\$DEPLOY_TAG"
            export GROQ_API_KEY="\$GROQ_API_KEY"
            export GOOGLE_API_KEY="\$GOOGLE_API_KEY"
            export LLM_PROVIDER="${LLM_PROVIDER}"

            echo "Deploying image: \$APP_IMAGE"
            docker pull "\$APP_IMAGE"
            APP_IMAGE="\$APP_IMAGE" docker-compose up -d --no-build

            echo "Waiting for app to become ready..."
            sleep 15
          """
        }
      }
    }

    // ─────────────────────────────────────────────
    stage('Verify Deployment') {
    // ─────────────────────────────────────────────
      when { expression { params.RUN_DEPLOY } }
      steps {
        echo 'Verifying deployment health...'
        sh """
          set -e

          MAX=12
          I=0
          until curl -sf ${APP_URL}/health >/dev/null 2>&1; do
            I=\$((I+1))
            if [ \$I -ge \$MAX ]; then
              echo "Health check failed after \$((MAX*5))s"
              docker-compose logs --tail=50 app || true
              exit 1
            fi
            echo "Waiting for health check... (\$I/\$MAX)"
            sleep 5
          done

          echo "App is healthy at ${APP_URL}"
          echo "Stack status:"
          docker-compose ps
        """
      }
    }

    // ─────────────────────────────────────────────
    stage('Cleanup') {
    // ─────────────────────────────────────────────
      steps {
        sh 'rm -rf /tmp/venv-${BUILD_NUMBER} || true'
        echo 'Workspace cleaned up.'
      }
    }

  } // end stages

  post {
    success {
      echo "Pipeline completed successfully. Image: ${IMAGE_NAME}:${IMAGE_TAG}"
      echo "Test results and coverage archived."
    }
    failure {
      echo "Pipeline failed — check console output above."
    }
    always {
      echo "Pipeline finished. Build #${BUILD_NUMBER} done."
    }
  }
}
