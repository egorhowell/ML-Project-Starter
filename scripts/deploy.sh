#!/bin/bash
# Pulls the latest code, refreshes dependencies if they changed, and restarts the
# Streamlit service. Triggered by .github/workflows/deploy.yml on every push to main.
#
# Before this script can run you need:
#   1. A VPS with this repo cloned at $REPO_PATH (see README "VPS setup")
#   2. A systemd service called $SERVICE_NAME that runs your Streamlit app
#   3. Poetry installed at /root/.local/bin/poetry (the default install location)

set -e  # Exit immediately if any command fails.

# EDIT THIS: point to where you cloned the repo on your VPS.
REPO_PATH="/root/ml-project-starter"
# EDIT THIS: the name of the systemd service running your Streamlit app.
SERVICE_NAME="streamlit-app.service"

echo "Navigating to repository..."
cd "$REPO_PATH"

echo "Pulling latest code from GitHub..."
git pull

# Only re-install dependencies if pyproject.toml changed in this push.
# Saves time on deploys that only changed application code.
if git diff --name-only HEAD@{1} HEAD | grep -q "pyproject.toml"; then
    echo "pyproject.toml changed. Installing new dependencies..."
    /root/.local/bin/poetry install --without dev
fi

echo "Restarting systemd service: $SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

echo "Deployment complete."
