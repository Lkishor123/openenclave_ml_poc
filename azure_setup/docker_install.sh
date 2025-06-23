#!/bin/bash

# This script automates the installation of Docker Engine on Ubuntu.
# It is based on the official Docker documentation and common practices.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting Docker installation on Ubuntu..."

# Step 1: Update the apt package index
echo "1. Updating apt package index..."
sudo apt update -y || { echo "Failed to update apt. Exiting."; exit 1; }
echo "Apt package index updated."

# Step 2: Install necessary packages to allow apt to use repositories over HTTPS
echo "2. Installing prerequisite packages..."
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common || \
    { echo "Failed to install prerequisite packages. Exiting."; exit 1; }
echo "Prerequisite packages installed."

# Step 3: Add Docker's official GPG key
echo "3. Adding Docker's official GPG key..."
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg || \
    { echo "Failed to add Docker GPG key. Exiting."; exit 1; }
echo "Docker GPG key added."

# Step 4: Set up the stable repository
# The architecture is set to AMD64 and the release is set to 'focal' as per your reference dump.
# You might want to replace 'focal' with $(lsb_release -cs) for dynamic release detection.
echo "4. Setting up the Docker stable repository..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null || \
    { echo "Failed to add Docker repository. Exiting."; exit 1; }
echo "Docker repository added."

# Step 5: Update the apt package index again with the new repository
echo "5. Updating apt package index with Docker repository..."
sudo apt update -y || { echo "Failed to update apt after adding Docker repo. Exiting."; exit 1; }
echo "Apt package index updated again."

# Step 6: Install Docker Engine, containerd, and Docker Compose (if not already installed)
echo "6. Installing Docker Engine..."
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin || \
    { echo "Failed to install Docker Engine. Exiting."; exit 1; }
echo "Docker Engine installed."

# Step 7: Add the current user to the docker group
# This allows running docker commands without sudo.
echo "7. Adding current user '${USER}' to the 'docker' group..."
sudo usermod -aG docker "${USER}" || \
    { echo "Failed to add user to docker group. Exiting."; exit 1; }
echo "User '${USER}' added to the 'docker' group."
echo "IMPORTANT: For this change to take effect, you need to log out and log back in,"
echo "or run 'newgrp docker' in your current terminal session."

# Step 8: Verify the installation by running the hello-world image
echo "8. Verifying Docker installation (requires re-login or 'newgrp docker' for current session)..."
# This command might fail if the user hasn't re-logged in or used newgrp.
# We'll try to run it with sudo as a fallback for verification purposes.
if docker run hello-world > /dev/null 2>&1; then
    echo "Docker 'hello-world' ran successfully (without sudo)."
else
    echo "Attempting 'hello-world' with sudo for verification..."
    if sudo docker run hello-world > /dev/null 2>&1; then
        echo "Docker 'hello-world' ran successfully (with sudo)."
    else
        echo "Failed to run 'hello-world'. Please check your Docker installation."
    fi
fi

# Step 9: Display current groups for the user (for verification)
echo "9. Current groups for user '${USER}':"
groups "${USER}"

echo "Docker installation script finished."
echo "Remember to log out and log back in, or run 'newgrp docker' to use Docker without 'sudo'."
