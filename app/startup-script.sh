#!/bin/bash
VERSION=2.0.2
OS=linux  # or "darwin" for macOS, "windows" for Windows.
ARCH=amd64  # or "386" for 32-bit OSs, "arm64" for ARM 64.

apt-get update
apt-get install -y curl tar



# Download the docker-credential-gcr binary and set it as executable.
curl -fsSL "https://github.com/GoogleCloudPlatform/docker-credential-gcr/releases/download/v${VERSION}/docker-credential-gcr_${OS}_${ARCH}-${VERSION}.tar.gz" \
  | tar xz --to-stdout ./docker-credential-gcr \
  > /usr/bin/docker-credential-gcr && chmod +x /usr/bin/docker-credential-gcr

# Configure Docker to use docker-credential-gcr for GCR registries.
docker-credential-gcr configure-docker

apt-get install -y docker.io
sudo docker run --restart=always -p 8080:8080 asia-northeast3-docker.pkg.dev/primeval-span-410215/cloud-run-source-deploy/test 
