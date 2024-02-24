#!/bin/sh

docker build - < .devcontainer/Dockerfile --tag ccmagruder/runpod-symmetry --platform linux/amd64
docker push ccmagruder/runpod-symmetry
