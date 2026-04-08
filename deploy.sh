#!/bin/bash
set -e
echo "Deploying Agri Alert RL to Hugging Face Spaces..."
huggingface-cli login
huggingface-cli upload khaiwal009/agri-alert-rl . . --repo-type space
echo "Deployment complete. Visit: https://huggingface.co/spaces/khaiwal009/agri-alert-rl"
