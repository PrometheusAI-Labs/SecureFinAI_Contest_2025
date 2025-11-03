#!/bin/bash
# Setup script for redirecting Hugging Face cache to /dev/shm
# This helps avoid "No space left on device" errors

set -e

CACHE_DIR="/dev/shm/cache"
HF_CACHE_DIR="$CACHE_DIR/huggingface"
USER_CACHE_DIR="$HOME/.cache"

echo "Setting up Hugging Face cache on /dev/shm..."

# Create cache directories
mkdir -p "$HF_CACHE_DIR"/{hub,datasets,xet}

# Move existing cache if it exists and is not a symlink
if [ -d "$USER_CACHE_DIR/huggingface" ] && [ ! -L "$USER_CACHE_DIR/huggingface" ]; then
    echo "Moving existing cache from $USER_CACHE_DIR/huggingface to $HF_CACHE_DIR..."
    mv "$USER_CACHE_DIR/huggingface"/* "$HF_CACHE_DIR"/ 2>/dev/null || true
    rmdir "$USER_CACHE_DIR/huggingface" 2>/dev/null || true
fi

# Create symlink
if [ ! -e "$USER_CACHE_DIR/huggingface" ]; then
    mkdir -p "$USER_CACHE_DIR"
    ln -sf "$HF_CACHE_DIR" "$USER_CACHE_DIR/huggingface"
    echo "Created symlink: $USER_CACHE_DIR/huggingface -> $HF_CACHE_DIR"
fi

# Set environment variables for this session
export HF_HOME="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR/hub"

echo ""
echo "Cache setup complete!"
echo "HF_HOME=$HF_CACHE_DIR"
echo "HF_DATASETS_CACHE=$HF_CACHE_DIR/datasets"
echo "TRANSFORMERS_CACHE=$HF_CACHE_DIR/hub"
echo ""
echo "Current disk usage:"
df -h / | tail -1
df -h /dev/shm | tail -1
echo ""
echo "To make these environment variables permanent, add to your ~/.bashrc:"
echo "export HF_HOME=$HF_CACHE_DIR"
echo "export HF_DATASETS_CACHE=$HF_CACHE_DIR/datasets"
echo "export TRANSFORMERS_CACHE=$HF_CACHE_DIR/hub"

