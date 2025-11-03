#!/bin/bash
# Script to set up Hugging Face token for authentication

set -e

echo "=== Hugging Face Token Setup ==="
echo ""

# Method 1: Check if already logged in via CLI
if command -v huggingface-cli &> /dev/null; then
    if huggingface-cli whoami &> /dev/null; then
        echo "✓ Already logged in via huggingface-cli"
        echo "  User: $(huggingface-cli whoami 2>/dev/null)"
        exit 0
    fi
fi

# Method 2: Check for token file
if [ -f ~/.huggingface/token ]; then
    echo "✓ Token file found at ~/.huggingface/token"
    TOKEN=$(cat ~/.huggingface/token | tr -d '\n')
    export HF_TOKEN="$TOKEN"
    export HUGGINGFACE_TOKEN="$TOKEN"
    echo "  Token loaded from file"
    exit 0
fi

# Method 3: Check environment variables
if [ -n "$HF_TOKEN" ]; then
    echo "✓ HF_TOKEN found in environment"
    exit 0
elif [ -n "$HUGGINGFACE_TOKEN" ]; then
    echo "✓ HUGGINGFACE_TOKEN found in environment"
    export HF_TOKEN="$HUGGINGFACE_TOKEN"
    exit 0
fi

# If no token found, prompt user
echo "No Hugging Face token found!"
echo ""
echo "To authenticate, choose one of these methods:"
echo ""
echo "Method 1: Login via CLI (recommended)"
echo "  huggingface-cli login"
echo ""
echo "Method 2: Set environment variable"
echo "  export HF_TOKEN='your-token-here'"
echo "  # Add to ~/.bashrc for persistence:"
echo "  echo 'export HF_TOKEN=\"your-token-here\"' >> ~/.bashrc"
echo ""
echo "Method 3: Create token file"
echo "  mkdir -p ~/.huggingface"
echo "  echo 'your-token-here' > ~/.huggingface/token"
echo ""
echo "Get your token from: https://huggingface.co/settings/tokens"
echo ""
exit 1

