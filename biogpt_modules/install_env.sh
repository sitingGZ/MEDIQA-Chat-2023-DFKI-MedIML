#!/bin/bash

# Install required packages for PyTorch and CUDA
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev build-essential cmake git libopenblas-dev libblas-dev libboost-all-dev libjpeg-dev zlib1g-dev libpng-dev libffi-dev libgmp-dev libssl-dev libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev liblzma-dev libffi-dev liblzma-dev wget

# Install the latest version of PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
