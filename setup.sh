#!/bin/bash

# Exit on any error
set -e

echo "Setting up the environment for Linux..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root or using sudo"
  exit 1
fi

# Update package list and upgrade existing packages
apt update && apt upgrade -y

# Install basic dependencies
apt install -y build-essential cmake wget unzip libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev freeglut3-dev

# Remove any existing NVIDIA drivers
apt autoremove nvidia* --purge -y

# Install NVIDIA drivers automatically
ubuntu-drivers autoinstall
sudo reboot

# Install CUDA Toolkit 12.2
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get update
apt-get -y install cuda

# Set up environment variables for CUDA
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
ldconfig

# Install cuDNN 8.9.7 (You must download the file from NVIDIA's site)
CUDNN_FILE="cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz"
if [ ! -f "$CUDNN_FILE" ]; then
    echo "Please download cuDNN from NVIDIA's website and place it in this directory."
    exit 1
fi
tar -xvf "$CUDNN_FILE"
cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
rm -rf cudnn-*-archive
rm "$CUDNN_FILE"

# Install cJSON
apt install -y libcjson-dev

# Install PyTorch C++ (libtorch) with CUDA 12.2
wget https://download.pytorch.org/libtorch/cu122/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu122.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu122.zip -d /usr/local
rm libtorch-cxx11-abi-shared-with-deps-2.1.0+cu122.zip

# Set up environment variables for PyTorch
echo 'export LD_LIBRARY_PATH=/usr/local/libtorch/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Build the project
mkdir -p build && cd build
cmake ..
make -j$(nproc)

echo "Setup complete! You can now run the project."