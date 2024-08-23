# Exit on any error
$ErrorActionPreference = "Stop"

Write-Output "Setting up the environment for Windows..."

# Check if running as Administrator
If (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Output "Please run this script as an Administrator"
    Exit 1
}

# Install Chocolatey (if not installed)
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
}

# Install basic dependencies
choco install -y visualstudio2019buildtools cmake wget 7zip

# Install CUDA Toolkit 12.2
$cudaInstaller = "cuda_12.2.0_535.54.03_windows.exe"
$cudaUrl = "https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/$cudaInstaller"
Invoke-WebRequest $cudaUrl -OutFile $cudaInstaller
Start-Process -FilePath $cudaInstaller -ArgumentList "--silent", "--toolkit" -Wait
Remove-Item $cudaInstaller

# Install cuDNN 8.9.7 (You must download the file from NVIDIA's site)
$cudnnFile = "cudnn-windows-x86_64-8.9.7.29_cuda12-archive.zip"
If (-Not (Test-Path $cudnnFile)) {
    Write-Output "Please download cuDNN from NVIDIA's website and place it in this directory."
    Exit 1
}
7z x $cudnnFile -o"$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\v12.2" -aoa
Remove-Item $cudnnFile

# Install PyTorch C++ (libtorch) with CUDA 12.2
$libtorchUrl = "https://download.pytorch.org/libtorch/cu122/libtorch-win-shared-with-deps-2.1.0%2Bcu122.zip"
Invoke-WebRequest $libtorchUrl -OutFile "libtorch.zip"
7z x "libtorch.zip" -o"C:\libtorch"
Remove-Item "libtorch.zip"

# Set up environment variables
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin;C:\libtorch\lib", [EnvironmentVariableTarget]::Machine)
[Environment]::SetEnvironmentVariable("CUDA_PATH", "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\v12.2", [EnvironmentVariableTarget]::Machine)

# Build the project (Assumes CMakeLists.txt is present in the current directory)
mkdir build
cd build
cmake ..
cmake --build . --config Release

Write-Output "Setup complete! You can now run the project."