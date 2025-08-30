"""
Setup script for the face recognition project.
This script creates a virtual environment and installs required packages.
Cross-platform compatible for Windows, macOS, and Linux.
"""
import platform
import subprocess
import sys
from pathlib import Path

def get_platform_specific_paths():
    """Get platform-specific paths for virtual environment."""
    is_windows = platform.system() == "Windows"
    scripts_dir = "Scripts" if is_windows else "bin"
    venv_path = Path(".venv")
    python_executable = venv_path / scripts_dir / ("python.exe" if is_windows else "python")
    pip_executable = venv_path / scripts_dir / ("pip.exe" if is_windows else "pip")
    return venv_path, python_executable, pip_executable

def setup_virtual_environment():
    """Set up a virtual environment and install required packages."""
    venv_path, python_executable, pip_executable = get_platform_specific_paths()
    
    # Create virtual environment if it doesn't exist
    if not venv_path.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        
    print("Installing required packages...")
    
    # Upgrade pip first
    subprocess.run([str(pip_executable), "install", "--upgrade", "pip"], check=True)
    
    # Install packages using requirements.txt if it exists
    requirements_file = Path("requirements.txt")
    assert requirements_file.exists()
    subprocess.run([str(pip_executable), "install", "-r", str(requirements_file)], check=True)
    print("Setup completed successfully!")

if __name__ == "__main__":
    setup_virtual_environment()
