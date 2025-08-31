"""Setup script for the project."""
import platform
import subprocess
import sys
from pathlib import Path


def get_platform_specific_paths() -> tuple[Path, Path, Path]:
    """Get platform-specific paths for virtual environment.

    Returns:
        Tuple containing paths to virtual environment, Python and pip executables.
    """
    is_windows = platform.system() == "Windows"
    scripts_dir = "Scripts" if is_windows else "bin"
    venv_path = Path(".venv")
    suffix = ".exe" if is_windows else ""

    python_executable = venv_path / scripts_dir / f"python{suffix}"
    pip_executable = venv_path / scripts_dir / f"pip{suffix}"

    return venv_path, python_executable, pip_executable


def setup_virtual_environment() -> None:
    """Set up a virtual environment and install required packages."""
    venv_path, _, pip_executable = get_platform_specific_paths()

    # Create virtual environment if it doesn't exist
    if not venv_path.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv",
                       str(venv_path)], check=True)

    print("Installing required packages...")

    # Upgrade pip first
    subprocess.run([str(pip_executable), "install",
                   "--upgrade", "pip"], check=True)

    # Install packages from requirements.txt
    requirements_file = Path("requirements.txt")
    assert requirements_file.exists()
    subprocess.run([str(pip_executable), "install", "-r",
                   str(requirements_file)], check=True)
    print("Setup completed successfully!")


if __name__ == "__main__":
    setup_virtual_environment()
