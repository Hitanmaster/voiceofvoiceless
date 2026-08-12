import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

REQUIRED_PACKAGES = [
    "opencv-python",
    "pandas",
    "numpy",
    "gTTS",
    "scikit-learn",
    "mediapipe"
]

def main():
    print("Checking and installing required dependencies...\n")
    for package in REQUIRED_PACKAGES:
        try:
            print(f"Installing {package}...")
            install(package)
            print(f"Successfully installed {package}.\n")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}. Error: {e}")
            sys.exit(1)

    print("All dependencies have been successfully installed!")

if __name__ == "__main__":
    main()
