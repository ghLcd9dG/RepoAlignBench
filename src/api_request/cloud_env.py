import sys
import subprocess


def install_packages():
    packages = [
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
        "scikit-learn",
        "flask",
        "django",
        "requests",
        "beautifulsoup4",
        "lxml",
        "pytest",
        "jupyter",
        "notebook",
        "ipython",
        "pyyaml",
        "opencv-python",
        "pillow",
        "sqlalchemy",
        "pymongo",
        "openai",
    ]

    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} done!\n")
        except subprocess.CalledProcessError:
            print(f"Error while installing {package} \n")


if __name__ == "__main__":
    install_packages()
