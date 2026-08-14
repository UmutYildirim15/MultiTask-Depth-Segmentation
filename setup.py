from setuptools import setup, find_packages

setup(
    name="multitask-depth-segmentation",
    version="0.1.0",
    author="UMUT YILDIRIM",
    description="Multi-Task Learning for Semantic Segmentation and Monocular Depth Estimation",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "albumentations",
        "transformers",
        "timm",
        "tqdm",
        "matplotlib"
    ],
)