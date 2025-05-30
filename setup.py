from setuptools import setup, find_packages

setup(
    name="ocr-rag",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pytesseract>=0.3.10",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "scikit-image>=0.21.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0",
    ],
) 