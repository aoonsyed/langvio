from setuptools import setup, find_packages

setup(
    name="langvio",
    version="0.1.0",
    description="Connect language models to vision models for natural language visual analysis",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.18.0",
        "opencv-python>=4.5.0",
        "pyyaml>=5.4.0",
        "langchain>=0.0.267",
        "ultralytics>=8.0.0",
        "pillow>=8.0.0",
    ],
    extras_require={
        "openai": ["openai>=0.27.0"],
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "isort>=5.9.1",
            "flake8>=3.9.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "langvio=langvio.cli:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)