import setuptools

setuptools.setup(
    name="ActionRecognition",
    version="0.0.1",
    author="Parth Diwanji",
    author_email="diwanji.parth@gmail.com",
    description="Deep Learning model for action recognition in videos",
    url="https://github.com/Parth27/ActionRecognitionVideos",
    packages=setuptools.find_packages(),
    license='Apache License 2.0',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['pandas','numpy>=1.6.1','torch==1.5.1','bisect','scipy>=0.9','opencv-python==3.4.4','scikit-learn','glob'],
)
