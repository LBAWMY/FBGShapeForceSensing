# FBGShapeForceSensing

## Introduction
This project is to acheve the shape and force sensing of a soft robot with FBG sensors. The sensing process is based on the deep learning method. The training process is based on the [PyTorch](https://pytorch.org/) framework.
### Pre-requisites
1. Create a virtual environment with Python 3.9
    ```shell
    conda create -n sfs python=3.9 -y
    conda activate sfs
    ```
2. Install dependencies
    ```shell
    pip install -r requirements.txt
    ```
### Get started
1. Run the training process
    ```shell
    python main.py
    ```
2. Run the inference process
    ```shell
    python demo.py
    ```
## License

FBGShapeForceSensing is released under the [MIT license](LICENSE).