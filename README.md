# DeepVO: Visual Odometry with Deep Learning

This project aims to recreate the **DeepVO (Deep Visual Odometry)** network, a deep learning model designed for estimating camera motion from consecutive image frames. Visual odometry is an essential technique in robotics and autonomous systems, enabling position and trajectory estimation using visual data.

DeepVO combines convolutional neural networks (CNNs) for feature extraction with recurrent neural networks (RNNs) for temporal modeling. This project uses pre-trained FlowNet weights and trains the model on the KITTI odometry dataset to replicate and evaluate its performance.

---

## Features
- **Visual Odometry**: Estimate the camera's position and orientation in 3D space.
- **Deep Learning-Based Approach**: Leverages CNNs for feature extraction and RNNs for sequence modeling.
- **Pre-trained FlowNet Weights**: Utilizes FlowNet for optical flow feature extraction.
- **KITTI Dataset**: Training and evaluation on real-world autonomous driving data.

---

## Instructions

### 1. Download and Prepare the Dataset
- Download the **odometry dataset (color)** (65 GB) and **odometry ground truth poses** (4 MB) from the [KITTI dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).
- Save the downloaded files in the **`dataset`** folder.

### 2. Download Pre-trained Weights
- Download the weights for the **FlowNet** model, which will be used for the CNN layers of DeepVO.
- Use the file **`flownets_bn_EPE2.459.pth`** from this [Google Drive link](https://drive.google.com/drive/folders/16eo3p9dO_vmssxRoZCmWkTpNjKRzJzn5?dmr=1&ec=wgc-drive-globalnav-goto).
- Save the downloaded weights in the **`pretrained_model_weights`** folder.

### 3. Run the Notebook
- Open the **`DeepVO.ipynb`** notebook in Jupyter Notebook or a similar environment.
- Train the model for **200 epochs** using the prepared dataset and pre-trained weights.
- Evaluate the model on test sequences provided in the KITTI dataset.


###  Preprocess Images (Optional)
- Use the provided `preprocess_data.py` script to preprocess images by subtracting their mean RGB values. This step normalizes the data but **does not seem to significantly improve performance** in this case, so it is optional.
- To preprocess images, run:

## Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA-enabled GPU (recommended for faster training)
- Required Python packages (install via `pip install -r requirements.txt`):
  - `torch`
  - `torchvision`
  - `numpy`
  - `matplotlib`
  - `opencv-python`

---

## References
- KITTI Dataset: [https://www.cvlibs.net/datasets/kitti/](https://www.cvlibs.net/datasets/kitti/)
- FlowNet Pre-trained Weights: [Google Drive](https://drive.google.com/drive/folders/16eo3p9dO_vmssxRoZCmWkTpNjKRzJzn5?dmr=1&ec=wgc-drive-globalnav-goto)
- DeepVO Paper: [https://arxiv.org/abs/1709.08429](https://arxiv.org/abs/1709.08429)

