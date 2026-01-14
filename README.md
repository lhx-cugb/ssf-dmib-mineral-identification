## Robust Multimodal Mineral Identification based on a Parameter-Efficient Method

This repository is the open-source implementation of the paper "Robust Multimodal Mineral Identification based on a Parameter-Efficient Method".
This project proposes a multimodal mineral identification framework that integrates Scale and Shift Feature Adaptation (SSF) and Dynamic Multimodal Information Bottleneck (DMIB). It addresses issues such as "modality missing" and "modality data errors" in practical geological scenarios. While ensuring identification accuracy, it only introduces a 13.8% increase in parameter volume, making it highly practical and scalable.



## 1.Repo Structure

- `SSF_Net_v7_error_3mo_v4.py`: Training script for SSF+DMIB with 3 modalities (images + hardness, luster, streak), including data loading, training, validation, and checkpoint saving
- `SSF_Net_v7_error_2mo_12_v4_v3.py / _13_ / _23_` : Training scripts for 2-modality combinations
- `SSF_Net_v7_error_1mo_1_v4_v3.py / _2_ / _3_`: Training scripts for 1-modality combination
- `SSF_Net_v7_error_0mo_v4_v3.py`: Training scripts for 0-modality / single-modality (with only  image)
- `loss_modules.py`: **Collection of loss functions** (e.g., `logit_DKD`, `DKDloss`, `RKDLoss`, `MMD`, `CMD`, `Focal_loss`, etc.)
- `meta_encoder.py`: **Modality encoders / residual layers**, etc. (e.g., `ResNormLayer`, `SharedEncoder`, `ResidualAE`)
- `requirements.txt`: Development Environment Dependencies



## 2.Installation

### 2.1 Recommended Configuration

- Python: >= 3.8 (3.9/3.10 is recommended)
- PyTorch: 2.x is recommended (must match the CUDA version)
- GPU: Recommended (CPU is supported but runs extremely slowly)

### 2.2 Conda Installation

```bash
conda create -n ssf-dmib python=3.10 -y
conda activate ssf-dmib
```

### 2.3 Pytorch Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2.4 Minimum dependencies (covering only the core packages actually used by the scripts in this repository)

```bash
pip install numpy pillow tqdm loguru tensorboard timm torchtoolbox
```



## 3.Dataset Preparation

**Image directory**: `img_root` (hard-coded in the script by default as `/root/autodl-tmp/images`)

**Annotation directory**: `meta_root` (hard-coded in the script by default as `/root/autodl-tmp/meta`), which should contain at least:

- `train.txt`
- `val.txt`

### 3.1 train.txt / val.txt Format

Each line is in the form of

```
<image_id>/<class_name>
```



## 4. How to Run

### 4.1 Run 3-modality (main version example)

```python
python SSF_Net_v7_error_3mo_v4.py
```

### 4.2 Run 0/1/2-modality (main version example)

```python
python SSF_Net_v7_error_0mo_v4_v3.py
python SSF_Net_v7_error_1mo_1_v4_v3.py
python SSF_Net_v7_error_2mo_12_v4_v3.py
# The rest of the scripts follow the same logic
```



### 4.3 Key Configuration

-  `img_root` / `meta_root`: Data paths (different for Windows/Linux) 

- `batch_size`: Default 64 

- `epoches`: Default 100

- `learning_rate`: Default 0.01 

- `error_rate`: e.g., 0.3 / 0.6 (varies for different experiments in the script) 

- `model_weight`: Path to pre-trained weights

-  `model_name`: Prefix for output checkpoints Output files 

  

  (checkpoints) are saved in the running directory by default:

  ```
  {model_name}_epoch_{epoch}_ckt.pth
  ```



## 5.Test set example

We have provided test set files that can be directly used for testing. You can modify the code in this repository to complete the test verification:[TestDataSet(OneDrive)](https://1drv.ms/f/c/c52c056caa61004f/IgAtIXZcc9CmT5bNXYT5PywAAW7vvjuHqkO_66bsIOIZ72s?e=cgvEkH) / [TestDataSet(BaiDu)](https://pan.baidu.com/s/1gMKJ6UFfDmrZPeffiBuN4A?pwd=6335 )



- The test set is in tar compressed format. After downloading, you can decompress and use it with the command `tar -xf filename.tar`. If you need to adjust the test logic, you can modify the path/parameter configuration of the corresponding test scripts in the repository.



