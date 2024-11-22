# Introduction
This is modification based on (https://github.com/YadiraF/PRNet). Because of the implementation of the original code using python 2.x and TF 1.x, it hard using the code in python env 3.10.x and TF 2.x. So this repo will help implementation original code in the current environment

# Requirements
- Python 3.10.x
- Tensoflow 2.18.x
- numpy
- opencv2
- MTCNN
- You could download the model in [Google Drive](https://drive.google.com/file/d/1UoE-XuW1SDLUjZmJPkIZ1MLxvQFgmTFH/view) for original code (TF 1.14). Buat for TF 2.x, the model is in `Data` folder.

GPU is highly recommended. The run time is ~0.01s with GPU(GeForce GTX 1080) and ~0.2s with CPU(Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz).

# How to use
There is notebook to show in very simple way to implement the code to generate depht map, 3d map image, landmark, etc. 
```
how to use.ipynb
```
# CITATION
```
@inProceedings{wang2018fastd,
  title     = {Exploiting Temporal and Depth Information for Multi-frame Face Anti-spoofing},
  author    = {Zezheng Wang, Chenxu Zhao, Yunxiao Qin, Qiusheng Zhou, Guojun Qi, Jun Wan, Zhen Lei},
  booktitle = {arXiv:1811.05118},
  year      = {2018}
}
@inProceedings{feng2018prn,
  title     = {Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network},
  author    = {Yao Feng, Fan Wu, Xiaohu Shao, Yanfeng Wang, Xi Zhou},
  booktitle = {ECCV},
  year      = {2018}
}
```
