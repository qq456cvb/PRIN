# PRIN
## Pointwise Rotation-Invariant Network in PyTorch

# News
An improved version of PRIN (SPRIN) is released [here](https://github.com/qq456cvb/SPRIN) and described in [PRIN/SPRIN: On Extracting Point-wise Rotation Invariant Features](https://arxiv.org/abs/2102.12093), which achieves much better results.

## Overview
This repository is the Pytorch implementation of [PRIN (Pointwise Rotation-Invariant Network)](https://arxiv.org/pdf/1811.09361.pdf).
## Dependencies
* Install s2cnn (https://github.com/jonas-koehler/s2cnn) and its dependencies (pytorch, cupy, lie_learn, pynvrtc).
* Install pybind11 and compile the script under src (https://pybind11.readthedocs.io/)

## Dataset and pretrained weights
* Download ShapeNet 17 Part Segmentation Dataset in h5py format from 
https://drive.google.com/drive/folders/1wC-DpeRtxuuEvffubWdhwoGXGeW052Vy?usp=sharing
* Download pretrained weights (trained on unrotated shapes) from
https://drive.google.com/open?id=1QnFqQdWmx0cYtYeN9tJNlf-E5ZLawRBv
## Usage
* For training, run "python train.py --log_dir log --model_path ./model.py --num_workers 4"
* For testing, run "python test.py --weight_path ./state.pkl --model_path ./model.py --num_workers 4"
## License
MIT

## References
Our paper is available on https://arxiv.org/abs/1811.09361.

## Citation
@inproceedings{you2020pointwise,  
&emsp;&emsp;title={Pointwise rotation-invariant network with adaptive sampling and 3d spherical voxel convolution},  
&emsp;&emsp;author={You, Yang and Lou, Yujing and Liu, Qi and Tai, Yu-Wing and Ma, Lizhuang and Lu, Cewu and Wang, Weiming},  
&emsp;&emsp;booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},  
&emsp;&emsp;volume={34},  
&emsp;&emsp;number={07},  
&emsp;&emsp;pages={12717--12724},  
&emsp;&emsp;year={2020}  
}
