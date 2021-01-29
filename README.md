# PRIN
## Pointwise Rotation-Invariant Network in PyTorch

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
@article{prin,  
&emsp;&emsp;title={PRIN: Pointwise Rotation-Invariant Network},  
&emsp;&emsp;author={Yang You and Yujing Lou and Qi Liu and Yu-Wing Tai and Weiming Wang and Lizhuang Ma and Cewu Lu},  
&emsp;&emsp;journal={arXiv preprint arXiv:1811.09361},  
&emsp;&emsp;year={2018}  
}
