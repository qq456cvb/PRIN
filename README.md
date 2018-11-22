# PRIN
## Pointwise Rotation-Invariant Network in PyTorch

## Overview
This repository is the Pytorch implementation of PRIN (Pointwise Rotation-Invariant Network).
## Dependencies
* Install s2cnn (https://github.com/jonas-koehler/s2cnn) and its dependencies （pytorch, cupy, lie_learn, pynvrtc).

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

