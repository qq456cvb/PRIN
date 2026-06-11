# PRIN: Pointwise Rotation-Invariant Network (AAAI 2020)

PyTorch implementation of [Pointwise Rotation-Invariant Network with Adaptive Sampling and 3D Spherical Voxel Convolution (AAAI 2020)](https://ojs.aaai.org/index.php/AAAI/article/view/6965).

PRIN learns **point-wise, rotation-invariant features** for 3D point clouds: input points are converted to spherical voxels with density-aware adaptive sampling, processed with 3D spherical voxel convolutions (built on [s2cnn](https://github.com/jonas-koehler/s2cnn)), and mapped back to per-point features — so part segmentation works on arbitrarily rotated shapes without rotation augmentation.

> **News:** an improved version, **SPRIN**, achieves much better results and is released at [qq456cvb/SPRIN](https://github.com/qq456cvb/SPRIN). It is described in [PRIN/SPRIN: On Extracting Point-wise Rotation Invariant Features (IEEE TPAMI)](https://arxiv.org/abs/2102.12093).

## Dependencies

- [s2cnn](https://github.com/jonas-koehler/s2cnn) and its dependencies (PyTorch, cupy, lie_learn, pynvrtc).
- [pybind11](https://pybind11.readthedocs.io/): compile the sampling extension under `src/` with CMake.

## Dataset and Pretrained Weights

- Download the ShapeNet 17-category part segmentation dataset (h5py format) from [Google Drive](https://drive.google.com/drive/folders/1wC-DpeRtxuuEvffubWdhwoGXGeW052Vy?usp=sharing) and extract it to `./hdf5_data`.
- Download the pretrained weights (trained on unrotated shapes) from [Hugging Face](https://huggingface.co/qq456cvb/PRIN):

```bash
hf download qq456cvb/PRIN state.pkl --local-dir .
```

## Usage

Train:

```bash
python train.py --log_dir log --model_path ./model.py --num_workers 4
```

Test with pretrained weights:

```bash
python test.py --weight_path ./state.pkl --model_path ./model.py --num_workers 4
```

## License

MIT

## Citation

```bibtex
@inproceedings{you2020pointwise,
  title={Pointwise Rotation-Invariant Network with Adaptive Sampling and 3D Spherical Voxel Convolution},
  author={You, Yang and Lou, Yujing and Liu, Qi and Tai, Yu-Wing and Ma, Lizhuang and Lu, Cewu and Wang, Weiming},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={07},
  pages={12717--12724},
  year={2020}
}
```
