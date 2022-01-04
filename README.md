# Monodepth2.jl

Implementation of the ["Digging Into Self-Supervised Monocular Depth Estimation"](https://arxiv.org/abs/1806.01260) paper.

## Monodepth

Monocular depth estimation. Using single image to predict disparity map.

![Depth](./res/depth-kitti.gif)

Training parameters:
- resolution `416x128`;
- ResNet 18 model;
- no automasking & using pose prediction network.

### Supported datasets

- [KITTI odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
- [CommaAI Depth10k](https://github.com/commaai/depth10k)

### Install

Install model, image-augmentation library and the package itself:

```bash
]add https://github.com/pxl-th/ResNet.jl.git
]add https://github.com/pxl-th/Augmentations.jl.git
]add https://github.com/pxl-th/Monodepth2.jl.git
```

## Simple disparity estimation

Simple disparity estimation using gradient descent with parameters:

- disparity map;
- rotation vector (so3);
- translation vector.

![Triplet](./res/image.png)

Visualization of the disparity map learning dynamics for the triplet above.

![Depth](./res/simple-depth.gif)

## Important

- Norm function is computed using `sqrt.(sum(abs2, ...))`.
However, `sqrt` function has `NaN` gradient at `0`.
This can be mitigated by defining subgradient or even better,
`norm` function that can act on the given axis,
[similar to PyTorch](https://github.com/pytorch/pytorch/issues/37354).

- For poses, struct `Pose` is used instead of arrays or tuple because
of [this issue](https://github.com/FluxML/Zygote.jl/issues/522).
