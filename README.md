# Monodepth2.jl

## Monodepth

Monocular depth estimation. Using single image to predict disparity map.

TODO visualization, model info, etc.

## Simple disparity estimation

Simple disparity estimation using gradient descent with parameters:

- disparity map;
- rotation vector (so3);
- translation vector.

![Triplet](./res/image.png)

Visualization of the disparity map learning dynamics for the triplet above.

![Depth](./res/output.gif)

## Important

Norm function is computed using `sqrt.(sum(abs2, ...))`. However, `sqrt` function has `NaN` gradient at `0`.
This can be mitigated by defining subgradient or even better, `norm` function that can act on the given axis, similar to PyTorch:
https://github.com/pytorch/pytorch/issues/37354
