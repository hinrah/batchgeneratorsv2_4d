# 4D batchgeneratiors

 **This is a fork of [batchgenerators](https://github.com/MIC-DKFZ/batchgeneratorsv2)**, originally
 developed by the Division of Medical Image Computing, German Cancer Research
 Center (DKFZ), and licensed under the Apache License 2.0.

 This fork extends nnU-Net with 4D convolution support. See
 [CHANGES.md](CHANGES.md) for a list of modifications.

 This project is **not affiliated with, endorsed by, or maintained by** the
 original nnU-Net authors or the DKFZ. Please direct issues with this fork here,
 not to the upstream repository.

 Large parts of this README are derived from the original nnU-Net documentation.

# batchgeneratorsv2
This repository is work in progress. If builds upon the [batchgenerators](https://github.com/MIC-DKFZ/batchgenerators) 
framework but makes several key changes to the transforms:

1. Transforms now explicitly distinguish between data types: images, segmentation, pixel-wise regression target, keypoints, bbox
2. All transforms have been reimplemented from scratch with a focus on performance. In case of performance parity 
between previous numpy and new torch-based implementations, preference is given to pytorch. 
3. Transforms are applied on a sample level, not a batch level as was done previously!

Caveats:
- performance is optimized for CPU. GPU-based data augmentation is not supported (implementation may use numpy etc) and will not be supported
- currently this repository only covers a small subset of the transforms available in batchgenerators. Feel free to contribute more


# Acknowledgements

 **This is a fork of [batchgenerators](https://github.com/MIC-DKFZ/batchgeneratorsv2)**, originally
 developed by the Division of Medical Image Computing, German Cancer Research
 Center (DKFZ), and licensed under the Apache License 2.0.

 This fork extends nnU-Net with 4D convolution support. See
 [CHANGES.md](CHANGES.md) for a list of modifications.

 This project is **not affiliated with, endorsed by, or maintained by** the
 original nnU-Net authors or the DKFZ. Please direct issues with this fork here,
 not to the upstream repository.
