# Multi-organ-segmentation-low-resource-environment
This is the implementation for the article Multi-organ segmentation in a low resource architecture by Valentin Ogrean and Remus Brad. 

The implementation uses the miscnn framework - https://github.com/frankkramer-lab/MIScnn
Müller, D., Kramer, F. MIScnn: a framework for medical image segmentation with convolutional neural networks and deep learning. BMC Med Imaging 21, 12 (2021).
DOI: https://doi.org/10.1186/s12880-020-00543-7

- segthor-***.py -> training DL networks using different strategies (single, multi-organ)
- segthor-evaluation-***.py -> automatic segmentation generation based on trained networks
- utils/utils.py -> different methods used in fusion of results, preprocessing, supporting calculations

Citing information will be available once the article will be published
