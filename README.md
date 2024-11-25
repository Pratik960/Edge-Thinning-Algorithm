
Edge thinning algorithm

This repository contains an implementation of the edge-thinning algorithm proposed in the research paper:
Ren, Lijuan, Wang, Xionghui, Wang, Nina, Zhang, Guangpeng, Yongchang, Li, & Yang, Zhijian. (2022). An edge thinning algorithm based on newly defined single‐pixel edge patterns. IET Image Processing, 17, 1161–1169. DOI: 10.1049/ipr2.12703.

The algorithm processes grayscale edge images to produce uniform, smooth, and connected one-pixel-wide edges while maintaining edge connectivity and precise positioning.



## Features

- Generates single-pixel-wide edges from grayscale edge images.
- Retains edge connectivity, smoothness, and uniformity.
- Supports both grayscale and binary images (via a data conversion method).
- Implements 24 single-pixel connection patterns for accurate edge thinning.
