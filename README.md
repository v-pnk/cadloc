# Benchmark for visual localization on 3D mesh models
Repository for the localization benchmark at [v-pnk.github.io/cadloc](https://v-pnk.github.io/cadloc/). To upload the results to the benchmark, run [the evaluation script](https://github.com/v-pnk/cadloc/blob/main/evaluate_dcre.py) on your pose estimates and create a pull request to add your results into the tables.

## Citation
If you use the benchmark or the code from this repository, please cite the following paper:
```
@article{Panek2023cadloc,
title={{Visual Localization using Imperfect 3D Models from the Internet}},
author={Vojtech Panek and Zuzana Kukelova and Torsten Sattler},
booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
year={2023}}
```

>>>>>>> c5d6fd0 (Usable web version)
## Installation
The evaluation tool needs few Python packages:
```
conda create --name cadloc python=3.9
conda activate cadloc
pip3 install open3d==0.17 pycolmap==0.3 numpy argparse tqdm
```
## Preparation of data for the benchmark
[The evaluation script](https://github.com/v-pnk/cadloc/blob/main/evaluate_dcre.py) expects the estimated poses to be stored in the .txt format used by [visuallocalization.net](https://www.visuallocalization.net/), i.e., each line contains an image name and estimated pose described by a quaternion `q` and translation vector `t`:
```
<name> <q[W]> <q[X]> <q[Y]> <q[Z]> <t[X]> <t[Y]> <t[Z]>
```

## Uploading new benchmark results
Add your evaluation and metadata files (e.g., my_awesome_method.txt and my_awesome_method.yaml) into the results directory and create pull request.

>>>>>>> c5d6fd0 (Usable web version)
## License
This repository is licensed under the 3-Clause BSD License. See the [LICENSE](https://github.com/v-pnk/cadloc/blob/main/LICENSE) file for full text.
