# Benchmark for visual localization on 3D mesh models
Repository for the localization benchmark at [v-pnk.github.io/cadloc](https://v-pnk.github.io/cadloc/). To upload the results to the benchmark, run the evaluation script on your pose estimates to get the DCRE values and create a pull request to add your results into the tables.
## Installation
The evaluation tool needs few Python packages:
```
conda create --name cadloc python=3.9
conda activate cadloc
pip3 install open3d==0.17 pycolmap==0.3 numpy argparse tqdm
```
## Preparation of data for the benchmark
### Evaluation script usage
The [evaluation script]() expects the estimated poses to be stored in the .txt format used by [visuallocalization.net](https://www.visuallocalization.net/), i.e., each line contains an image name and estimated pose described by a quaternion `q` and translation vector `t`:
```
<name> <q[W]> <q[X]> <q[Y]> <q[Z]> <t[X]> <t[Y]> <t[Z]>
```
## License
This repository is licensed under the 3-Clause BSD License. See the [LICENSE](https://github.com/v-pnk/cadloc/blob/main/LICENSE) file for full text.
