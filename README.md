# Benchmark for visual localization on 3D mesh models
Repository for the localization benchmark at [v-pnk.github.io/cadloc](https://v-pnk.github.io/cadloc/). To upload the results to the benchmark, run the evaluation script on your pose estimates, create the metadata file and create a pull request to add your results into the tables.
## Installation
The evaluation tool needs few Python packages:
```
conda create --name cadloc python=3.9
conda activate cadloc
pip3 install open3d==0.16 pycolmap==0.3 numpy argparse tqdm
```
## Preparation of data for the benchmark
### Evaluation script usage
The [evaluation script]() expects the estimated poses to be stored in the .txt format used by [visuallocalization.net](https://www.visuallocalization.net/), i.e., each line contains an image name and estimated pose described by a quaternion `q` and translation vector `t`:
```
<name> <q[W]> <q[X]> <q[Y]> <q[Z]> <t[X]> <t[Y]> <t[Z]>
```
### Metadata generation
Metadata file contains the basic information corresponding to the values in the evaluation file. It has the same name as the evaluation file, but with .yaml extension. You can generate such file using the [metadata generation script]() with parameters: 
- `--method_name` - Name of the used method (should be unique within the benchmark).
- `--pub_link` - Web link to the publication describing the used method.
- `--code_link` - Web link to the code of the used method.
- `--output_path` - Path to the output metadata .yaml file.
## Uploading new benchmark results
Add your evaluation and metadata files (e.g., my_awesome_method.txt and my_awesome_method.yaml) into the results directory and create pull request.
## License
This repository is licensed under the 3-Clause BSD License. See the [LICENSE](https://github.com/v-pnk/cadloc/blob/main/LICENSE) file for full text.