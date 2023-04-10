#!/usr/bin/env python3

# Copyright (c) 2023, Vojtech Panek and Zuzana Kukelova and Torsten Sattler
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import copy
import math
import argparse
from tqdm import tqdm
import numpy as np
import open3d as o3d
import pycolmap


parser = argparse.ArgumentParser(description="Evaluation tool", 
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--est_file", type=str, required=True,
    help="Path to the file with with pose estimates of query images")
parser.add_argument("--colmap_gt", type=str, required=True,
    help="Path to the COLMAP model with GT query camera poses")
parser.add_argument("--reference_mesh", type=str, required=True,
    help="Path to the reference model in coordinate frame of the GT poses")
parser.add_argument("--internet_mesh", type=str, required=True,
    help="Path to the Internet mesh used for alignment")
parser.add_argument("--output_dcre", type=str, required=False,
    help="Path to an output file containing the computed DCREs for each image")
parser.add_argument("--print_html", action="store_true",
    help="Print the results in HTML table format")
parser.add_argument("--output_html", type=str, required=False,
    help="Path to an output file containing the results in HTML table format")


def main(args):
    assert os.path.isfile(args.est_file)
    assert os.path.isdir(args.colmap_gt)
    assert os.path.isfile(args.reference_mesh)
    assert os.path.isfile(args.internet_mesh)

    print("- loading the estimated poses")
    est_poses_dict = read_est_file(args.est_file)

    print("- loading the GT COLMAP model")
    colmap_gt_dict = read_colmap_model(args.colmap_gt)

    print("- loading the reference mesh")
    reference_mesh = o3d.io.read_triangle_model(args.reference_mesh, False)

    print("- loading the internet mesh")
    internet_mesh = o3d.io.read_triangle_model(args.internet_mesh, False)

    print("- running the evaluation")
    results_dict = evaluate(reference_mesh, internet_mesh, colmap_gt_dict, est_poses_dict)

    if args.output_dcre is not None:
        write_dcre(args.output_dcre, results_dict)
    
    dcre_ga_mean_list = extract_dict_list(results_dict, "GA_mean")
    recalls_ga_mean = compute_recall(dcre_ga_mean_list, [0.1, 0.2, 0.3], len(colmap_gt_dict))
    dcre_ga_max_list = extract_dict_list(results_dict, "GA_max")
    recalls_ga_max = compute_recall(dcre_ga_max_list, [0.1, 0.2, 0.3], len(colmap_gt_dict))
    dcre_lr_mean_list = extract_dict_list(results_dict, "LR_mean")
    recalls_lr_mean = compute_recall(dcre_lr_mean_list, [0.1, 0.2, 0.3], len(colmap_gt_dict))
    dcre_lr_max_list = extract_dict_list(results_dict, "LR_max")
    recalls_lr_max = compute_recall(dcre_lr_max_list, [0.1, 0.2, 0.3], len(colmap_gt_dict))

    print_results(recalls_ga_mean, recalls_ga_max, recalls_lr_mean, recalls_lr_max, args.print_html)
    save_html(recalls_ga_mean, recalls_ga_max, recalls_lr_mean, recalls_lr_max, args.output_html)

def evaluate(reference_mesh, internet_mesh, colmap_gt_dict, est_poses_dict):
    results_dict = {}

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
    
    # - reusing the renderer (not definining it again for each image) is faster,
    #   but fails when garbage collector tries to destroy the object
    # osr_reference = o3d.visualization.rendering.OffscreenRenderer(800, 800)
    # osr_reference.scene.add_model("mesh", reference_mesh)
    # osr_internet = o3d.visualization.rendering.OffscreenRenderer(800, 800)
    # osr_internet.scene.add_model("mesh", internet_mesh)

    for img_name in tqdm(list(est_poses_dict.keys())):
        est_img = est_poses_dict[img_name]
        gt_img = colmap_gt_dict[img_name]

        # Render the depth maps
        # - from query model for ground truth pose
        query_gt_depth = render_depth(
            reference_mesh, gt_img["T"], gt_img["K"], gt_img["w"], gt_img["h"])
        # - from Internet model for estimated pose
        internet_est_depth = render_depth(
            internet_mesh, est_img["T"], gt_img["K"], gt_img["w"], gt_img["h"])
        # - from Internet model for grount truth pose
        internet_gt_depth = render_depth(
            internet_mesh, gt_img["T"], gt_img["K"], gt_img["w"], gt_img["h"])
        # query_gt_depth = render_depth(
        #     osr_reference, gt_img["T"], gt_img["K"], gt_img["w"], gt_img["h"])
        # # - from Internet model for estimated pose
        # internet_est_depth = render_depth(
        #     osr_internet, est_img["T"], gt_img["K"], gt_img["w"], gt_img["h"])
        # # - from Internet model for grount truth pose
        # internet_gt_depth = render_depth(
        #     osr_internet, gt_img["T"], gt_img["K"], gt_img["w"], gt_img["h"])

        # Compute DCRE on the globally aligned poses (GA)
        if valid_depth(internet_est_depth):
            dcre_ga_max = compute_error_dcre(internet_est_depth, est_img["T"], gt_img["T"],
                                            gt_img["K"][0, 0], gt_img["w"], 
                                            use_max=True, depth_mult=1.0)
            dcre_ga_mean = compute_error_dcre(internet_est_depth, est_img["T"], gt_img["T"],
                                            gt_img["K"][0, 0], gt_img["w"], 
                                            use_max=False, depth_mult=1.0)
            diagonal = np.sqrt(gt_img["w"]**2 + gt_img["h"]**2)
            dcre_ga_max = dcre_ga_max / diagonal
            dcre_ga_mean = dcre_ga_mean / diagonal
        else:
            dcre_ga_max = -1
            dcre_ga_mean = -1

        # Compute DCRE on the locally refined poses (LR)
        if valid_depth(query_gt_depth) and valid_depth(internet_gt_depth):
            # compute the ICP just on a sumbsample of pixels to speed it up
            T_icp = align_depths(query_gt_depth, internet_gt_depth, gt_img)
        else:
            T_icp = np.eye(4)

        if valid_depth(internet_est_depth):
            dcre_lr_max = compute_error_dcre(internet_est_depth, est_img["T"], T_icp @ gt_img["T"],
                                             gt_img["K"][0, 0], gt_img["w"],
                                             use_max=True, depth_mult=1.0)
            dcre_lr_mean = compute_error_dcre(internet_est_depth, est_img["T"], T_icp @ gt_img["T"],
                                              gt_img["K"][0, 0], gt_img["w"],
                                              use_max=False, depth_mult=1.0)
            diagonal = np.sqrt(gt_img["w"]**2 + gt_img["h"]**2)
            dcre_lr_max = dcre_lr_max / diagonal
            dcre_lr_mean = dcre_lr_mean / diagonal
        else:
            dcre_lr_max = -1
            dcre_lr_mean = -1
        
        results_dict[img_name] = {"GA_mean" : dcre_ga_mean, 
                                  "GA_max" : dcre_ga_max, 
                                  "LR_mean" : dcre_lr_mean, 
                                  "LR_max" : dcre_lr_max}

    return results_dict


# Render depth map
def render_depth(mesh, T, K, w, h):
    renderer = o3d.visualization.rendering.OffscreenRenderer(w, h)
    renderer.scene.add_model("mesh", mesh)
    renderer.setup_camera(K, T, w, h)

    depth = np.array(renderer.render_to_depth_image(True))
    depth[np.isinf(depth)] = 0.0

    return depth

# def render_depth(osr, T, K, w, h):
#     osr.setup_camera(K, T, w, h)

#     depth = np.array(osr.render_to_depth_image(True))
#     depth[np.isinf(depth)] = 0.0

#     return depth


# Read the file with pose estimates
# - <image name> <quaternion> <translation vector>
def read_est_file(est_file_path):
    f_est = open(est_file_path, 'r')
    data_dict = {}

    for line in f_est:
        img_dict = {}
        data = line.split()
        img_name = data[0]

        img_name = os.path.splitext(img_name)[0]
        est_qvec = np.array(list(map(float, data[1:5])))
        est_R = quat2R(est_qvec)
        est_tvec = np.array(list(map(float, data[5:8])))
        est_T = np.eye(4)
        est_T[0:3, 0:3] = est_R
        est_T[0:3, 3] = est_tvec
        img_dict["T"] = est_T
        data_dict[img_name] = img_dict

    f_est.close()
    return data_dict


# Read the intrinsic and extrinsic parameters from a COLMAP model
# - https://colmap.github.io/format.html#sparse-reconstruction
def read_colmap_model(model_path):
    colmap_model = pycolmap.Reconstruction(model_path)
    data_dict = {}

    for image_i in colmap_model.images:
        img_dict = {}
        image = colmap_model.images[image_i]
        camera = colmap_model.cameras[image.camera_id]

        img_dict["K"] = camera.calibration_matrix()
        img_dict["w"] = camera.width
        img_dict["h"] = camera.height

        T = np.eye(4)
        T[0:3, 0:3] = quat2R(image.qvec)
        T[0:3, 3] = image.tvec
        img_dict["T"] = T

        img_name = os.path.splitext(image.name)[0]
        data_dict[img_name] = img_dict

    return data_dict


def write_dcre(dcre_path, results_dict):
    f_dcre = open(dcre_path, 'w')

    f_dcre.write("# Localization evaluation\n")
    f_dcre.write("# - each line contains:\n")
    f_dcre.write(
        "#   mean DCRE divided by query diagonal (EST --> internet depth --> GT)\n")
    f_dcre.write(
        "#   max DCRE divided by query diagonal (EST --> internet depth --> GT)\n")
    f_dcre.write(
        "#   mean DCRE divided by query diagonal (EST --> internet depth --> ICP GT)\n")
    f_dcre.write(
        "#   max DCRE divided by query diagonal (EST --> internet depth --> ICP GT)\n")

    for img_i in results_dict:

        f_dcre.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(img_i,
            results_dict[img_i]["GA_mean"], results_dict[img_i]["GA_max"], 
            results_dict[img_i]["LR_mean"], results_dict[img_i]["LR_max"]))

    f_dcre.close()


# Function for computation of Dense Correspondence Reprojection Error
# - code originally from: https://github.com/tsattler/visloc_pseudo_gt_limitations/blob/main/evaluation_util.py
# - depth_mult = 1000 if the depth is stored in mm, 1 if stored in m
def compute_error_dcre(depth, pgt_pose, est_pose, rgb_focal_length, 
                       rgb_image_width, use_max=False, depth_mult=1.0):
    '''
    Compute the dense reprojection error.
    Expects poses to map camera coordinate to world coordinates.
    Needs access to image depth.
    Calculates the max. DCRE per images, or the mean if use_max=False.
    '''

    # pgt_pose = torch.from_numpy(pgt_pose).cuda()
    # est_pose = torch.from_numpy(est_pose).cuda()

    depth = depth.astype(np.float64)
    depth /= depth_mult  # to meters

    d_h = depth.shape[0]
    d_w = depth.shape[1]

    rgb_to_d_scale = d_w / rgb_image_width
    d_focal_length = rgb_focal_length * rgb_to_d_scale

    # reproject depth map to 3D eye coordinates
    prec_eye_coords = np.zeros((4, d_h, d_w))
    # set x and y coordinates
    prec_eye_coords[0] = np.dstack([np.arange(0, d_w)] * d_h)[0].T
    prec_eye_coords[1] = np.dstack([np.arange(0, d_h)] * d_w)[0]
    prec_eye_coords = prec_eye_coords.reshape(4, -1)

    eye_coords = prec_eye_coords.copy()
    depth = depth.reshape(-1)

    # filter pixels with invalid depth
    depth_mask = (depth > 0.01)
    eye_coords = eye_coords[:, depth_mask]
    depth = depth[depth_mask]
    
    # - Return -1 if there is no valid depth
    if depth.size == 0:
        return -1

    # eye_coords = torch.from_numpy(eye_coords).cuda()
    # depth = torch.from_numpy(depth).cuda()

    # save original pixel positions for later
    # pixel_coords = eye_coords[0:2].clone()
    pixel_coords = eye_coords[0:2].copy()

    # substract depth principal point (assume image center)
    eye_coords[0] -= d_w / 2
    eye_coords[1] -= d_h / 2
    # reproject
    eye_coords[0:2] *= depth / d_focal_length
    eye_coords[2] = depth
    eye_coords[3] = 1

    # transform to world and back to cam
    # scene_coords = torch.matmul(pgt_pose, eye_coords)
    # eye_coords = torch.matmul(torch.inverse(est_pose), scene_coords)
    scene_coords = pgt_pose @ eye_coords
    eye_coords = np.linalg.inv(est_pose) @ scene_coords

    # project
    depth = eye_coords[2]
    eye_coords = eye_coords[0:2]

    eye_coords *= (d_focal_length / depth)

    # add RGB principal point (assume image center)
    eye_coords[0] += d_w / 2
    eye_coords[1] += d_h / 2

    reprojection_errors = np.linalg.norm(eye_coords - pixel_coords, ord=2.0, axis=0)

    if use_max:
        return float(reprojection_errors.max()) / rgb_to_d_scale
    else:
        return float(np.mean(reprojection_errors)) / rgb_to_d_scale

# Align depth maps
def align_depths(depth_source, depth_target, cam_intr, sample_ratio=0.1):
    o3d_cam_intr = o3d.camera.PinholeCameraIntrinsic(
        width=cam_intr["w"], height=cam_intr["h"],
        fx=cam_intr["K"][0, 0], fy=cam_intr["K"][1, 1],
        cx=cam_intr["K"][0, 2], cy=cam_intr["K"][1, 2])
    
    pc_source = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(depth_source.astype(np.float32)),
        intrinsic=o3d_cam_intr, depth_scale=1.0)
    pc_target = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(depth_target.astype(np.float32)),
        intrinsic=o3d_cam_intr, depth_scale=1.0)
    
    # Aligning point clouds along camera z axis often helps ICP to converge
    z_mean_source = np.mean(np.asarray(pc_source.points)[:,2])
    z_mean_target = np.mean(np.asarray(pc_target.points)[:,2])
    z_shift = z_mean_target - z_mean_source
    T_glob = np.eye(4)
    T_glob[2,3] = z_shift

    pc_sub_source = copy.deepcopy(pc_source).random_down_sample(sample_ratio)
    pc_sub_target = copy.deepcopy(pc_target).random_down_sample(sample_ratio)

    mean_pc_dist = np.mean(np.asarray(
        pc_sub_target.compute_point_cloud_distance(pc_sub_source)))
    max_dist = mean_pc_dist

    reg_p2p = o3d.pipelines.registration.registration_icp(
        pc_sub_source, pc_sub_target, max_dist, T_glob,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    return reg_p2p.transformation


# Quaternion (WXYZ) to rotation matrix
def quat2R(q):
    R = np.array([
        [1 - 2*(q[2]*q[2] + q[3]*q[3]), 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[1]*q[3] + q[0]*q[2])],
        [2*(q[1]*q[2] + q[0]*q[3]), 1 - 2 * (q[1]*q[1] + q[3]*q[3]), 2*(q[2]*q[3] - q[0]*q[1])],
        [2*(q[1]*q[3] - q[0]*q[2]), 2*(q[2]*q[3] + q[0]*q[1]), 1 - 2*(q[1]*q[1] + q[2]*q[2])]
    ])

    R = np.squeeze(R)

    return R


# Rotation matrix to quaternion (WXYZ)
def R2quat(R):
    tr = np.trace(R)

    if (tr > 0):
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif ((R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2])):
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif (R[1, 1] > R[2, 2]):
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([[w], [x], [y], [z]])


# Check if the depth_map contains the minimal necessary sample
def valid_depth(depth_map, sample_ratio=0.1):
    if sample_ratio > 0:
        return np.sum(depth_map > 0.01) >= (1 / sample_ratio)
    else:
        return np.sum(depth_map > 0.01) > 0


# Extract values of specific key from list of dictionaries
def extract_dict_list(dict_list, key_name):
    out_list = []
    for d in dict_list:
        out_list.append(dict_list[d][key_name])

    return np.array(out_list)


# Compute recall at given error threshold
def compute_recall(errors, thr_list, gt_num):
    recall_list = []
    for thr in thr_list:
        recall_list.append(100 * np.sum(errors <= thr) / gt_num)
    return recall_list


def print_results(ga_mean, ga_max, lr_mean, lr_max, html):
    print(args.est_file + "\n")

    if html:
        print("<tr>")
        print("    <td><a href=\"example.org\">paper link</a></td>")
        print("    <td><a href=\"example.org\">code link</a></td>")
        print("    <td class=\"recall_3column\">{:.1f} / {:.1f} / {:.1f}</td>".format(*ga_mean))
        print("    <td class=\"recall_3column\">{:.1f} / {:.1f} / {:.1f}</td>".format(*ga_max))
        print("    <td class=\"recall_3column\">{:.1f} / {:.1f} / {:.1f}</td>".format(*lr_mean))
        print("    <td class=\"recall_3column\">{:.1f} / {:.1f} / {:.1f}</td>".format(*lr_max))
        print("</tr>")
    else:
        print("printing recall at 10%, 20% and 30% of image diagonal length thresholds")
        print("mean GA: {:.1f} {:.1f} {:.1f}".format(*ga_mean))
        print("mean GA: {:.1f} {:.1f} {:.1f}".format(*ga_max))
        print("mean GA: {:.1f} {:.1f} {:.1f}".format(*lr_mean))
        print("mean GA: {:.1f} {:.1f} {:.1f}".format(*lr_max))


def save_html(ga_mean, ga_max, lr_mean, lr_max, html_path):
    with open(html_path, 'wt') as f:
        f.write("printing recall at 10%, 20% and 30% of image diagonal length thresholds\n")
        f.write("<tr>\n")
        f.write("    <td>method name</td>\n")
        f.write("    <td><a href=\"example.org\">paper link</a></td>\n")
        f.write("    <td><a href=\"example.org\">code link</a></td>\n")
        f.write("    <td class=\"recall_3column\">{:.1f} / {:.1f} / {:.1f}</td>\n".format(*ga_mean))
        f.write("    <td class=\"recall_3column\">{:.1f} / {:.1f} / {:.1f}</td>\n".format(*ga_max))
        f.write("    <td class=\"recall_3column\">{:.1f} / {:.1f} / {:.1f}</td>\n".format(*lr_mean))
        f.write("    <td class=\"recall_3column\">{:.1f} / {:.1f} / {:.1f}</td>\n".format(*lr_max))
        f.write("</tr>\n")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
