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
import numpy as np
import open3d as o3d
import struct
import PIL
import argparse
import collections
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Render images from 3D model")
parser.add_argument("--model", type=str, required=True,
    help="Path to the 3D mesh model in a format supported by Open3D")
parser.add_argument("--colmap_model", type=str, required=False,
    help="Path to the colmap model (for camera definitions) - directory containing images.txt and cameras.txt")
parser.add_argument("--output_dir", type=str, required=True,
    help="Path to the output directory")
parser.add_argument("--only_black_frames", action="store_true",
    help="Check output dir. and render only if the output does not exist or the image is all black")
args = parser.parse_args()


def main(args):
    assert os.path.isfile(args.model)
    assert os.path.isdir(args.output_dir)

    valid_cam_info = (args.colmap_model is not None) and os.path.isdir(args.colmap_model)
    assert valid_cam_info, "No valid camera informations passed to the script - specify valid colmap_model argument"

    # Load the mesh
    print('Loading the mesh')
    mesh = o3d.io.read_triangle_model(args.model, True)

    # Load the images
    print('Loading the images and cameras')
    cam_list = []

    if args.colmap_model is not None:
        cameras, images = read_model(args.colmap_model)
        for i in images:
            qvec = images[i].qvec
            tvec = images[i].tvec
            cam_data = cameras[images[i].camera_id]
            cam_dict = parse_cam_model(cam_data)

            K = np.array([
                [cam_dict["fx"], 0.0, cam_dict["cx"]],
                [0.0, cam_dict["fx"], cam_dict["cy"]],
                [0.0, 0.0, 1.0]])

            R = qvec2rotmat(qvec)
            T = np.eye(4)
            T[0:3, 0:3] = R
            T[0:3, 3] = tvec
            w, h = cam_dict["width"], cam_dict["height"]

            basename = os.path.splitext(images[i].name)[0]

            cam_list.append({'basename':basename, 'K':K, 'T':T, 'w':w, 'h':h})
    elif args.vrephoto_dir is not None:
        file_list = os.listdir(args.vrephoto_dir)
        for file in file_list:
            if not(file.endswith(".cam")):
                continue

            cam_file_path = os.path.join(args.vrephoto_dir, file)
            res_file_path = os.path.join(args.vrephoto_dir, file[:-4] + ".res")

            w, h = parse_res_file(res_file_path)
            T, K = parse_cam_file(cam_file_path, w, h)

            basename = os.path.splitext(file)[0]

            cam_list.append({'basename':basename, 'K':K, 'T':T, 'w':w, 'h':h})

    # - all possible Open3D renderer shaders found in
    #   Open3D/cpp/open3d/visualization/gui/Materials/ directory
    for iter in range(len(mesh.materials)):
        mesh.materials[iter].shader = "defaultUnlit"
        
        # - the original colors make the textures too dark - set to white
        mesh.materials[iter].base_color = [1.0, 1.0, 1.0, 1.0]

    for cam in tqdm(cam_list):
        output_path = os.path.join(args.output_dir, "{}_rendered_color.png".format(cam["basename"].replace('/', '_')))

        if args.only_black_frames and os.path.exists(output_path):
            out_img = np.asarray(PIL.Image.open(output_path))
            if not(np.all(out_img <= 1)):
                # the output exists and is not all-black frame
                continue
            else:
                print("rerendering: {}".format(os.path.basename(output_path)))

        T = cam["T"]
        K = cam["K"]
        w, h = cam["w"], cam["h"]

        renderer = o3d.visualization.rendering.OffscreenRenderer(w, h)
        renderer.scene.add_model("Scene mesh", mesh)

        renderer.setup_camera(K, T, w, h)

        light_name_list = []

        # - setup lighting
        renderer.scene.scene.enable_sun_light(False)
        
        color = np.array(renderer.render_to_image())
        
        # # - OpenGL - left-handed CS
        # #   (http://www.songho.ca/opengl/gl_projectionmatrix.html)
        # # depth = ((far*near)/(far-near)) / (depth + (far/(near-far)))
        # # - OpenGL - right-handed CS
        # # depth = ((far*near)/(far-near)) / (depth + (far/(far-near)))
        # # - Filament - inverse of eq. 119 at
        # # (https://google.github.io/filament/Filament.md.html#imagingpipeline)
        # # depth = (near - (far*depth)) / (depth*(near - far))
        # # depth = (depth*(far+near)-2*near)/(depth*(far-near))
        # # - Open3D VisualizerRender.cpp lines 410-413
        # # depth = 2.0 * near * far / \
        # #     (far + near - (2.0 * depth - 1.0) * (far - near))
        # # - the same as above, just rewritten
        # depth = near * far / (far - depth * (far - near))

        depth = np.array(renderer.render_to_depth_image(True))
        depth[np.isinf(depth)] = 0.0

        img_rendering = PIL.Image.fromarray(color)
        img_rendering.save(output_path)
        np.savez_compressed(os.path.join(args.output_dir, "{}_depth.npz".format(cam["basename"].replace('/', '_'))), depth=depth.astype(np.float16))

        # - remove all lights from the scene
        for light_name in light_name_list:
            renderer.scene.scene.remove_light(light_name)

#### Code taken from Colmap:
# from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                # xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       # tuple(map(float, elems[1::3]))])
                # point3D_ids = np.array(tuple(map(int, elems[2::3])))
                # images[image_id] = Image(
                    # id=image_id, qvec=qvec, tvec=tvec,
                    # camera_id=camera_id, name=image_name,
                    # xys=xys, point3D_ids=point3D_ids)
                images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys={}, point3D_ids={})
    return images


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys={}, point3D_ids={})
    return images


def detect_model_format(path, ext):
    if os.path.isfile(os.path.join(path, "cameras"  + ext)) and \
       os.path.isfile(os.path.join(path, "images"   + ext)):
        print("Detected model format: '" + ext + "'")
        return True

    return False


def read_model(path, ext=""):
    # try to detect the extension automatically
    if ext == "":
        if detect_model_format(path, ".bin"):
            ext = ".bin"
        elif detect_model_format(path, ".txt"):
            ext = ".txt"
        else:
            print("Provide model format: '.bin' or '.txt'")
            return

    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_images_text(os.path.join(path, "images" + ext))
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
    return cameras, images


def parse_cam_model(cam_data):
    model = cam_data.model
    width = cam_data.width
    height = cam_data.height

    if model == "SIMPLE_PINHOLE" or model == "SIMPLE_RADIAL" or model == "RADIAL" or model == "SIMPLE_RADIAL_FISHEYE" or model == "RADIAL_FISHEYE":
        fx = cam_data.params[0]
        fy = fx
        cx = cam_data.params[1]
        cy = cam_data.params[2]
    elif model == "PINHOLE" or model == "OPENCV" or model == "OPENCV_FISHEYE" or model == "FULL_OPENCV" or model == "FOV" or model == "THIN_PRISM_FISHEYE":
        fx = cam_data.params[0]
        fy = cam_data.params[1]
        cx = cam_data.params[2]
        cy = cam_data.params[3]

    return {"width":width, "height":height, "fx":fx, "fy":fy, "cx":cx, "cy":cy}


def parse_cam_file(path, w, h):
    R = np.eye(3)
    t = np.zeros((3,1))
    T = np.eye(4)

    f = open(path, 'r')
    line1 = f.readline()
    line2 = f.readline()

    t[0], t[1], t[2], R[0, 0], R[0, 1], R[0, 2], R[1, 0], R[1, 1], R[1, 2], R[2, 0], R[2, 1], R[2, 2] = map(float, line1.split())
    T[0:3, 0:3] = R
    T[0:3, 3] = t.flatten()

    f_norm, _, _, aspect, cx_w, hcy_h = map(float, line2.split())

    fx = f_norm * np.float32(max(w, h))
    fy = aspect * fx
    cx = cx_w * w
    cy = h - (hcy_h * h)
    K = np.array([[fx, 0.0, cx],[0.0, fy, cy],[0.0, 0.0, 1.0]])

    f.close()

    return T, K


def parse_res_file(path):
    f = open(path, 'r')
    line = f.readline()
    words = line.split()
    w = int(words[0])
    h = int(words[1])
    f.close()

    return w, h


if __name__ == "__main__":
    main(args)
