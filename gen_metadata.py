#!/usr/bin/env python3

# Copyright (c) 2022, Vojtech Panek and Zuzana Kukelova and Torsten Sattler
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
import yaml
import argparse


parser = argparse.ArgumentParser(description="Metadata file generator",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--method_name", type=str, required=True,
                    help="Name of the used method (should be unique within the benchmark). Place within quatation marks if it contains spaces.")
parser.add_argument("--pub_link", type=str,
                    help="Web link to the publication describing the used method.")
parser.add_argument("--code_link", type=str,
                    help="Web link to the code of the used method.")
# parser.add_argument("--description", type=str,
#                     help="Longer string with the method description. Place the within quatation marks if it contains spaces.")
parser.add_argument("--output_path", type=str,
                    help="Path to the output metadata .yaml file.")


def main(args):
    # define a method name length limit so it fits into the table
    method_name_len_limit = 30 # TODO: figure out the real length which fits into the table

    args.method_name = args.method_name.rstrip()

    if len(args.method_name) > method_name_len_limit:
        method_name_crop = args.method_name[:method_name_len_limit]

        print("\033[93m" + "WARN: Method name longer than the limit ({} characters), it will be cropped to:".format(method_name_len_limit) + "\033[0m")
        print("\033[93m      " + method_name_crop + "\033[0m")
    
    metadata_dict = {}
    metadata_dict["method_name"] = args.method_name
    if args.pub_link is not None:
        metadata_dict["pub_link"] = args.pub_link
    if args.code_link is not None:
        metadata_dict["code_link"] = args.code_link

    with open(args.output_path, 'w') as f:
        f.write(yaml.dump(metadata_dict))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)