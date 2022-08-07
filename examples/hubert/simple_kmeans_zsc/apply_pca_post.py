#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import shutil

import sys
import argparse
import os
import os.path as osp
import math
import numpy as np
import tqdm
import torch
from shutil import copyfile

from npy_append_array import NpyAppendArray

def get_parser():
    parser = argparse.ArgumentParser(
        description="transforms features via a given pca and stored them in target dir"
    )
    parser.add_argument('source', help='directory with features')
    parser.add_argument('target_dir', help='directory out with features')
    parser.add_argument('--pca-path', type=str, help='pca location. will append _A.npy and _b.npy', required=True)

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    #pca_A = torch.from_numpy(np.load(args.pca_path + "_A.npy")).cuda()
    #pca_b = torch.from_numpy(np.load(args.pca_path + "_b.npy")).cuda()

    pca_A = np.load(args.pca_path + "_A.npy")
    pca_b = np.load(args.pca_path + "_b.npy")

    feat_file = args.source
    basename = os.path.basename(feat_file)
    feat_path_out = os.path.join(args.target_dir, basename)

    #with torch.no_grad():


    feats = np.load(feat_file)
    #feats = torch.from_numpy(feats).cuda()

    #x = torch.matmul(feats, pca_A) + pca_b
    x = np.matmul(feats, pca_A) + pca_b

    #np.save(feat_path_out, x.cpu().numpy())
    np.save(feat_path_out, x)


if __name__ == "__main__":
    main()
