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
    parser.add_argument('target', help='directory out with features')
    parser.add_argument('--pca-path', type=str, help='pca location. will append _A.npy and _b.npy', required=True)
    parser.add_argument('--nshards', type=str, help='number shards', required=True)
    parser.add_argument('--setname', type=str, help='name of the set', required=True)

    return parser

def load_feature_shard(feat_dir, split, nshard, rank, feat_dir_out):
    feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
    leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"

    feat_path_out = f"{feat_dir_out}/{split}_{rank}_{nshard}.npy"
    leng_path_out = f"{feat_dir_out}/{split}_{rank}_{nshard}.len"

    with open(leng_path, "r") as f:
        ids = [line.rstrip().split()[0].split("/")[-1].replace(".flac","") for line in f]
    with open(leng_path, "r") as f:
        lengs = [int(line.rstrip().split()[1]) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

    nsample = int(np.ceil(len(lengs)))
    feat = np.load(feat_path, mmap_mode="r")
    return feat, ids, lengs, leng_path, feat_path_out, leng_path_out


def main():
    parser = get_parser()
    args = parser.parse_args()

    pca_A = torch.from_numpy(np.load(args.pca_path + "_A.npy")).cuda()
    pca_b = torch.from_numpy(np.load(args.pca_path + "_b.npy")).cuda()

    feat_dir = args.source
    feat_dir_out = args.target
    nshard = int(args.nshards)
    setname = args.setname

    with torch.no_grad():
        for r in range(nshard):
            print("doing "+str(r)+" shard...")
            feat_shard, shard_ids, lens, leng_path_in, feat_path_out, leng_path_out = load_feature_shard(feat_dir, setname, nshard, r, feat_dir_out)
            feat_c = np.array(feat_shard, copy=True)
            del feat_shard

            x = torch.from_numpy(feat_c).cuda()
            x = torch.matmul(x, pca_A) + pca_b
            del feat_c

            pca_dim = str(x.shape[1])
            np.save(feat_path_out, x.cpu().numpy())
            if(leng_path_in != leng_path_out):
                shutil.copyfile(leng_path_in, leng_path_out)

if __name__ == "__main__":
    main()
