import sys
import numpy as np
import os
import logging
import tqdm
from pathlib import Path
from multiprocessing.dummy import Pool as ThreadPool


def load_feature(feat_dir, split, out_path):

    name_id=0

    feat_path = f"{feat_dir}/{split}_0_1.npy"
    leng_path = f"{feat_dir}/{split}_0_1.len"
    with open(leng_path, "r") as f:
        id_list = [line.rstrip().split()[0].split("/")[-1].replace(".flac","") for line in f]

    with open(leng_path, "r") as f:
        lengs = [int(line.rstrip().split()[1]) for line in f]

        offsets = [0] + np.cumsum(lengs[:-1]).tolist()
        feats = np.load(feat_path, mmap_mode="r")
        for i in range(len(lengs)):
            feat = feats[offsets[i]: offsets[i] + lengs[i]]
            np.save(out_path+"/"+id_list[i].replace(".wav",".npy"), feat)

feats_dir=sys.argv[1]
split=sys.argv[2]
out_path=sys.argv[3]


feat = load_feature(feats_dir, split, out_path)

