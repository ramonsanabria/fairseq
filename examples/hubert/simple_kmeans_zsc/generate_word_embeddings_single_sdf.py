import sys
import textgrid
import numpy as np
import os
import shutil
import pathlib
from os import path as osp
from npy_append_array import NpyAppendArray
import tqdm



aligment_folder="/disk/scratch1/ramons/data/zerospeech/"

def subsample(feats, n):
    k = len(feats) / n

    result = []
    for i in range(n):
        result.extend(feats[int(k * i)])

    return np.array(result)

def get_alignmens_segment(alignments_file, idx_segment):

    words_in_segments = []

    for word_alignment in alignments_file:
        start, end, word = word_alignment

        words_in_segments.append(word_alignment)

    return words_in_segments


def load_word_alignments(path_alignments):
    all_words={}

    with open(path_alignments) as inputfile:
        for word_line in inputfile.readlines():
            file_id = word_line.strip().split()[0]
            start = float(word_line.strip().split()[1])
            end = float(word_line.strip().split()[2])
            word = str(word_line.strip().split()[3])
            if(file_id not in all_words):
                all_words[file_id] = [(start, end, word)]
            else:
                list_segments = all_words[file_id] 
                list_segments.append((start, end, word))
                all_words[file_id] = list_segments
    return all_words

def load_feature_shard(feat_dir, split, nshard, rank):
    feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
    leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"
    with open(leng_path, "r") as f:
        ids = [line.rstrip().split()[0].split("/")[-1].replace(".wav","") for line in f]

    with open(leng_path, "r") as f:
        lengs = [int(line.rstrip().split()[1]) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

    nsample = int(np.ceil(len(lengs)))
    feat = np.load(feat_path, mmap_mode="r")
    feat_list = [feat[offsets[i]: offsets[i] + lengs[i]] for i in range(len(lengs))]
    return feat_list, ids, lengs



setname=sys.argv[1]
model=sys.argv[2]
layer=sys.argv[3]
out_dict={}

feature_pooled_dir=sys.argv[7]
feature_path=feature_pooled_dir+"/"+setname+"_features_sdf.npz"

tsv_dir=sys.argv[4]
nshard=int(sys.argv[5])
feat_dir=sys.argv[6]

pathlib.Path(feature_pooled_dir).mkdir(parents=True, exist_ok=True)

if os.path.exists(feature_path):
    os.remove(feature_path)
else:
    print("features do not exist")


path_aligments=osp.join(aligment_folder,setname+".wrd")
if(not osp.exists(path_aligments)):
    print("PATH NOT FOUND: "+path_aligments)
    sys.exit()

word_alignments = load_word_alignments(path_aligments)

for r in range(nshard):

    #get feats and ids
    feat_shard, shard_ids, lens = load_feature_shard(feat_dir, setname, nshard, r)
    len_shard=len(feat_shard)
    print("Startng shard: "+str(r))

    for idx, key in tqdm.tqdm(enumerate(shard_ids), total=len_shard):


        alignments_file = word_alignments[key]

        mat=feat_shard[idx]

        for word_index in range(len(alignments_file)):

            start, end, word = alignments_file[word_index]


            if(word == "" or word == "SIL"):
                continue
            else:

                min_sample = int(((start)*1000)/20)
                max_sample = int(((end)*1000)/20)

                dataset_id=key.split("_")[0]+"-"+key.split("_")[1]
                spk_id=key.split("_")[2]
                utt_id=key.split("_")[3]



                npz_id = word + "_" + spk_id +"_"+dataset_id+"-"+utt_id+"_%06d-%06d" % (int(round(start*100)),int(round(end*100)) + 1)

                awe = mat[min_sample:max_sample,:]

                out_dict[npz_id] = awe

np.savez_compressed(feature_path, **out_dict)
