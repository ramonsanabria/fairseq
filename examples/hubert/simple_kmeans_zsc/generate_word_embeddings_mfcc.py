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
    min_time=(float(idx_segment)-1)*30
    max_time=float(idx_segment)*30
    words_in_segments = []
    real_start_and_end = []

    for word_alignment in alignments_file:
        start, end, word = word_alignment

        if((min_time < float(start)) and (max_time > float(end))):
            words_in_segments.append(word_alignment)
            real_start_and_end.append((start,end))

    return min_time, words_in_segments, real_start_and_end


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

pooling_type=sys.argv[4]
feature_pooled_dir=sys.argv[8]
feature_path=feature_pooled_dir+"/"+setname+"_features_"+pooling_type+".npy"
word_path=feature_pooled_dir+"/"+setname+"_features_"+pooling_type+".wrd"

if("sub" in pooling_type):
    subsample_value=int(pooling_type.replace("sub",""))
    pooling_type="sub"
tsv_dir=sys.argv[5]
nshard=int(sys.argv[6])
feat_dir=sys.argv[7]

pathlib.Path(feature_pooled_dir).mkdir(parents=True, exist_ok=True)

if os.path.exists(feature_path):
    os.remove(feature_path)
else:
    print("features do not exist")

if os.path.exists(word_path):
    os.remove(word_path)
else:
    print("features do not exist")

with NpyAppendArray(feature_path) as npaa, open(word_path,"w") as wordfile:

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

            idx_segment = int(key.split("_")[-1])
            filename = "_".join(key.split("_")[:-1])
            alignments_file = word_alignments[filename]

            min_time, word_segments, real_start_end = get_alignmens_segment(alignments_file, idx_segment)

            mat=feat_shard[idx]

            for word_index in range(len(word_segments)):

                start, end, word = word_segments[word_index]

                if(word != ""):

                    min_sample = int(((start-min_time)*1000)/10)
                    max_sample = int(((end-min_time)*1000)/10)

                    if(pooling_type == "avg"):
                        awe = np.average(mat[min_sample:max_sample,:], axis=0)
                    elif(pooling_type == "max"):
                        awe = np.max(mat[min_sample:max_sample,:], axis=0)
                    elif(pooling_type == "sum"):
                        awe = np.sum(mat[min_sample:max_sample,:], axis=0)
                    elif(pooling_type == "sub"):
                        awe = subsample(mat[min_sample:max_sample,:], subsample_value)

                    real_start, real_end = real_start_end[word_index]
                    key_id=key.split("_")[0]
                    real_start, real_end = real_start_end[word_index]
                    spk_id = key_id[:3]
                    utt_id = key_id[3:]
                    npz_id = word + "_" + spk_id +"_"+utt_id+"_%06d-%06d" % (int(round(real_start*100)),int(round(real_end*100)) + 1)

                    awe = np.expand_dims(awe, axis=0)
                    wordfile.write(npz_id+"\n")
                    npaa.append(awe)

