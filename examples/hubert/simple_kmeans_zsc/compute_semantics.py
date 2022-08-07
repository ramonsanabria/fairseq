import numpy as np
import os
import argparse
import sys
from scipy import stats
from scipy.spatial import distance

def get_parser():
    parser = argparse.ArgumentParser(
        description="compute kmeans codebook from kaldi-computed feats"
    )
    # fmt: off
    parser.add_argument('--indir', help='directory with the input features', required=True)
    parser.add_argument('--outdir', help='directory where to write the output features', required=True)
    parser.add_argument('--poolingmethod', help='pooling method used', required=True)

    return parser

parser = get_parser()
args = parser.parse_args()

path_save=args.indir
path_npy=args.outdir
pooling_method=args.poolingmethod


dict_awes={}
dict_count={}

print("loading AWE")
for numpyfile in os.listdir(path_npy):
    if("_features_"+pooling_method+".npy" in numpyfile):

        path_npy_file = os.path.join(path_npy, numpyfile)
        awes = np.load(path_npy_file)
        path_wrd = os.path.join(path_npy_file.replace(".npy",".wrd"))

        with open(path_wrd) as inputfile:
            for idx, wrdl in enumerate(inputfile.readlines()):
                wrd = wrdl.strip().lower()
                if(wrd not in dict_awes):
                    dict_awes[wrd] = awes[idx,:]
                    dict_count[wrd] = 1
                else:
                    dict_awes[wrd] = dict_awes[wrd]+awes[idx,:]
                    dict_count[wrd] = 1+dict_count[wrd]

for key in dict_awes.keys():
    dict_awes[key] = dict_awes[key]/dict_count[wrd]

print("AWE computed")



filenames_real = ["mc-30.csv","men.csv","mturk-287.csv", "mturk-771.csv", "rg-65.csv","rw.csv","semeval17.csv","simlex999.csv", "verb-143.csv", "wordsim353-rel.csv","wordsim353-sim.csv","yp-130.csv"]


#"simverb-3500.csv",

corr_acum=0

with open(os.path.join(path_save,pooling_method+".sim"), "w") as outresults:

    for filename_real in filenames_real:
        with open("/disk/scratch1/ramons/myapps/fairseq/examples/hubert/simple_kmeans/word-benchmarks/word-similarity/monolingual/en/"+filename_real) as inputfile:

            real_dist=[]
            rep_dist=[]
            for wrdl in inputfile.readlines():
                wrda=wrdl.strip().split(",")[1].split("-")[0]
                wrdb=wrdl.strip().split(",")[2].split("-")[0]
                distance_real=wrdl.strip().split(",")[3]

                if((wrda in dict_awes) and (wrdb in dict_awes)):
                    distance_rep = 1-distance.cosine(dict_awes[wrda], dict_awes[wrdb])

                    real_dist.append(distance_real)
                    rep_dist.append(distance_rep)

            corr = stats.spearmanr(real_dist, rep_dist)
            outresults.write(filename_real.replace(".csv","")+","+str(corr.correlation)+","+str(corr.pvalue)+"\n")
            corr_acum = corr.correlation+corr_acum

    outresults.write("AVG,"+str(corr_acum/float(len(filenames_real)))+","+"\n")


