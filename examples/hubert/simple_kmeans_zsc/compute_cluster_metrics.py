import numpy as np
import os
import argparse
import sys
from scipy import stats
from scipy.spatial import distance
import matplotlib.pyplot as plt


def get_parser():
    parser = argparse.ArgumentParser(
        description="compute kmeans codebook from kaldi-computed feats"
    )
    # fmt: off
    parser.add_argument('--indir', help='directory with the input features', required=True)
    parser.add_argument('--outdir', help='directory where to write the output features', required=True)
    parser.add_argument('--clusters', help='clustrers', required=True)
    parser.add_argument('--sets', help='sets', required=True)

    return parser

parser = get_parser()
args = parser.parse_args()

path_in=args.outdir
path_save=args.outdir
clusters=args.clusters
sets=args.clusters.strip().split(",")



wordplot=False

for pooling in ["avg", "sum", "max"]:
    for filename in os.listdir(path_in):
        data_points = {}
        if(filename.endswith("_"+pooling+".c"+clusters)):

            filename_full=os.path.join(path_in,filename)
            with open(filename_full) as inputfile:
                for line in inputfile.readlines():
                    if(int(line.strip()) not in data_points):
                        data_points[int(line.strip())]=1
                    else:
                        data_points[int(line.strip())]=data_points[int(line.strip())] + 1
            data_points_list = [int(data_points[key]) for key in data_points.keys() ]
            data_points_list = sorted(data_points_list)
            data_points_list.reverse()
            plt.plot(data_points_list, label=filename.replace(".c"+str(clusters),""))


ax = plt.gca()
ax.set_yscale('log')
plt.legend(loc="upper right")
plt.ylabel('number of instances')
plt.xlabel('cluster ID')
plt.savefig(path_save+"/comparission_pooling_hist_c"+clusters+".png")
print("graph plotted here: "+path_save+"/comparission_pooling_hist_c"+clusters+".png")
