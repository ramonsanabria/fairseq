
source /disk/scratch1/ramons/myenvs/fairseq/bin/activate

FAIRSEQ_PATH=/disk/scratch1/ramons/myapps/fairseq

tsv_dir=/disk/scratch1/ramons/data/zerospeech/tsv/pt/french
split=all
nshard=10
nclusters=100
rank=1
feat_dir=/disk/scratch1/ramons/data/hubert_data/raw/zsc/french_pt/
lab_dir=/disk/scratch1/ramons/data/zerospeech/tsv/pt/french

nshard_iter="$((${nshard}-1))"

rm -r ${feat_dir}/*

for split in dev;
do
	for rank in $(seq ${nshard_iter})
	do
		python dump_mfcc_feature.py ${tsv_dir} ${split} ${nshard} ${rank} ${feat_dir}
	done
done


for nclusters in 100 500;
do
	km_path=/disk/scratch1/ramons/data/hubert_data/raw/zsc/french_pt/kmeans_${nclusters}
	mkdir -p ${km_path}

	python learn_kmeans.py ${feat_dir} all ${nshard} ${km_path} ${n_cluster} --percent 0.1
	exit

	for split in dev;
	do

		for rank in $(seq ${nshard_iter})
		do
			python dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}

		done
	done
done


rm -r ${feat_dir}


