
source /disk/scratch1/ramons/myenvs/fairseq/bin/activate

FAIRSEQ_PATH=/disk/scratch1/ramons/myapps/fairseq

tsv_dir=/disk/scratch1/ramons/data/zerospeech/tsv/pt/french
split=all
nshard=10
rank=1
feat_dir=/disk/scratch1/ramons/data/hubert_data/raw/zsc/french_pt/mfcc
mkdir -p ${feat_dir}
lab_dir=/disk/scratch1/ramons/data/zerospeech/tsv/pt/french/

nshard_iter="$((${nshard}-1))"


rm -r ${feat_dir}/*

for split in valid all train;
do
	for rank in $(seq 0 ${nshard_iter})
	do
		python dump_mfcc_feature.py ${tsv_dir} ${split} ${nshard} ${rank} ${feat_dir}
	done
done

source deactivate
source /disk/scratch1/ramons/myenvs/p3_dave/bin/activate

for nclusters in 100 500;
do
	km_path=/disk/scratch1/ramons/data/hubert_data/raw/zsc/french_pt/k${nclusters}_mfcc.model

	python learn_kmeans.py ${feat_dir} all ${nshard} ${km_path} ${nclusters} --percent 0.1

	for split in valid train;
	do
		final_folder=${lab_dir}/mfcc/${nclusters}
		mkdir -p ${final_folder}

		for rank in $(seq 0 ${nshard_iter})
		do
			python dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${final_folder}

		done
		for rank in $(seq 0 ${nshard_iter}); do
		  cat ${final_folder}/${split}_${rank}_${nshard}.km
		done > ${final_folder}/${split}_mfcc.km

		for rank in $(seq 0 ${nshard_iter}); do
			rm ${final_folder}/${split}_${rank}_${nshard}.km
		done 

		for x in $(seq 0 $((n_clusters - 1))); do
			echo "$x 1"
		done >> ${final_folder}/dict.km.txt
	done
done



