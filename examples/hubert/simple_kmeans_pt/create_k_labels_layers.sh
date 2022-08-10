
source /disk/scratch1/ramons/myenvs/fairseq/bin/activate

export CUDA_VISIBLE_DEVICES=0

FAIRSEQ_PATH=/disk/scratch1/ramons/myapps/fairseq

tsv_dir=/disk/scratch1/ramons/data/zerospeech/tsv/pt/french
split=all
nshard=10
lab_dir=/disk/scratch1/ramons/data/zerospeech/tsv/pt/french/

ckpt_folder=/disk/scratch1/ramons/data/hubert_models/
ckpt_path=${ckpt_folder}/hubert_large_ll60k.pt

#layers="1 11 15 19 23"
layers="1"

nshard_iter="$((${nshard}-1))"



for layer in ${layers}
do

feat_dir=/disk/scratch1/ramons/data/hubert_data/raw/zsc/french_pt_hb/l_${layer}
mkdir -p ${feat_dir}
rm -r ${feat_dir}/*

for split in all valid train;
do
	for rank in $(seq 0 ${nshard_iter})
	do
		python dump_hubert_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}

	done
done
done


source deactivate
source /disk/scratch1/ramons/myenvs/p3_dave/bin/activate

for layer in ${layers}
do

feat_dir=/disk/scratch1/ramons/data/hubert_data/raw/zsc/french_pt_hb/l_${layer}
for nclusters in 100 500;
do
	km_path=/disk/scratch1/ramons/data/hubert_data/raw/zsc/french_pt_hb/l_${layer}/k${nclusters}_l${layer}.model

	python learn_kmeans.py ${feat_dir} all ${nshard} ${km_path} ${nclusters} --percent 0.1

	for split in valid train;
	do

		for rank in $(seq 0 ${nshard_iter})
		do
			python dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}

		done
		final_folder=${lab_dir}/l${layer}/${nclusters}

		mkdir -p ${final_folder}


		for rank in $(seq 0 ${nshard_iter}); do
		  cat $final_folder/${split}_${rank}_${nshard}.km
		done > ${final_folder}/${split}.km

		for rank in $(seq 0 ${nshard_iter}); do
		  rm $final_folder/${split}_${rank}_${nshard}.km
		done 

		for x in $(seq 0 $((n_clusters - 1))); do
			  echo "$x 1"
		 done >> ${final_folder}/dict.km.txt


	done
done
done

rm -r ${feat_dir}
