
source /disk/scratch1/ramons/myenvs/fairseq/bin/activate
export CUDA_VISIBLE_DEVICES=1

tsv_dir=/disk/scratch1/ramons/data/zerospeech/tsv

ckpt_folder=/disk/scratch1/ramons/data/hubert_models/
model=hubert_large_ll60k
n_cluster=
pooling_methods="sub10 avg max sum"


splits="buckeye"
nshard=1

for split in ${splits}
do
	nshard_iter="$((${nshard}-1))"
	feat_dir=/disk/scratch1/ramons/data/hubert_data/raw/zsc/mfcc/norm
	ckpt_path=${ckpt_folder}/${model}.pt


	#extracing features all ranks
	for rank in $(seq 0 ${nshard_iter})
	do

		python dump_mfcc_feature.py ${tsv_dir} ${split} ${nshard} ${rank} ${feat_dir} --sample_rate 16000

	done


	for pooling in ${pooling_methods}
	do
		feat_pooled_dir=/disk/scratch1/ramons/data/mfcc_data/word_pooled/zsc/norm/
		echo "POOLING: ${pooling}, SPLIT: ${split}"
		if [[ "${split}" == "mboshi" ]] || [[ "${split}" == "xitsonga" ]] || [[ "${split}" == "xitsonga" ]]

		then

			python generate_word_embeddings_mfcc_single_norm.py ${split} dummy dummy ${pooling} ${tsv_dir} ${nshard} ${feat_dir} ${feat_pooled_dir}
																		      else
																python generate_word_embeddings_mfcc_norm.py ${split} dummy dummy ${pooling} ${tsv_dir} ${nshard} ${feat_dir} ${feat_pooled_dir}
		fi


	done
	rm -rf ${feat_dir}/*
done


