
source /disk/scratch1/ramons/myenvs/fairseq/bin/activate
export CUDA_VISIBLE_DEVICES=0

tsv_dir=/disk/scratch1/ramons/data/zerospeech/tsv
layers="1 11 15 19 23"

ckpt_folder=/disk/scratch1/ramons/data/hubert_models/
model=hubert_large_ll60k
n_cluster=


#splits="xitstonga buckeye english"
splits="buckeye_utd"
nshard=10

for split in ${splits}
do

	for layer in ${layers}
	do

		nshard_iter="$((${nshard}-1))"
		feat_dir=/disk/scratch1/ramons/data/hubert_data/raw/zsc/${model}/${layer}/norm
		ckpt_path=${ckpt_folder}/${model}.pt


		#extracing features all ranks
		for rank in $(seq 0 ${nshard_iter})
		do

			python dump_hubert_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}


		done


		echo "POOLING: ${pooling}, SPLIT: ${split}"

		feat_pooled_dir=/disk/scratch1/ramons/data/hubert_data/word_pooled/zsc/${model}/${layer}/norm/
		if [[ "${split}" == "mboshi" ]] || [[ "${split}" == "xitsonga" ]] 
		then
			python generate_frame_embeddings_single_norm.py ${split} ${model} ${layer} ${pooling} ${tsv_dir} ${nshard} ${feat_dir} ${feat_pooled_dir}
		else
			python generate_frame_embeddings_norm.py ${split} ${model} ${layer} ${pooling} ${tsv_dir} ${nshard} ${feat_dir} ${feat_pooled_dir}

		fi

		rm -rf ${feat_dir}/*
	done
done


