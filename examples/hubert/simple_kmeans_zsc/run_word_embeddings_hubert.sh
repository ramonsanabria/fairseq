
source /disk/scratch1/ramons/myenvs/fairseq/bin/activate
export CUDA_VISIBLE_DEVICES=3

tsv_dir=/disk/scratch1/ramons/data/zerospeech/tsv
layers="15 19 23"

ckpt_folder=/disk/scratch1/ramons/data/hubert_models/
model=hubert_large_ll60k
n_cluster=
pooling_methods="one"


#splits="buckeye"
splits="buckeye"
nshard=10

for split in ${splits}
do

	for layer in ${layers}
	do

		nshard_iter="$((${nshard}-1))"
		feat_dir=/disk/scratch1/ramons/data/hubert_data/raw/zsc/${model}/${layer}/
		ckpt_path=${ckpt_folder}/${model}.pt


		#extracing features all ranks
		for rank in $(seq 0 ${nshard_iter})
		do

			python dump_hubert_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}

		done


		for pooling in ${pooling_methods}
		do
			echo "POOLING: ${pooling}, SPLIT: ${split}"
			feat_pooled_dir=/disk/scratch1/ramons/data/hubert_data/word_pooled/zsc/${model}/${layer}/

			if [[ "${split}" == "mboshi" ]] || [[ "${split}" == "xitsonga" ]] || [[ "${split}" == "libri" ]]

			then
				   python generate_word_embeddings_single.py ${split} ${model} ${layer} ${pooling} ${tsv_dir} ${nshard} ${feat_dir} ${feat_pooled_dir}

			else
								                                                                                                                   python generate_word_embeddings.py ${split} ${model} ${layer} ${pooling} ${tsv_dir} ${nshard} ${feat_dir} ${feat_pooled_dir}
																					fi

		done
		rm -rf ${feat_dir}/*
	done
done


