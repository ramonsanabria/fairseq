
source /disk/scratch1/ramons/myenvs/fairseq/bin/activate
export CUDA_VISIBLE_DEVICES=0

tsv_dir=/disk/scratch1/ramons/data/zerospeech/tsv
layers="1 11 15 19 23"


ckpt_folder=/disk/scratch1/ramons/data/w2v2_models/
#model=xlsr_53_56k
model=wav2vec_vox_new
n_cluster=
pooling_methods="avg"


splits="french"
#splits="libri_dev"
#splits="libri libri_dev"
nshard=10

for split in ${splits}
do

	for layer in ${layers}
	do

		nshard_iter="$((${nshard}-1))"
		feat_dir=/disk/scratch1/ramons/data/w2v2_data/raw/zsc/${model}/${layer}/norm
		ckpt_path=${ckpt_folder}/${model}.pt


		#extracing features all ranks
		for rank in $(seq 0 ${nshard_iter})
		do

			python dump_w2v2_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}

		done


		for pooling in ${pooling_methods}
		do
			echo "POOLING: ${pooling}, SPLIT: ${split}"

			feat_pooled_dir=/disk/scratch1/ramons/data/w2v2_data/word_pooled/zsc/${model}/${layer}/norm/
			if [[ "${split}" == "mboshi" ]] || [[ "${split}" == "xitsonga" ]] || [[ "${split}" == "libri" ]] || [[ "${split}" == "libri_dev" ]] || [[ "${split}" == "libri_all" ]] 

                        then
				python generate_word_embeddings_single_norm.py ${split} ${model} ${layer} ${pooling} ${tsv_dir} ${nshard} ${feat_dir} ${feat_pooled_dir}
			else
				                                                                                                  python generate_word_embeddings_norm.py ${split} ${model} ${layer} ${pooling} ${tsv_dir} ${nshard} ${feat_dir} ${feat_pooled_dir}

			fi

		done
		rm -rf ${feat_dir}/*
	done
done


