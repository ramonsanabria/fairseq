
source /disk/scratch1/ramons/myenvs/fairseq/bin/activate
export CUDA_VISIBLE_DEVICES=3

tsv_dir=/disk/scratch1/ramons/data/zerospeech/tsv
layers="1 5 11 19 23"

ckpt_folder=/disk/scratch1/ramons/data/w2v2_models/
model=libri960_big
#model=xlsr_53_56k

splits="xitsonga"
nshard=10

for split in ${splits}
do

	for layer in ${layers}
	do

		nshard_iter="$((${nshard}-1))"
		feat_dir=/disk/scratch1/ramons/data/w2v2_data/raw/zsc/${model}/${layer}/
		ckpt_path=${ckpt_folder}/${model}.pt


		#extracing features all ranks
		for rank in $(seq 0 ${nshard_iter})
		do

			python dump_w2v2_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}

		done
		feat_pooled_dir=/disk/scratch1/ramons/data/w2v2_data/word_pooled/zsc/${model}/${layer}/

			if [[ "${split}" == "mboshi" ]] || [[ "${split}" == "xitsonga" ]]
                        then

                                python generate_word_embeddings_single_sdf.py ${split} ${model} ${layer} ${tsv_dir} ${nshard} ${feat_dir} ${feat_pooled_dir}
                        else
                                python generate_word_embeddings_sdf.py ${split} ${model} ${layer} ${tsv_dir} ${nshard} ${feat_dir} ${feat_pooled_dir}
                        fi


		rm -rf ${feat_dir}/*
	done
done


