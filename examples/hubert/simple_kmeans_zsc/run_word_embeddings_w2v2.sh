
source /disk/scratch1/ramons/myenvs/fairseq/bin/activate
export CUDA_VISIBLE_DEVICES=0

tsv_dir=/disk/scratch1/ramons/data/zerospeech/tsv
layers="1 5 11 19 23"

ckpt_folder=/disk/scratch1/ramons/data/w2v2_models/
#model=libri960_big
model=xlsr_53_56k
pooling_methods="avg"


splits="english french mandarin"
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


		for pooling in ${pooling_methods}
		do
			echo "POOLING: ${pooling}, SPLIT: ${split}"

			feat_pooled_dir=/disk/scratch1/ramons/data/w2v2_data/word_pooled/zsc/${model}/${layer}/
			python generate_word_embeddings.py ${split} ${model} ${layer} ${pooling} ${tsv_dir} ${nshard} ${feat_dir} ${feat_pooled_dir}

		done
		rm -rf ${feat_dir}/*
	done
done


