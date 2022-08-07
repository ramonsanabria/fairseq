
source /disk/scratch1/ramons/myenvs/fairseq/bin/activate
export CUDA_VISIBLE_DEVICES=0

tsv_dir=/disk/scratch1/ramons/data/zerospeech/tsv
layers="1 5 11 19 23"

ckpt_folder=/disk/scratch1/ramons/data/w2v2_models/
model=libri960_big
#model=xlsr_53_56k
n_cluster=
pooling_methods="avg"


splits="english french mandarin"
margins="a3 a2 a1 s1 s2 s3"
nshard=10
dim=256


for split in ${splits}
do

	for layer in ${layers}
	do
		nshard_iter="$((${nshard}-1))"

		pca_model=/disk/scratch1/ramons/data/w2v2_data/pca/models/zsc/${model}/${layer}/${split}
		rm -rf ${pca_model}/*
		mkdir -p ${pca_model}


		aux_dir=/disk/scratch1/ramons/data/w2v2_data/raw/zsc_aux_${split}
		rm -rf ${aux_dir}/*

		feat_dir_aux=${aux_dir}/${model}/${layer}/${split}
		rm -rf ${feat_dir_aux}/*

		ckpt_path=${ckpt_folder}/${model}.pt

		echo ${ckpt_path}

		#extracing features all ranks
		for rank in $(seq 0 ${nshard_iter})
		do

			python dump_w2v2_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir_aux}

		done

		#train PCA
		python pca.py ${feat_dir_aux} --output ${pca_model} --dim ${dim}

		echo "decoding PCA..."
		python apply_pca.py ${feat_dir_aux} --pca-path ${pca_model}/${dim}_pca --nshards ${nshard} --setname ${split}
		echo "decoding PCA decoded."

		for margin in ${margins}
		do
			feat_pooled_dir=/disk/scratch1/ramons/data/w2v2_data/word_pooled/zsc/${model}/${layer}/pca_${dim}/${margin}

			for pooling in ${pooling_methods}
			do
				echo "POOLING: ${pooling}, SPLIT: ${split}"

				python generate_word_embeddings_mw.py ${split} ${model} ${layer} ${pooling} ${tsv_dir} ${nshard} ${feat_dir_aux} ${feat_pooled_dir} ${margin}

			done
		done
		rm -rf ${aux_dir}

	done
done


