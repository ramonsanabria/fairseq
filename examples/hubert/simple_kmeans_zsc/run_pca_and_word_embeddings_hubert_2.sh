
source /disk/scratch1/ramons/myenvs/fairseq/bin/activate
export CUDA_VISIBLE_DEVICES=0

tsv_dir=/disk/scratch1/ramons/data/zerospeech/tsv
layers="1 11 15 19 23"

ckpt_folder=/disk/scratch1/ramons/data/hubert_models/
model=hubert_large_ll60k
n_cluster=
pooling_methods="avg"


splits="buckeye libri xitsonga mandarin french"
nshard=10
dims="130 1020"

for split in ${splits}
do

	for layer in ${layers}
	do
		nshard_iter="$((${nshard}-1))"

		pca_model=/disk/scratch1/ramons/data/hubert_data/pca/models/zsc/${model}/${layer}/${split}_2
		rm -rf ${pca_model}/*
		mkdir -p ${pca_model}


		aux_dir=/disk/scratch1/ramons/data/hubert_data/raw/zsc_aux_${split}_2
		rm -rf ${aux_dir}/*

		feat_dir_aux=${aux_dir}/${model}/${layer}/${split}_2
		rm -rf ${feat_dir_aux}/*

		ckpt_path=${ckpt_folder}/${model}.pt

		echo ${ckpt_path}

		#extracing features all ranks
		for rank in $(seq 0 ${nshard_iter})
		do
			python dump_hubert_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir_aux}

		done
		for dim in ${dims}
		do
			echo starting dimension: ${dim}
			feat_dir_aux_pca=${feat_dir_aux}/pca_${dim}_2
			mkdir -p ${feat_dir_aux_pca}

			#train PCA
			python pca.py ${feat_dir_aux} --output ${pca_model} --dim ${dim}

			echo "decoding PCA..."
			python apply_pca.py ${feat_dir_aux} ${feat_dir_aux_pca} --pca-path ${pca_model}/${dim}_pca --nshards ${nshard} --setname ${split}
			echo "decoding PCA decoded..."

			for pooling in ${pooling_methods}
			do
				feat_pooled_dir=/disk/scratch1/ramons/data/hubert_data/word_pooled/zsc/${model}/${layer}/pca_${dim}
				echo "POOLING: ${pooling}, SPLIT: ${split}"
				if [[ "${split}" == "mboshi" ]] || [[ "${split}" == "xitsonga" ]] || [[ "${split}" == "libri" ]] || [[ "${split}" == "libri_all" ]]

				then
				
				    	python generate_word_embeddings_single.py ${split} ${model} ${layer} ${pooling} ${tsv_dir} ${nshard} ${feat_dir_aux_pca} ${feat_pooled_dir}
				else
					python generate_word_embeddings.py ${split} ${model} ${layer} ${pooling} ${tsv_dir} ${nshard} ${feat_dir_aux_pca} ${feat_pooled_dir}
				fi
			done
			rm -rf ${feat_dir_aux_pca}
		done
		rm -rf ${aux_dir}
	done
done


