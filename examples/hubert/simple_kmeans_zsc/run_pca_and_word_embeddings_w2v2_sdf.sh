
source /disk/scratch1/ramons/myenvs/fairseq/bin/activate
export CUDA_VISIBLE_DEVICES=0

tsv_dir=/disk/scratch1/ramons/data/zerospeech/tsv
layers="1 5 11 19 23"

ckpt_folder=/disk/scratch1/ramons/data/w2v2_models/
model=libri960_big
#model=xlsr_53_56k


splits="xitsonga"
nshard=10
dims="256"

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

		#extracing features all ranks
		for rank in $(seq 0 ${nshard_iter})
		do

			feat_dir_aux_pca=${feat_dir_aux}/pca_256
			python dump_w2v2_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir_aux}

		done

		for dim in ${dims}
		do
			feat_pooled_dir=/disk/scratch1/ramons/data/w2v2_data/word_pooled/zsc/${model}/${layer}/pca_${dim}
			feat_dir_aux_pca=${feat_dir_aux}/pca_${dim}
			mkdir -p ${feat_dir_aux_pca}


			#train PCA
			python pca.py ${feat_dir_aux} --output ${pca_model} --dim ${dim}

			echo "decoding PCA..."
			python apply_pca.py ${feat_dir_aux} ${feat_dir_aux_pca} --pca-path ${pca_model}/${dim}_pca --nshards ${nshard} --setname ${split}
			echo "decoding PCA decoded..."

			if [[ "${split}" == "mboshi" ]] || [[ "${split}" == "xitsonga" ]]
			then

				python generate_word_embeddings_single_sdf.py ${split} ${model} ${layer} ${tsv_dir} ${nshard} ${feat_dir_aux_pca} ${feat_pooled_dir} 
			else
				python generate_word_embeddings_sdf.py ${split} ${model} ${layer} ${tsv_dir} ${nshard} ${feat_dir_aux_pca} ${feat_pooled_dir} 
			fi

			rm -rf ${feat_dir_aux_pca}
		done
		rm -rf ${dir_aux}
	done
done


