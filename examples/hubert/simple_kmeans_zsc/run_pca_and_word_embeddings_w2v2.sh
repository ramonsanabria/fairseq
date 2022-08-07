
source /disk/scratch1/ramons/myenvs/fairseq/bin/activate
export CUDA_VISIBLE_DEVICES=3

tsv_dir=/disk/scratch1/ramons/data/zerospeech/tsv
layers="1 5 11 19 23"

ckpt_folder=/disk/scratch1/ramons/data/w2v2_models/
#model=wav2vec_vox_new
model=xlsr_53_56k
pooling_methods="sub10"


splits="buckeye xitsonga libri mandarin french"
nshard=10
dim="102"


for split in ${splits}
do

	for layer in ${layers}
	do
		nshard_iter="$((${nshard}-1))"

		pca_model=/disk/scratch1/ramons/data/w2v2_data/pca/models/zsc/${model}/${layer}/${split}
		rm -rf ${pca_model}/*
		mkdir -p ${pca_model}

		feat_pooled_dir=/disk/scratch1/ramons/data/w2v2_data/word_pooled/zsc/${model}/${layer}/pca_${dim}

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
		python apply_pca.py ${feat_dir_aux} ${feat_dir_aux} --pca-path ${pca_model}/${dim}_pca --nshards ${nshard} --setname ${split}
		echo "decoding PCA decoded..."

		for pooling in ${pooling_methods}
		do
			echo "POOLING: ${pooling}, SPLIT: ${split}"
			if [[ "${split}" == "mboshi" ]] || [[ "${split}" == "xitsonga" ]] || [[ "${split}" == "libri" ]] || [[ "${split}" == "libri_all" ]]
			then

				python generate_word_embeddings_single_norm.py ${split} ${model} ${layer} ${pooling} ${tsv_dir} ${nshard} ${feat_dir_aux} ${feat_pooled_dir} 
			else
				python generate_word_embeddings_norm.py ${split} ${model} ${layer} ${pooling} ${tsv_dir} ${nshard} ${feat_dir_aux} ${feat_pooled_dir} 
			fi
			rm -rf ${feat_dir_aux_pca}

		done
		rm -rf ${dir_aux}
	done
done


