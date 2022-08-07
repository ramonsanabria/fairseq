
source /disk/scratch1/ramons/myenvs/fairseq/bin/activate
export CUDA_VISIBLE_DEVICES=1

tsv_dir=/disk/scratch1/ramons/data/librispeech/tsv/
#layers="1 5 17 23"
layers="1 11 17 23"

ckpt_folder=/disk/scratch1/ramons/data/hubert_models/
#model=hubert_xtralarge_ll60k
model=hubert_large_ll60k
n_cluster=

splits="dev-clean_10k_seg_zero_3.0 dev-clean_10k_seg_zero_0.0 dev-clean_10k_seg_zero_0.36"

nshard=1

for split in ${splits}
do

	for layer in ${layers}
	do

		nshard_iter="$((${nshard}-1))"
		feat_dir=/disk/scratch1/ramons/data/hubert_data/row/${model}/${layer}/
		sub_folder=/disk/scratch1/ramons/asses_representations/submissions/hubert/${model}/${layer}/phonetic/${split}
		ckpt_path=${ckpt_folder}/${model}.pt


		#extracing features all ranks
		for rank in $(seq 0 ${nshard_iter})
		do

			rm -rf ${sub_folder}
			mkdir -p ${sub_folder}

			python dump_hubert_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}
			python convert_feats_npy.py ${feat_dir} ${split} ${sub_folder} 
			rm -rf ${feat_dir}

		done

	done
done


