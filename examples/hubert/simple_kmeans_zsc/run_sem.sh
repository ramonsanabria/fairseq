
source /disk/scratch1/ramons/myenvs/p3_dave/bin/activate
export CUDA_VISIBLE_DEVICES=1

tsv_dir=/disk/scratch1/ramons/data/librispeech/tsv/
layers="1 5 11 23"

ckpt_folder=/disk/scratch1/ramons/data/hubert_models/
model=hubert_xtralarge_ll60k
n_cluster=
pooling_methods="avg sum max"


for layer in ${layers}
do

	for pooling_method in ${pooling_methods}
	do
		python compute_semantics.py --indir /disk/scratch1/ramons/data/hubert_data/word_pooled/${model}/${layer}/ --poolingmethod ${pooling_method} --outdir /disk/scratch1/ramons/data/hubert_data/word_pooled/${model}/${layer}/ 

	done

done



