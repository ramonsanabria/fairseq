
source /disk/scratch1/ramons/myenvs/p3_dave/bin/activate
export CUDA_VISIBLE_DEVICES=1

tsv_dir=/disk/scratch1/ramons/data/librispeech/tsv/
#layers="1 5 11 23"
layers="1"
clusters="1000 5000 10000 15000"

ckpt_folder=/disk/scratch1/ramons/data/hubert_models/
model=hubert_xtralarge_ll60k

for layer in ${layers}
do
	for cluster in ${clusters}
	do

	python compute_cluster_metrics.py --indir /disk/scratch1/ramons/data/hubert_data/word_pooled/${model}/${layer}/  --outdir /disk/scratch1/ramons/data/hubert_data/word_pooled/${model}/${layer}/ --clusters ${cluster} --sets dev-clean,test-clean

	done
done



