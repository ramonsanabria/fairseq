
source /disk/scratch1/ramons/myenvs/fairseq/bin/activate

FAIRSEQ_PATH=/disk/scratch1/ramons/myapps/fairseq
ckpt_folder=/disk/scratch1/ramons/data/hubert_models/
ckpt_path=${ckpt_folder}/hubert_large_ll60k.pt

layers=mfcc
n_clusters=500

export CUDA_VISIBLE_DEVICES=1,2,3



for layer in ${layers}
do

	for n_cluster in ${n_clusters}
	do


python $FAIRSEQ_PATH/fairseq_cli/hydra_train.py \
  --config-dir $FAIRSEQ_PATH//examples/hubert/config/pretrain \
  --config-name hubert_large_french \
  task.data=/disk/scratch1/ramons/data/zerospeech/tsv/pt/french/ task.label_dir=/disk/scratch1/ramons/data/zerospeech/tsv/pt/french/${layer}/${n_cluster} task.labels='["km"]' model.label_rate=100  

	done
done

