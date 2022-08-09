
source /disk/scratch1/ramons/myenvs/fairseq/bin/activate

FAIRSEQ_PATH=/disk/scratch1/ramons/myapps/fairseq


python $FAIRSEQ_PATH/fairseq_cli/hydra_train.py \
  --config-dir $FAIRSEQ_PATH//examples/hubert/config/pretrain \
  --config-name hubert_large_french \
  task.data=/disk/scratch1/ramons/data/zerospeech/tsv/ task.label_dir=/path/to/labels task.labels='["km"]' model.label_rate=100
