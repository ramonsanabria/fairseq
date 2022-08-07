
source /disk/scratch1/ramons/myenvs/fairseq/bin/activate

FIARSEQ_PATH=/disk/scratch1/ramons/myapps/fairseq


python $FIARSEQ_PATH/fairseq_cli/hydra_train.py \
  --config-dir $FIARSEQ_PATH//examples/hubert/config/pretrain \
  --config-name hubert_base_librispeech \
  task.data=/path/to/data task.label_dir=/path/to/labels task.labels='["km"]' model.label_rate=100
