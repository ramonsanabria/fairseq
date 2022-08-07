
source /disk/scratch1/ramons/myenvs/fairseq/bin/activate
export CUDA_VISIBLE_DEVICES=0

tsv_dir=/disk/scratch1/ramons/data/zerospeech/tsv

ckpt_folder=/disk/scratch1/ramons/data/hubert_models/
model=hubert_large_ll60k
n_cluster=


split="buckeye"
dim="130"
pooling_method="sub10"
source_pca=""
norm="norm"
layer=23


name_source_file=${split}_features_${pooling_method}
source_feats=/disk/scratch1/ramons/data/hubert_data/word_pooled/zsc/hubert_large_ll60k/23/${norm}/${source_pca}/${name_source_file}.npy
source_words=/disk/scratch1/ramons/data/hubert_data/word_pooled/zsc/hubert_large_ll60k/23/${norm}/${source_pca}/${name_source_file}.wrd

pca_model=/disk/scratch1/ramons/data/hubert_data/pca/models/zsc/${model}/${layer}/${split}/post_${pooling_method}_s${source_pca}/${dim}_pca
feat_pooled_dir=/disk/scratch1/ramons/data/hubert_data/word_pooled/zsc/${model}/${layer}/post_pca_${dim}/source_${source_pca}_${norm}
mkdir -p ${feat_pooled_dir}
mkdir -p ${pca_model}


#train PCA
python pca_post.py ${source_feats} --output ${pca_model} --dim ${dim}

echo "decoding PCA..."
python apply_pca_post.py ${source_feats} ${feat_pooled_dir} --pca-path ${pca_model}/${dim}_pca
cp ${source_words}  ${feat_pooled_dir}

