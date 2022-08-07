
dataset=zerospeech

for element in libri_all
do

rm /disk/scratch1/ramons/data/${dataset}/tsv/tsv_aux

cat /disk/scratch1/ramons/data/${dataset}/tsv/${element}.path | parallel -j 32 ./add_samples.sh /disk/scratch1/ramons/data/${dataset}/tsv/tsv_aux {}

echo "/" > /disk/scratch1/ramons/data/${dataset}/tsv/${element}.tsv

cat /disk/scratch1/ramons/data/${dataset}/tsv/tsv_aux >> /disk/scratch1/ramons/data/${dataset}/tsv/${element}.tsv

rm /disk/scratch1/ramons/data/${dataset}/tsv/tsv_aux

done

