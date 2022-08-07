

n_samples=$(soxi -s ${2})

echo -e "${2}\t${n_samples}" >> ${1}
