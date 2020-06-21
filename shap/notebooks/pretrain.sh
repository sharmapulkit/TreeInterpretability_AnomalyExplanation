templates_path='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/submissionAll/Data/Templates'
outdir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/submissionAll/Data/AllTemplates_combined.csv'

awk 'FNR==1 && NR!=1 {while (/^url/) getline; } 1 {print}' $templates_path/*.csv > $outdir
