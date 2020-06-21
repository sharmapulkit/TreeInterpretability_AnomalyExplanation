train_templates_path='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/submissionAll/Data/Templates_subset600/train'
train_outdir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/submissionAll/Data/Templates_subset600/Templates_subset600_combined_train.csv'

awk 'FNR==1 && NR!=1 {while (/^url/) getline; } 1 {print}' $train_templates_path/*.csv > $train_outdir


test_templates_path='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/submissionAll/Data/Templates_subset600/test'
test_outdir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/submissionAll/Data/Templates_subset600/Templates_subset600_combined_test.csv'

awk 'FNR==1 && NR!=1 {while (/^url/) getline; } 1 {print}' $test_templates_path/*.csv > $test_outdir
