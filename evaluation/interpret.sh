column='baseline'
datadir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/submissionAll/Data/Templates_subset600/Templates_subset600_combined_test.csv'
modeldir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/submissionAll/rf_postgresql_runtime_Nest200_maxD20.pk'
outbasedir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/submissionAll/Intervened/explicit_intervention/'$column
mkdir $outbasedir
outdir=$outbasedir'/explanations'
mkdir $outdir
chunkSize=200
for (( testDataStart=0; testDataStart < 16600 ; testDataStart+=$chunkSize )); do
	sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python interpret.py --dataset_dir=$datadir --model_dir=$modeldir --outdir_ti_contribs=$outdir'/interpreted_TI_outs_'$testDataStart'.txt' --outdir_shap_contribs=$outdir'/interpreted_SHAP_outs_'$testDataStart'.txt' --TrainValTest_split='(0.0,0.0,1.0)' --datapoint_start=$testDataStart --datapoint_end=$(($testDataStart+$chunkSize));
done


column='index_level'
datadir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/submissionAll/Intervened/explicit_intervention/'$column'/Templates_subset600_combined_'$column'_test.csv'
modeldir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/submissionAll/rf_postgresql_runtime_Nest200_maxD20.pk'
outbasedir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/submissionAll/Intervened/explicit_intervention/'$column
mkdir $outbasedir
outdir=$outbasedir'/explanations'
mkdir $outdir
chunkSize=200
for (( testDataStart=0; testDataStart < 16600 ; testDataStart+=$chunkSize )); do
	sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python interpret.py --dataset_dir=$datadir --model_dir=$modeldir --outdir_ti_contribs=$outdir'/interpreted_TI_outs_'$testDataStart'.txt' --outdir_shap_contribs=$outdir'/interpreted_SHAP_outs_'$testDataStart'.txt' --TrainValTest_split='(0.0,0.0,1.0)' --datapoint_start=$testDataStart --datapoint_end=$(($testDataStart+$chunkSize));
done


column='memory_level'
datadir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/submissionAll/Intervened/explicit_intervention/'$column'/Templates_subset600_combined_'$column'_test.csv'
modeldir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/submissionAll/rf_postgresql_runtime_Nest200_maxD20.pk'
outbasedir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/submissionAll/Intervened/explicit_intervention/'$column
mkdir $outbasedir
outdir=$outbasedir'/explanations'
mkdir $outdir
chunkSize=200
for (( testDataStart=0; testDataStart < 16600 ; testDataStart+=$chunkSize )); do
	sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python interpret.py --dataset_dir=$datadir --model_dir=$modeldir --outdir_ti_contribs=$outdir'/interpreted_TI_outs_'$testDataStart'.txt' --outdir_shap_contribs=$outdir'/interpreted_SHAP_outs_'$testDataStart'.txt' --TrainValTest_split='(0.0,0.0,1.0)' --datapoint_start=$testDataStart --datapoint_end=$(($testDataStart+$chunkSize));
done


column='page_cost'
datadir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/submissionAll/Intervened/explicit_intervention/'$column'/Templates_subset600_combined_'$column'_test.csv'
modeldir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/submissionAll/rf_postgresql_runtime_Nest200_maxD20.pk'
outbasedir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/submissionAll/Intervened/explicit_intervention/'$column
mkdir $outbasedir
outdir=$outbasedir'/explanations'
mkdir $outdir
chunkSize=200
for (( testDataStart=0; testDataStart < 16600 ; testDataStart+=$chunkSize )); do
	sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python interpret.py --dataset_dir=$datadir --model_dir=$modeldir --outdir_ti_contribs=$outdir'/interpreted_TI_outs_'$testDataStart'.txt' --outdir_shap_contribs=$outdir'/interpreted_SHAP_outs_'$testDataStart'.txt' --TrainValTest_split='(0.0,0.0,1.0)' --datapoint_start=$testDataStart --datapoint_end=$(($testDataStart+$chunkSize));
done




