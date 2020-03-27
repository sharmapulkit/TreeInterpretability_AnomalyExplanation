#python RF_postgreSQL.py --save_model=True --dataset_dir='/home/pulkit/Documents/PDF/umass/sem2/cs696ds/TreeInterpretability_AnomalyExplanation/' --current_target=runtime --num_tree_estimators=200 --max_depth=20 --outdir='/home/pulkit/Documents/PDF/umass/sem2/cs696ds/TreeInterpretability_AnomalyExplanation/shap/notebooks/trainedModels/' --treatmentTraining=True --treatment_combination='(0,2,0)' --timing_info_outfile='/home/pulkit/Documents/PDF/umass/sem2/cs696ds/TreeInterpretability_AnomalyExplanation/shap/notebooks/trainedModels' --TrainValTest_split='(0.1,0.1,0.1)'

#sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python interpret.py \
#--dataset_dir='/home/s20psharma/cs696ds/TreeInterpretability_AnomalyExplanation/datasets/postgres-results.csv' \
#--model_dir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/trainedNets/RF_postgres_Nest200_maxD20_runtime_tr212' \
#--outdir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/trainedNets/interpretedOuts.txt' \
#--TrainValTest_split='(0.6,0.2,0.2)' \
#--datapoint_start=0 \
#--datapoint_end=500

# sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python interpret.py --dataset_dir='/home/s20psharma/cs696ds/TreeInterpretability_AnomalyExplanation/datasets/postgres-results.csv' --model_dir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/trainedNets/RF_postgres_Nest200_maxD20_runtime_tr212' --outdir_ti_contribs='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/trainedNets/interpreted_TI_outs.txt' --outdir_shap_contribs='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/trainedNets/interpreted_SHAP_outs.txt' --TrainValTest_split='(0.6,0.2,0.2)' --datapoint_start=0 --datapoint_end=500

chunkSize=10000
for (( testDataStart=0; testDataStart < 300000 ; testDataStart+=$chunkSize )); do
	sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python interpret.py --dataset_dir='/home/s20psharma/cs696ds/TreeInterpretability_AnomalyExplanation/datasets/postgres-results.csv' --model_dir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/trainedNets/RF_postgres_Nest200_maxD20_runtime_tr011' --outdir_ti_contribs='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/trainedNets/interpreted_TI_outs_'$testDataStart'.txt' --outdir_shap_contribs='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/trainedNets/interpreted_SHAP_outs_'$testDataStart'.txt' --TrainValTest_split='(0.6,0.2,0.2)' --datapoint_start=$testDataStart --datapoint_end=$(($testDataStart+$chunkSize))
done
