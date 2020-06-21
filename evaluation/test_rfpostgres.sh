testsetPath="/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/submissionAll/Data/Templates_subset600/"
modelPath="/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/submissionAll/"
sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python evaluateRF.py --model_dir=$modelPath'rf_postgresql_runtime_Nest200_maxD20.pk' --dataset_dir=$testsetPath'Templates_subset600_combined_test.csv' --current_target=runtime --TrainValTest_split='(1.0,0.0,0.0)'




