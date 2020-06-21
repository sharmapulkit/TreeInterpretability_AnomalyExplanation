dataset_dir="/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/submissionAll/Data/Templates_subset600/"
model_out_dir="/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/submissionAll/"
sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python rf_postgresql.py --save_model=True --evaluation=False --dataset_dir=$dataset_dir'Templates_subset600_combined_train.csv' --current_target=runtime --num_tree_estimators=200 --max_depth=20 --treatmentTraining=False --outdir=$model_out_dir'rf_postgresql_runtime_Nest200_maxD20.pk' --TrainValTest_split='(1.0,0.0,0.0)'
