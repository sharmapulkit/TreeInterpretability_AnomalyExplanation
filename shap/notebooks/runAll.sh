############# Split the data set #############


########### Train a Random Forest ############
#dataset_dir="/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/postgresTemplates/Subset/"
#sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python rf_postgresql.py --save_model=True --evaluation=False --dataset_dir=$dataset_dir'train_allCombs_header.csv' --current_target=runtime --num_tree_estimators=200 --max_depth=20 --treatmentTraining=False --outdir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/submissionAll/rf_postgresql_runtime_Nest200_maxD20.pk' --TrainValTest_split='(1.0,0.0,0.0)'

######### Evaluate Trained Random Forest to obtain R2 Score ########


######### Compute Attribution value variance ###########
python attributionVariance.py




