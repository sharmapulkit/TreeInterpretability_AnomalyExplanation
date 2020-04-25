#python RF_postgreSQL.py --save_model=True --dataset_dir='/home/pulkit/Documents/PDF/umass/sem2/cs696ds/TreeInterpretability_AnomalyExplanation/' --current_target=runtime --num_tree_estimators=200 --max_depth=20 --outdir='/home/pulkit/Documents/PDF/umass/sem2/cs696ds/TreeInterpretability_AnomalyExplanation/shap/notebooks/trainedModels/' --treatmentTraining=True --treatment_combination='(0,2,0)' --timing_info_outfile='/home/pulkit/Documents/PDF/umass/sem2/cs696ds/TreeInterpretability_AnomalyExplanation/shap/notebooks/trainedModels' --TrainValTest_split='(0.1,0.1,0.1)'

##### Train for all treatment combinations ######
#for (( tr1=0; tr1 < 3; tr1++ )); do 
#	for (( tr2=0; tr2 < 3; tr2++ )); do
#		for (( tr3=0; tr3 < 3; tr3++ )); do
#			sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python rf_postgresql.py --save_model=True --dataset_dir='/home/s20psharma/cs696ds/TreeInterpretability_AnomalyExplanation/' --current_target=runtime --num_tree_estimators=200 --max_depth=20 --outdir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/trainedNets/' --treatmentTraining=True --treatment_combination="($tr1,$tr2,$tr3)" --timing_info_outfile='/home/s20psharma/cs696ds/TreeInterpretability_AnomalyExplanation/shap/notebooks/trainedNetworks/timing_runtime_tr'$tr1$tr2$tr3'_2.txt' --TrainValTest_split='(0.6,0.2,0.2)'
#		done;
#	done;
#done

################
 sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python evaluateRF.py --model_dir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/trainedNets/postgresTemplates/train_200_combos/rf_postgresql_runtime_200combos.pk' --dataset_dir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/postgresTemplates/test_200_combos.csv' --current_target=runtime --TrainValTest_split='(1.0,0.0,0.0)'


############ Test on test set of combinations #############
#testsetPath="/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/postgresTemplates/Subset/Test_subset/"
#for filepath in $testsetPath*.csv; do
#	file=$(basename $filepath)
#	sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python evaluateRF.py --model_dir=$testsetPath'../../../../trainedNets/postgresTemplates/train_200_combos/rf_postgresql_runtime_200combos.pk' --dataset_dir=$testsetPath$file --outdir=$testsetPath'../evaluation/'"${file:0:-4}"'_evaluation.txt' --current_target=runtime --TrainValTest_split='(1.0,0.0,0.0)';
#done
