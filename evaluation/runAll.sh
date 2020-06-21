###### FLAGS 
SPLITDB_FLAG=0
PRETRAIN_PY_FLAG=1
TRAIN_FLAG=1
TEST_FLAG=1
INTERPRET_FLAG=1
EXPLICIT_INTERVENTION_SETUP_FLAG=1
EXPLICIT_INTERVENTION_INTERPRETATIONS_FLAG=1
EXPLICIT_INTERVENTION_ACCURACY_FLAG=1
EXPLICIT_INTERVENTION_VARIANCE_FLAG=1

BaseExperimentPath='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/submissionAll/'

############# Split the data set #############
python makeTemplates.py

############ Organize the data set ###########
python pretrain.py
sh pretrain.sh

########### Train a Random Forest ############
if [ "$TRAIN_FLAG" == 1 ]; then
	dataset_dir=$BaseExperimentPath"Data/Templates_subset600/"
	model_out_dir=$BaseExperimentPath'rf_postgresql_runtime_Nest200_maxD20.pk'
	sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python rf_postgresql.py --save_model=True --evaluation=False --dataset_dir=$dataset_dir'Templates_subset600_combined_train.csv' --current_target=runtime --num_tree_estimators=200 --max_depth=20 --treatmentTraining=False --outdir=$model_out_dir --TrainValTest_split='(1.0,0.0,0.0)'
fi

######### Evaluate Trained Random Forest to obtain R2 Score ########
if [ "$TEST_FLAG" == 1]; then
	testsetPath=$BaseExperimentPath"Data/Templates_subset600/Templates_subset600_combined_test.csv"
	modelPath=$BaseExperimentPath'rf_postgresql_runtime_Nest200_maxD20.pk'
	sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python evaluateRF.py --model_dir=$modelPath --dataset_dir=$testsetPath --current_target=runtime --TrainValTest_split='(1.0,0.0,0.0)';
fi

################ Explicit Intervention Setup ####################
if [ "$EXPLICIT_INTERVENTION_SETUP_FLAG" == 1]; then
	python flipped_bits.py
fi

############### Generate Interpretations ########################
if [ "$EXPLICIT_INTERVENTION_INTERPRETATIONS" == 1]; then
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
fi

################ Explicit Intervention Accuracy #################
if [ "$EXPLICIT_INTERVENTION_ACCURACY_FLAG" == 1]; then
	python attributionAccuracy.py
fi

######### Compute Attribution value variance ###########
if [ "$EXPLICIT_INTERVENTION_VARIACNE_FLAG" == 1]; then
	python attributionVariance.py
fi







