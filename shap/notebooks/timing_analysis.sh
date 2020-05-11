#python RF_postgreSQL.py --save_model=True --dataset_dir='/home/pulkit/Documents/PDF/umass/sem2/cs696ds/TreeInterpretability_AnomalyExplanation/' --current_target=runtime --num_tree_estimators=200 --max_depth=20 --outdir='/home/pulkit/Documents/PDF/umass/sem2/cs696ds/TreeInterpretability_AnomalyExplanation/shap/notebooks/trainedModels/' --treatmentTraining=True --treatment_combination='(0,2,0)' --timing_info_outfile='/home/pulkit/Documents/PDF/umass/sem2/cs696ds/TreeInterpretability_AnomalyExplanation/shap/notebooks/trainedModels' --TrainValTest_split='(0.1,0.1,0.1)'

##### Train for all treatment combinations ######
#for (( tr1=0; tr1 < 3; tr1++ )); do 
#	for (( tr2=0; tr2 < 3; tr2++ )); do
#		for (( tr3=0; tr3 < 3; tr3++ )); do
#			sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python rf_postgresql.py --save_model=True --dataset_dir='/home/s20psharma/cs696ds/TreeInterpretability_AnomalyExplanation/' --current_target=runtime --num_tree_estimators=200 --max_depth=20 --outdir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/trainedNets/' --treatmentTraining=True --treatment_combination="($tr1,$tr2,$tr3)" --timing_info_outfile='/home/s20psharma/cs696ds/TreeInterpretability_AnomalyExplanation/shap/notebooks/trainedNetworks/timing_runtime_tr'$tr1$tr2$tr3'_2.txt' --TrainValTest_split='(0.6,0.2,0.2)'
#		done;
#	done;
#done

############
#garage='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/'
#sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python rf_postgresql.py --save_model=True --evaluation=True --dataset_dir=$garage'dataset/postgresTemplates/train_200_combos.csv' --current_target=runtime --num_tree_estimators=200 --max_depth=5 --treatmentTraining=False --outdir=$garage'TimingAnalysis/Depth/rf_postgresql_runtime_200combos_MaxDepth_blah.pk' --TrainValTest_split='(0.9,0.0,0.1)' --timing_info_outfile=$garage'TimingAnalysis/Depth/rf_postgresql_runtime_200combos_MaxDepth_blah.txt'

#garage='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/'
#for (( depth_mult=1; depth_mult < 12; depth_mult++ )); do
#	depth=$(( $depth_mult*5 ))
#	sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python rf_postgresql.py --save_model=True --evaluation=True --dataset_dir=$garage'dataset/postgresTemplates/train_200_combos.csv' --current_target=runtime --num_tree_estimators=200 --max_depth=$depth --treatmentTraining=False --outdir=$garage'TimingAnalysis/Depth/rf_postgresql_runtime_200combos_MaxDepth'$depth'.pk' --TrainValTest_split='(0.9,0.0,0.1)' --timing_info_outfile=$garage'TimingAnalysis/Depth/rf_postgresql_runtime_200combos_MaxDepth'$depth'.txt';
#done

#garage='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/'
#for x in 1 3 5 7 9; do
#	trainSplit=0.0$x
#	testSplit=0.0$(( 10-$x ))
#	sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python rf_postgresql.py --save_model=True --evaluation=True --dataset_dir=$garage'dataset/postgresTemplates/train_200_combos.csv' --current_target=runtime --num_tree_estimators=200 --max_depth=20 --treatmentTraining=False --outdir=$garage'TimingAnalysis/TrainingSize/rf_postgresql_runtime_200combos_trainRatio'$trainSplit'_2.pk' --TrainValTest_split='('$trainSplit',0,'$testSplit')' --timing_info_outfile=$garage'TimingAnalysis/TrainingSize/rf_postgresql_runtime_200combos_trainRatio'$trainSplit'_2.txt';
#done


garage='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/'
for x in 1 3 5 7 9; do
	trainSplit=0.$x
	testSplit=0.$(( 10-$x ))
	sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python rf_postgresql.py --save_model=True --evaluation=True --dataset_dir=$garage'dataset/postgresTemplates/train_200_combos.csv' --current_target=runtime --num_tree_estimators=200 --max_depth=20 --treatmentTraining=False --outdir=$garage'TimingAnalysis/TrainingSize/rf_postgresql_runtime_200combos_trainRatio'$trainSplit'_3.pk' --TrainValTest_split='('$trainSplit',0,'$testSplit')' --timing_info_outfile=$garage'TimingAnalysis/TrainingSize/rf_postgresql_runtime_200combos_trainRatio'$trainSplit'_3.txt';
done


