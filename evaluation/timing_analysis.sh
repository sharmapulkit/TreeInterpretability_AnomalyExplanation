garage='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/'
for x in 1 3 5 7 9; do
	trainSplit=0.$x
	testSplit=0.$(( 10-$x ))
	sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python rf_postgresql.py --save_model=True --evaluation=True --dataset_dir=$garage'dataset/postgresTemplates/train_200_combos.csv' --current_target=runtime --num_tree_estimators=200 --max_depth=20 --treatmentTraining=False --outdir=$garage'TimingAnalysis/TrainingSize/rf_postgresql_runtime_200combos_trainRatio'$trainSplit'_3.pk' --TrainValTest_split='('$trainSplit',0,'$testSplit')' --timing_info_outfile=$garage'TimingAnalysis/TrainingSize/rf_postgresql_runtime_200combos_trainRatio'$trainSplit'_3.txt';
done


