#python RF_postgreSQL.py --save_model=True --dataset_dir='/home/pulkit/Documents/PDF/umass/sem2/cs696ds/TreeInterpretability_AnomalyExplanation/' --current_target=runtime --num_tree_estimators=200 --max_depth=20 --outdir='/home/pulkit/Documents/PDF/umass/sem2/cs696ds/TreeInterpretability_AnomalyExplanation/shap/notebooks/trainedModels/' --treatmentTraining=True --treatment_combination='(0,2,0)' --timing_info_outfile='/home/pulkit/Documents/PDF/umass/sem2/cs696ds/TreeInterpretability_AnomalyExplanation/shap/notebooks/trainedModels' --TrainValTest_split='(0.1,0.1,0.1)'

#sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python interpret.py \
#--dataset_dir='/home/s20psharma/cs696ds/TreeInterpretability_AnomalyExplanation/datasets/postgres-results.csv' \
#--model_dir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/trainedNets/RF_postgres_Nest200_maxD20_runtime_tr212' \
#--outdir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/trainedNets/interpretedOuts.txt' \
#--TrainValTest_split='(0.6,0.2,0.2)' \
#--datapoint_start=0 \
#--datapoint_end=500

# sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python interpret.py --dataset_dir='/home/s20psharma/cs696ds/TreeInterpretability_AnomalyExplanation/datasets/postgres-results.csv' --model_dir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/trainedNets/RF_postgres_Nest200_maxD20_runtime_tr212' --outdir_ti_contribs='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/trainedNets/interpreted_TI_outs.txt' --outdir_shap_contribs='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/trainedNets/interpreted_SHAP_outs.txt' --TrainValTest_split='(0.6,0.2,0.2)' --datapoint_start=0 --datapoint_end=500

#chunkSize=10000
#for (( testDataStart=0; testDataStart < 300000 ; testDataStart+=$chunkSize )); do
#	sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python interpret.py --dataset_dir='/home/s20psharma/cs696ds/TreeInterpretability_AnomalyExplanation/datasets/postgres-results.csv' --model_dir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/trainedNets/RF_postgres_Nest200_maxD20_runtime_tr011' --outdir_ti_contribs='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/trainedNets/interpreted_TI_outs_'$testDataStart'.txt' --outdir_shap_contribs='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/trainedNets/interpreted_SHAP_outs_'$testDataStart'.txt' --TrainValTest_split='(0.6,0.2,0.2)' --datapoint_start=$testDataStart --datapoint_end=$(($testDataStart+$chunkSize))
#done

#chunkSize=200
#for (( testDataStart=0; testDataStart < 10000 ; testDataStart+=$chunkSize )); do
#	sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python interpret.py --dataset_dir='/home/s20psharma/cs696ds/TreeInterpretability_AnomalyExplanation/datasets/postgres-results.csv' --model_dir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/trainedNets/average_model/RF_postgres_Nest200_maxD20_runtime.pk' --outdir_ti_contribs='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/trainedNets/average_model/interpretations/interpreted_TI_avg_outs_'$testDataStart'.txt' --outdir_shap_contribs='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/trainedNets/average_model/interpretations/interpreted_SHAP_avg_outs_'$testDataStart'.txt' --TrainValTest_split='(0.6,0.2,0.2)' --datapoint_start=$testDataStart --datapoint_end=$(($testDataStart+$chunkSize))
#done

######## Interpretation over complete Testing Set ##########
#dataPath='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/postgresTemplates/Subset/Test_subset/'
#modelPath='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/trainedNets/postgresTemplates/train_200_combos/rf_postgresql_runtime_200combos.pk'
#iter=0
#for filepath in $dataPath*.csv; do
#	file=$(basename $filepath)
#	iter=$(($iter+1))
#	sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python interpret.py --dataset_dir=$dataPath$file --model_dir=$modelPath --outdir_ti_contribs=$dataPath"../../interpretations_shuffleFixed/interpreted_TI_outs_"${file:5:-4}".txt" --outdir_shap_contribs=$dataPath'../../interpretations_shuffleFixed/interpreted_SHAP_outs_'"${file:5:-4}"'.txt' --TrainValTest_split='(0.0,0.0,1.0)';
#done

#### Interpret 500 points for time profiling
#dataPath='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/postgresTemplates/'
#modelPath='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/trainedNets/postgresTemplates/train_200_combos/rf_postgresql_runtime_200combos.pk'
#file='some_700_train_points.csv'
#sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python interpret.py --dataset_dir=$dataPath$file --model_dir=$modelPath --outdir_ti_contribs=$dataPath"../../_interpreted_TI_outs_"${file:5:-4}".txt" --outdir_shap_contribs=$dataPath'../../_interpreted_SHAP_outs_'"${file:5:-4}"'.txt' --TrainValTest_split='(0.0,0.0,1.0)'

#### Interpret N points for time profiling
#dataPath='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/postgresTemplates/TimingAnalysis/Depth/'
#modelDir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/TimingAnalysis/Depth/'
#for depth in 5 10 15 20 25 30 35 40 45 50 55; do
#		modelPath=$modelDir'rf_postgresql_runtime_200combos_MaxDepth'$depth'.pk'
#		for filenumber in 50 100 200 300 500 700 900; do
#			file='some_'$filenumber'_train_points.csv'
#			sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python interpret.py --dataset_dir=$dataPath$file --model_dir=$modelPath --outdir_ti_contribs=$dataPath"TI_contribs/_interpreted_TI_outs_MaxDepth"$depth"_numFiles"$filenumber".txt" --outdir_shap_contribs=$dataPath'SHAP_contribs/_interpreted_SHAP_outs_MaxDepth'$depth"_numFiles"$filenumber'.txt' --TrainValTest_split='(0.0,0.0,1.0)' --TimingOutFile=$dataPath'Runtime/runtime_MaxDepth'$depth'_NumDataPoints'$filenumber;
#		done;
#done

#### Interpret N points for time profiling
#dataPath='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/postgresTemplates/TimingAnalysis/TrainingSize/'
#modelDir='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/TimingAnalysis/TrainingSize/'
#for x in 1 3 5 7 9; do
#	trainSplit=0.$x
#	testSplit=0.$(( 10-$x ))
#	modelPath=$modelDir'rf_postgresql_runtime_200combos_trainRatio'$trainSplit'.pk'
#	sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python interpret.py --dataset_dir=$dataPath'../Depth/some_700_train_points.csv' --model_dir=$modelPath --outdir_ti_contribs=$dataPath"TI_contribs/_interpreted_TI_outs_trainRatio"$trainSplit"_numFiles900.txt" --outdir_shap_contribs=$dataPath'SHAP_contribs/_interpreted_SHAP_outs_trainRatio'$trainSplit"_numFiles900.txt" --TrainValTest_split='(0.0,0.0,1.0)' --TimingOutFile=$dataPath'Runtime/runtime_trainSplit'$trainSplit;
#done


#### Interpret 500 points for Accuracy computation
#dataPath='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/topFeatAccuracy/memory_level/'
#modelPath='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/trainedNets/postgresTemplates/train_200_combos/rf_postgresql_runtime_200combos.pk'
#file='some_500_test_points_Baseline.csv'
#sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python interpret.py --dataset_dir=$dataPath$file --model_dir=$modelPath --outdir_ti_contribs=$dataPath"_interpreted_TI_outs_500_memorylevel_increased2.txt" --outdir_shap_contribs=$dataPath'_interpreted_SHAP_outs_500_memorylevel_increased2.txt' --TrainValTest_split='(0.0,0.0,1.0)'


#### Interpret 500 points for Accuracy computation
#dataPath='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/topFeatAccuracy/'
#for x in 1 3 5 7 9; do
#	modelPath='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/TimingAnalysis/TrainingSize/rf_postgresql_runtime_200combos_trainRatio0.'$x'.pk'
#	for tr in 'indexlevel' 'pagecost' 'memorylevel'; do
#		file=$tr'/some_500_test_points_'$tr'_increased2.csv'
#		sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python interpret.py --dataset_dir=$dataPath$file --model_dir=$modelPath --outdir_ti_contribs=$dataPath$tr'/_interpreted_TI_outs_500_trainRatio0.'$x'_'$tr'_increased2.txt' --outdir_shap_contribs=$dataPath$tr'/_interpreted_SHAP_outs_500_trainRatio0.'$x'_'$tr'_increased2.txt' --TrainValTest_split='(0.0,0.0,1.0)' --TimingOutFile=$dataPath$tr'/runtime_500_outs_trainRatio0.'$x'_'$tr'_increased2.txt'; 
#	done;
#done

#dataPath='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/topFeatAccuracy/'
#for x in 1 3 5 7 9; do
#	modelPath='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/TimingAnalysis/TrainingSize/rf_postgresql_runtime_200combos_trainRatio0.'$x'.pk'
#	file='some_500_test_points_Baseline.csv'
#	sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python interpret.py --dataset_dir=$dataPath$file --model_dir=$modelPath --outdir_ti_contribs=$dataPath'_interpreted_TI_outs_500_trainRatio0.'$x'_Baseline.txt' --outdir_shap_contribs=$dataPath$tr'_interpreted_SHAP_outs_500_trainRatio0.'$x'_Baseline.txt' --TrainValTest_split='(0.0,0.0,1.0)' --TimingOutFile=$dataPath$tr'runtime_500_outs_trainRatio0.'$x'_Baseline.txt'; 
#done

######## Interpretation over Testing Set with flipped bits ##########
#dataPath='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/postgresTemplates/Subset/Test_subset/'
#for x in 1 3 5 7 9; do
#	modelPath='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/TimingAnalysis/TrainingSize/rf_postgresql_runtime_200combos_trainRatio0.'$x'.pk'
#	for tr in 'index_level' 'page_cost' 'memory_level'; do
#		dataPath='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/topFeatAccuracy/testCases/'$tr'/testData/'
#		iter=0
#		for filepath in $dataPath*.csv; do
#			file=$(basename $filepath)
#			iter=$(($iter+1))
#			sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python interpret.py --dataset_dir=$dataPath$file --model_dir=$modelPath --outdir_ti_contribs=$dataPath'../interpreted_TI_outs_'${file:5:-4}'_'$tr'_trainRatio0.'$x'_increased2.txt' --outdir_shap_contribs=$dataPath'../interpreted_SHAP_outs_'"${file:5:-4}"'_'$tr'_trainRatio0.'$x'_increased2.txt' --TrainValTest_split='(0.0,0.0,1.0)' --TimingOutFile=$dataPath'../timing/runtime_outs_'"${file:5:-4}"'_'$tr'_trainRatio0.'$x'_increased2.txt';
#		done
#	done;
#done

######### Interpreatation over complete Test Set ###########
for x in 1 3 5 7 9; do
	modelPath='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/TimingAnalysis/TrainingSize/rf_postgresql_runtime_200combos_trainRatio0.'$x'.pk'
	dataPath='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/postgresTemplates/Subset/Test_subset/'
	outPath='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/topFeatAccuracy/testCases/baseline/'
	iter=0
	for filepath in $dataPath*.csv; do
		file=$(basename $filepath)
		iter=$(($iter+1))
		sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python interpret.py --dataset_dir=$dataPath$file --model_dir=$modelPath --outdir_ti_contribs=$outPath'interpreted_TI_outs_'${file:5:-4}'_trainRatio0.'$x'_baseline.txt' --outdir_shap_contribs=$outPath'interpreted_SHAP_outs_'"${file:5:-4}"'_trainRatio0.'$x'_baseline.txt' --TrainValTest_split='(0.0,0.0,1.0)' --TimingOutFile=$outPath'timing/runtime_outs_'"${file:5:-4}"'_trainRatio0.'$x'_baseline.txt';
	done;
done

