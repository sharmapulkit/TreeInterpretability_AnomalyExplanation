modelPath='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/trainedNets/postgresTemplates/train_200_combos/rf_postgresql_runtime_200combos.pk'
dataPath='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/postgresTemplates/'
#srun python postGresRandomBaselineRankCorrelation.py --model_dir=$modelPath --test_dir=$dataPath'interpretations_2/' --train_dir=$dataPath'Subset/Train_subset' --outdir=$dataPath'diffInterpretations_templates_randomBaseline/'
sbatch --partition=defq --job-name=job1 ~/nrun_inf.sh python postGresRandomBaselineRankCorrelation.py --model_dir=$modelPath --test_dir=$dataPath'interpretations_2/' --train_dir=$dataPath'Subset/Train_subset' --outdir=$dataPath'diffInterpretations_templates_randomBaseline/'
