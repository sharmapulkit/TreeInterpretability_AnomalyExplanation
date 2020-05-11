CODE:
Navigate to ./shap/notebooks for most of the code:

FILE DESCRIPTIONS:
Analysse_X_dataset.ipynb: Jupyter Notebook to analyse the corresponding dataset 'X'

postgresql_dataConfig.py: Configuration file for postgreSQL dataset. Declares the feature, treatment, covariate and output variables of postgreSQL dataset.

treeRegressor.py: Train a RandomForest regression model. (Implemented as a Class)
evaluateRF.py: Evaluate a trained RF model on the given testing set
rf_postgresql.py: Calles treeRegressor to train and save a RF model on postgreSQL dataset
interpret.py: Generate and Save Interpretations from TI and SHAP on the given set of data points
dataLoader.py: Dataloader utility to load, parse and organize the data into usable/trainable form
attributionAccuracy.py: Compute attribution accuracy for Interventional experiment
flipped_bits.py: Intervene on the given data and store the updated dataset. Any of the treatment variables can be changed from this script
utils.py: Some utility functions useful to print outputs
postGresRandomBaselineRankCorrelation.py: Experiment to compute contributions subtracting the median contribution


CORRESPONDING SHELL FILES:
train_rfpostgres.sh --> rf_postgresql.py
test_rfpostgres.sh --> evaluateRF.py
interpret.sh --> interpret.py
timing_analysis.sh --> rf_postgresql.py (Store the running times)
griffin_postGresRandomBaselineRankCorrelation.sh --> postGresRandomBaselineRankCorrelation.py

HOW TO:
Eg: GET INTERVENTIONAL ACCURACY:
1. Run train_rfpostgres.sh with dataset path and output path to save the trained RF model
2. Run test_rfpostgres.sh with test Dataset path to evaluate performance of trained RF model
3. Run interpret.sh to generate interpretations from SHAP and TI for given test data on the trained model. Input the output directory for TI and SHAP contributions. Provide TimingOutFile for saving the runtime logs.
4. Run flipped_bits.py to create the intervened dataset
5. Run attributionAccuracy.py to get Top-1, Top-2, Top-3 Accuracies

Note: RUN "python $script_name -h" for additional options
Note: All the paths need to be updated accordingly

SWARM commands:
1. To Run a python script: sbatch --partition=defq --job-name=jobname ~/nrun.sh python $python_script.py
2. To see job queue: squeue -u $USERNAME

