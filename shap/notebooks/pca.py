from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from dataLoader import *
from postgresql_dataConfig import *

def main(n_components, datasetPath, outfile):
	pca = PCA(n_components)

	dataloader = DataLoader(datasetPath, covariate_columns, treatment_columns, target_columns)
	Xtr, logYtr, _, _, _, _ = dataloader.preprocessData(train_frac=1.0, val_frac=0.0, test_frac=0.0)

	Xtr = Xtr.loc[:, feature_columns].values
	Xtr = StandardScaler().fit_transform(Xtr)
	principalComponents = pca.fit_transform(Xtr)
	print("Expained Var:", pca.explained_variance_ratio_)

	principalComponentsDF = pd.DataFrame(principalComponents)
	principalComponentsDF.to_csv(outfile)

	return

if __name__=="__main__":
	datasetPath = '/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/reClean/another_case/trainPostgres_0.7.csv'
	outfile = '/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/reClean/another_case/trainPostgres_pca_0.7_components13.csv'
	main(13, datasetPath, outfile)
