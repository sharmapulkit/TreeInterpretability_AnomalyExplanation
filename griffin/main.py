
import numpy as np
from baseline import getRandomBaselineContributions

def printContributions(prediction, num, bias, featureNames, contributions):
    """
    Print sorted absolute value feature contributions
    """
    for i in range(num):
        print ("Instance", 0)
        print ("Prediction", prediction[i])
        print ("Bias (trainset mean)", bias[i])
        print ("Feature contributions:")
        for c, feature in sorted(zip(contributions[i], 
                                     boston.feature_names), 
                                 key=lambda x: -abs(x[0])):
            print (feature, round(c, 2))
        print ("-"*20)
        
def calculateFeatureDifference(baselineContribution, testContribution):
    """
    Take difference of testContribution dependending on its baseline contribution
    """
    return testContribution - baselineContribution
 

from sklearn.datasets import load_boston
boston = load_boston()
#print(boston.feature_names)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

X = boston.data
y= boston.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

rf.fit(X_train, y_train)

from treeinterpreter import treeinterpreter as ti
train_prediction, train_bias, train_contributions = ti.predict(rf, X_train)

#Write code for setting and fetching the baseline
printContributions(train_prediction, 2, train_bias, 
                   boston.feature_names, 
                   train_contributions)

#Write code for setting and fetching the baseline
baselineContribution = getRandomBaselineContributions(train_contributions)
print(baselineContribution)

test_prediction, test_bias, test_contributions = ti.predict(rf, X_test)
#print(test_contributions.shape)
#print(baselineContribution.shape)
anomalyContribution = calculateFeatureDifference(baselineContribution, test_contributions)


