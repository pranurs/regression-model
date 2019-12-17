import pandas as pd
import numpy as np

import preprocess
from linear_regression import Regression_Model
import performance_analysis

if __name__ == '__main__':

    '''   
       Driver Function.
    '''

    data=pd.read_csv("3D_spatial_network.txt", header=None, usecols=[1,2,3])

    x1 = data.iloc[:,0]
    x2 = data.iloc[:,1]    
    Y = data.iloc[:,2]

    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    Y = np.asarray(Y)

    x1_normalized = preprocess.normalize(x1)
    x2_normalized = preprocess.normalize(x2)
    Y_normalized = preprocess.normalize(Y)

    ### Edit degree according to the degree of polynomial required to be fit to the data ###
    degree = 1

    x1_Train, x1_Test, x1_Val, x2_Train, x2_Test, x2_Val, Y_Train, Y_Test, Y_Val = preprocess.train_test_val_split(x1_normalized, x2_normalized, Y)

    featureMatrix = preprocess.generate_feature_matrix (x1_Train, x2_Train, degree)

    model = Regression_Model(featureMatrix, Y_Train)

    cost, alphas = model.batch_gradient_descent()
    # cost = model.stochastic_gradient_descent()
    # model.normal_equation()

    testFeatureMatrix = preprocess.generate_feature_matrix (x1_Test, x2_Test, degree)

    predicted = model.hypothesis_function(testFeatureMatrix, model.theta)

    performance_analysis.get_performance_info(Y_Test, predicted, degree, model.theta, 'Batch Gradient Descent')

    # performance_analysis.plot_cost_vs_iterations(cost,20)




