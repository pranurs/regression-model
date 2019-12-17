import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split



def normalize (x, type = 'min_max_normalization') :

	'''
		
		Performs normalization on the given vector according to the type of normalization specified.
		The type of normalization may be 'standardization' or 'min_max_normalization'.
		Defaults to 'min_max_normalization'.

		Arguments :

		x : the vector to be normalized
		type : the type of normalization to be performed (default = 'min_max_normalization')

		Returns :

		normalized vector x

	'''

	if type == 'standardization' :

		return ((x - np.mean(x)) / np.std(x))  

	else :

		return ((x - np.min(x)) / (np.max(x) - np.min(x)))



def train_test_val_split (x1, x2, Y) :

	'''

		Performs a train : test : cross-validation split in the ratio 0.8 : 0.1 : 0.1 on the given data.

		Arguments :

		x1 : vector representing latitudes in data
        x2 : vector representing longitudes in data
        Y : vector representing the target variable altitude in data

		Returns : 

		x1_Train, x1_Test, x1_Val,
		x2_Train, x2_Test, x2_Val,
		Y_Train, Y_Test, Y_Val,

		in the sizes (80%, 10%, 10%) for (Train, Test, Val) respectively as described above.


	'''
	x1 = np.reshape(x1, (x1.shape[0],1))
	x2 = np.reshape(x2, (x2.shape[0],1))
	X = np.concatenate((x1, x2), axis = 1)

	Y = np.reshape(Y, (Y.shape[0],1))

	X_Train, X_Test, Y_Train, Y_Test = train_test_split (X, Y, test_size = 0.2, random_state = 0)
	X_Test, X_Val, Y_Test, Y_Val = train_test_split (X_Test, Y_Test, test_size = 0.5, random_state = 3)

	x1_Train = np.reshape( X_Train [:,0], (X_Train [:,0].shape[0],1))
	x1_Test = np.reshape( X_Test [:,0], (X_Test [:,0].shape[0],1))
	x1_Val = np.reshape( X_Val [:,0], (X_Val [:,0].shape[0],1))

	x2_Train = np.reshape( X_Train [:,1], (X_Train [:,1].shape[0],1))
	x2_Test = np.reshape( X_Test [:,1], (X_Test [:,1].shape[0],1))
	x2_Val = np.reshape( X_Val [:,1], (X_Val [:,1].shape[0],1))

	return x1_Train, x1_Test, x1_Val, x2_Train, x2_Test, x2_Val, Y_Train, Y_Test, Y_Val

def generate_feature_matrix (x1, x2, degree) :

	'''
        Returns the feature matrix for any polynomial degree, when the number of variables is two.
        
        For example, the feature matrix constructed for fitting an 'n-degree' polynomial of two variables (x1 and x2)
        to the data would have each column as each of the features listed below:
        
        1, x1, x2, x1^2, x1*x2, x2^2, ..., x1^n, x1^(n-1)*x2, ..., x1*x2^(n-1), x2^n

        The feature matrix would look like:

        [[1   x1(1)   x2(1)   x1(1)^2   x1(1) * x2(1)   ...    x2(1)^n],
         [1   x1(2)   x2(2)   x1(2)^2   x1(2) * x2(2)   ...    x2(2)^n],
         [1   x1(3)   x2(3)   x1(3)^2   x1(3) * x2(3)   ...    x2(3)^n],
									
									...

         [1   x1(N)   x2(N)   x1(N)^2   x1(N) * x2(N)   ...    x2(N)^n]]

        
        Hence, the total number of features for a polynomial of degree D in two variables is given by :
        
        (D+1) * (D+2) / 2


        Arguments :

        x1 : vector representing latitudes in data
        x2 : vector representing longitudes in data
        degree : the desired degree of polynomial that we wish to fit to the data

        Returns :

        featureMatrix : for the specified degree, as elaborated above
         
    '''
    

	X = np.concatenate((x1,x2), axis = 1)

	featureMatrix = np.ones((X.shape[0],1))

	for d in range(1,degree+1):
	    
	    for i in range(d+1):
	        
	        feature = np.multiply((x1**(d-i)),(x2**(i)))
	        feature = np.reshape(feature,(feature.shape[0],1))

	        featureMatrix = np.concatenate((featureMatrix,feature),axis=1)
	    
	featureMatrix = np.asarray(featureMatrix)

	return featureMatrix
