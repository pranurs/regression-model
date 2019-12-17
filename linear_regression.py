import pandas as pd
import numpy as np

from tqdm import tqdm, tqdm_notebook

class Regression_Model :

	'''
	A class used to represent a polynomial regression model in two variables.

	~~~~~~~~~~~~~~~

	 Attributes
	------------

	featureMatrix : the matrix containing features as columns according to the degree of polynomial (in two variables) that must be fitted to the data
	
	targetAttribute : the target values for altitude that we want to predict
	
	initialWeights : may be passed by the user as pre-trained weights. Optional parameter, defaults to None.
	
	lambdaReg : regularization parameter. Defaults to 0.
	
	regularizationType : specifies the type of regularization. Type may be 'L1' , 'L2' , or 'None'. Defaults to 'None'.
	
	learningRate : initial learning rate for use in batch gradient descent. Defaults to 0.01.
	
	epochs : number of times over the whole training set to run stochastic gradient descent. Defaults to 2.
	
	isBatch : takes a value 1, if the object uses batch gradient descent for weight estimation, and 0 if it uses stochastic gradient descent.
				Irrelevent in the case of weight estimation using normal equation. Defaults to 1.
	
	maxTolerance : the maximum tolerance value for difference in the cost of the current iteration and previous iteration in batch gradient descent.
					Used for conditional termination of the batch gradient descent algorithm. Defaults to 1e-10.
	
	maxIter : the maximum number of iterations for which the batch gradient descent algorithm should run. Used to force termination. Defaults to 10000.


	 Methods
	---------

	hypothesis_function (X, theta) : returns prediction, for specified weight matrix and feature matrix

	cost_function () : returns halved mean squared error

	update_weights (X_i, Y_i) : updates the attribute theta of the object according to the appropriate weight update rule

	batch_gradient_descent () : performs batch gradient descent on the object's attributes

	stochastic_gradient_descent () : performs stochastic gradient descent on the object's attributes

	normal_equation () : estimates object attribute theta using an analytical method

	'''

	def __init__ (self, featureMatrix, targetAttribute, initialWeights = None, lambdaReg = 0, regularizationType = 'None', learningRate = 0.01, epochs = 2, isBatch = 1, maxTolerance = 1e-10, maxIters = 10000) :

		'''
		Initializes the object attributes with user specified values (or) default values where applicable.
		Refer class description for an elaboration of the object attributes.


		 Necessary Parameters
		----------------------

		featureMatrix : ndarray
		targetAttribute : ndarray


		 Optional Parameters
		---------------------

		initialWeights : ndarray, defaults to None
		lambdaReg : float, defaults to 0
		regularizationType : str, defaults to 'None'
		learningRate : float, defaults to 0.01
		epochs : int, defaults to 2
		isBatch : int, defaults to 1
		maxTolerance : float, defaults to 1e-10
		maxIters : int, defaults to 10000

		'''
		self.X = featureMatrix
		self.yTrain = targetAttribute
		self.theta = initialWeights
		self.lambdaReg = lambdaReg
		self.regularizationType = regularizationType
		self.alpha = learningRate
		self.epochs = epochs
		self.isBatch = isBatch
		self.maxTolerance = maxTolerance
		self.maxIters = maxIters



	def hypothesis_function (self, X, theta) :

		'''
		Returns the predicted value of target variable altitude, using the weights of the model.
		It uses the formula Y_predicted = featureMatrix * theta

		 Arguments
		----------

		X : feature matrix
		theta : weights of model

		 Returns 
		---------

		The predicted values for the target variable altitude
		'''

		return X.dot(theta)



	def cost_function (self) :

		'''
		Returns the mean square error (halved), using the feature matrix, weights and training labels.
		This function handles the case for no regularization, L1 regularization and L2 regularization.
		Defaults to no regularization.

		 Returns
		---------
		cost : the halved mean square error for predicting the target attribute by using the weights associated with this object.
		
		'''

		diff = self.hypothesis_function(self.X, self.theta) - self.yTrain
		m = self.X.shape[0]
		cost = np.sum(diff**2)/(2*m)

		if self.regularizationType == 'L2' :

			cost += (lambda_reg/2) * np.sum(theta**2)

		if self.regularizationType == 'L1' :

			cost += (lambda_reg) * np.sum(np.abs(theta))

		return cost




	def update_weights (self,X_i = None, Y_i = None) :

		'''
		Applies the appropriate weight update rule to the theta attribute associated with the instance.
	
		This function handles the case for no regularization, L1 regularization and L2 regularization,
		for both batch gradient descent and stochastic gradient descent.
		
		Defaults to no regularization.

		 Arguments
		-----------

		X_i : The features for one training data point, as needed for stochastic gradient descent. Defaults to None.
		Y_i : The target value for one training data point, as needed for stochastic gradient descent. Defaults to None.

		'''


		if self.isBatch :
		    
		    diff = self.hypothesis_function (self.X, self.theta) - self.yTrain
		    m = self.X.shape[0]

		    if self.regularizationType == 'L2' :

		        self.theta *= (1 - self.alpha * self.lambdaReg)
		        self.theta -= ((self.alpha/m)*(np.dot((self.X).T,diff)))

		    elif self.regularizationType == 'L1' :

		        temp_theta = np.copy(self.theta)
		        temp_theta[temp_theta > 0] = 1
		        temp_theta[temp_theta < 0] = -1

		        self.theta -= (self.alpha * self.lambdaReg * temp_theta)
		        self.theta -= ((self.alpha/m)*(np.dot((self.X).T,diff)))

		    else :

		        self.theta -= ((self.alpha/m)*(np.dot((self.X).T,diff)))


		        
		else:
		    
		    diff = np.sum(np.multiply(self.theta, X_i)) - Y_i
		    
		    if self.regularizationType == 'L2' :
		        
		        self.theta *= (1 - self.alpha * self.lambdaReg)
		        self.theta -= (self.alpha * diff * X_i)
		        
		    elif self.regularizationType == 'L1' :
		        
		        temp_theta = np.copy(self.theta)
		        temp_theta[temp_theta > 0] = 1
		        temp_theta[temp_theta < 0] = -1

		        self.theta -= (self.alpha * self.lambdaReg * temp_theta)
		        self.theta -= (self.alpha * diff * X_i)
		        
		    else :
		        
		        self.theta -= (self.alpha * diff * X_i)
		    



   
	def batch_gradient_descent (self) :

		'''
		Performs batch gradient descent for fitting the polynomial of a desired degree (as appropriated by the feature matrix),
		to the data.

		If no initial weights are specified, it initializes the weights to random values.

		The maximum number of iterations (maxIter) and maximum acceptable tolerance level (maxTolerance) for decrease in 
		mean square error (halved) in subsequent weight updates, jointly act as the stopping critera for batch gradient descent.
		This means that the algorithm will stop either if the number of iterations has reached the maximum limit, or sooner, if 
		the change in cost for subsequent iterations is less than the maximum accepted tolerance.

		We also employ the 'Bold Driver Technique' for dynamic learning rate adaptation.
		This works as follows:
		If the cost in the current iteration has decreased as compared to the cost in the previous iteration, we increase alpha
		by 5 % of its current value.
		However, if the cost in the current iteration is found to have increased, it means that the algorithm has overshot the minima.
		In this event, we reset the weights to their previous values before the update, and penalize alpha by decreasing it to 50 % of 
		its current value.

		Thus, as alpha is adjusted accordingly, it eliminates the need to optimize this hyperparameter by trying different values randomly.
		It also makes the descent faster, as alpha is dynamically adjusted in order for it to speed up.

		Batch gradient descent may be performed with L1, L2 or no regularization. By default, we use no regularization.

		The lambda parameter for L1 and L2 regularization, however, may be determined by running the model for different values
		of lambdaReg.

		No weights are returned, as the weights are changed in the theta attribute of the object, and may be consequently accessed. 


		 Returns 
		---------

		cost : A list with the halved mean squared errors as calculated for each iteration throughout the course of gradient descent.

		alphas :  A list of learning rates following the bold driver technique used during gradient descent, for the purpose of analysis.
	
		'''

		if not self.theta:
			
			self.theta = np.random.rand (self.X.shape[1],1)

		m = self.X.shape[0]

		prev_theta = np.copy (self.theta)


		cost = []
		alphas = []

		for i in tqdm(range(1, self.maxIters+1)):

			prev_theta[:] = self.theta

			self.update_weights()

			cost.append(self.cost_function())
			# print(cost[-1])

			# print('\nalpha',self.alpha)
			alphas.append(self.alpha)

			# To change alpha or reset weights according to bold driver method for dynamic learning rate adaptation

			if len(cost)>1 and cost[-1]-cost[-2] < 0:
			    
			    self.alpha*=1.05
			    
			elif len(cost)>1 and cost[-1]-cost[-2]>0:

			    # print('\n\nbefore', self.theta)
			    self.theta[:] = prev_theta
			    # print('\n\nafter', self.theta)

			    self.alpha*=0.5

			    del alphas[-1]
			    del cost[-1]
			    
			    continue
			    
			# Termination condition for gradient descent

			if len(cost)>1 and (cost[-2]-cost[-1]) <= self.maxTolerance:

			    break

		return cost, alphas
    


 
	def stochastic_gradient_descent (self) :

		'''
		Performs stochastic gradient descent on the object's attributes and updates its theta attribute accordingly.

		If no initial weights are specified, it initializes the weights to random values.

		Stochastic gradient descent runs over the entire set of training data points as many times as specified by the
		object attribute epochs.

		Does not return weights, as they are updated in, and may be consequently accessed by, the object parameter theta.

		If the cost for each iteration is not required to be calculated, it may be avoided for faster execution of the descent.

		 Returns
		---------
		cost : A list with the halved mean squared errors as calculated for each iteration throughout the course of stochastic gradient descent.

		'''


		self.isBatch = 0
		if not self.theta :
			self.theta = np.random.rand(self.X.shape[1],1)

		m = self.X.shape[0]

		cost = []

		for epoch in range(self.epochs): 
	        
			for i in tqdm(range(m)):

				X_i = np.reshape(self.X[i],(self.X[i].shape[0],1))
				Y_i = self.yTrain[i]

				self.update_weights (X_i, Y_i)

				### Comment the line below for calculating cost, if not required, for fastest completion of stochastic gradient descent.

				cost.append(self.cost_function())

		return cost




	def normal_equation (self):

	    '''
		Estimates the weights for fitting the polynomial of the desired degree, using an analytical method.

		The formula used is :

		theta = inverse (X_transpose * X)  *  (X_transpose * Y)

		where X is the feature matrix, and Y represents the target attribute

		This method involves no training of the parameters theta, and is a direct formula-based method.
	    '''

	    self.theta = np.matmul(np.linalg.pinv(np.matmul(self.X.T,self.X)),np.matmul(self.X.T,self.yTrain))
	    
            