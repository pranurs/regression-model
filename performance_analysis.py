import numpy as np
import matplotlib.pyplot as plt


class color :

    '''
    To modify display of the rendered results.
    '''

    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def calc_RMSE (actual, predicted):

    '''
    To calculate the Root Mean Squared Error for the model prediction.

     Arguments
    -----------

    actual : the truth values of target attribute altitude
    predicted : the predicted values for target variable altitude

     Returns
    ---------

    RMSE : the root mean squared error 
    '''
    
    size = actual.shape[0]
    diff = actual - predicted
    RMSE = np.sqrt(np.sum(diff**2)/(size))
    
    return RMSE


def calc_R2_score (actual, predicted) :
    
    '''
    To calculate the R^2 Score for the model prediction.
    More information can be found at https://en.wikipedia.org/wiki/Coefficient_of_determination

     Arguments
    -----------

    actual : the truth values of target attribute altitude
    predicted : the predicted values for target variable altitude

     Returns
    ---------

    r_squared_score : the R squared score 
    '''
    
    diff = actual - predicted
    diff = diff**2
    
    residual_variance = np.sum(diff)
    
    mean_actual = np.sum(actual)/(actual.shape[0])
    diff_from_mean = actual - mean_actual
    diff_from_mean = diff_from_mean**2

    total_variance = np.sum(diff_from_mean)
    
    r_squared_score = 1 - (residual_variance/total_variance)
    
    # print('mean actual',mean_actual)
    # print('residual var',residual_variance)
    # print(' total var', total_variance)
      
    
    return r_squared_score

def get_performance_info (actual, predicted, degree, theta, trainingTechnique): 

    '''
    To display the performance information (R2 Score, RMSE, and weights generated) for 
    a candidate regression model.

     Arguments
    -----------

    actual : the truth values of target attribute altitude
    predicted : the predicted values for target variable altitude
    degree : the degree of polynomial fitted
    theta : the weights of the model for which performance is being analysed
    trainingTechnique : the technique (Batch Gradient Descent, Stochastic Gradient Descent, or Normal Equation Method)
                        used to estimate the weights of the model of interest.
    '''
    
    print("\n\n\n" + color.BOLD + color.UNDERLINE + trainingTechnique + ":"+color.END + "\n")
    print("R2-Score: ", calc_R2_score(actual, predicted))
    print("RMSE: ", calc_RMSE(actual, predicted))
    print("Weights Generated for Degree {}: {}".format(degree, ', '.join(str(weight) for weight in theta)))
    print("\n\n\n")

def plot_cost_vs_iterations (cost, interval) :

    '''
    Plots the halved mean square error cost function used in gradient descent for every (interval) number
    of iterations of the algorithm, using the library matplotlib.pyplot.

     Arguments
    -----------

    cost : a Python list containing the halved mean squared errors as observed in each iteration of gradient descent.
    interval : specifies the interval between successive plotting of cost values

    '''

    plt.figure()
    x = np.arange(1,len(cost)+1,interval)
    y = [cost[i-1] for i in x]
    plt.plot(x,y)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost (Halved Mean Square Error)')
    plt.title('Plot of Cost vs Iterations (every {} iterations)'.format(interval))
    # plt.savefig('cost_vs_iter_{}.png'.format(interval))


