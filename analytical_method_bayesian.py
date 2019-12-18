
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.special



def beta (a, b):
    '''
    Utility function to return the value of the beta function for given positive inputs a and b.
    Calculated using the formula:
    
    Beta (a,b) = gamma(a) * gamma(b) / gamma(a+b)

    where, for the gamma function, we make use of the inbuilt gamma function in the scipy.special package. 

     Arguments
    -----------
    a, b : positive inputs to the beta function

     Returns
    ---------
    the value of the beta function for passed parameters a and b
    '''
    
    const = ((scipy.special.gamma(a) * scipy.special.gamma(b))) / scipy.special.gamma(a+b)
    
    return const


def pdf_beta (mu, a, b) :
    '''
    Utility function to return the beta probability distribution given the arguments explained below.
    The formula used for its calculation is:

    (gamma(a+b)/gamma(a)*gamma(b)) * ((mu ^ (a-1)) * ((1-mu) ^(b-1)))

    where gamma(a+b)/gamma(a)*gamma(b) = 1/beta(a,b)

     Arguments
    -----------
    mu : numpy array of possible values of a random variable of the beta probability distribution, for which we want to calculate the probability distribution.
    a, b : characteristic parameters for the beta probability distribution function. The mean of the beta distribution is a/(a+b).
    '''
    
    const = 1 / beta(a,b)
    pdf = const * np.multiply((mu ** (a-1)), ((1-mu) ** (b-1)))
                              
    return pdf


def initialize_data (X_num, X_mean, mu_num, a, b):

    '''
    Constructs a dataset such that the specifications passed as arguments are met.

    We assume that our prior distribution for the mean 'mu' of a Bernoulli distribution is a beta distribution (parameters: a and b), 
    as it is the conjugate prior for a Bernoulli distribution. 
    
    'X' ​ is a random variable of the Bernoulli probability distribution, which can take values 1 or 0 (head or tail). 
    We construct a dataset in which the expected value for ​ 'X' ​ being 1 is 'X_mean'. We have 'X_num' such values in our dataset. 
    Thus, we accordingly adjust the number of ones and zeros in the dataset to meet these specifications.

    'mu' (the probability of getting a head when a coin is tossed), can hence take values from 0 to 1. 
    We take 'mu_num' values between 0 and 1 to model the possibilities of 'mu'. 

     Arguments
    -----------
    X_num : the number of values in our dataset, representing a random variable of the Bernoulli probability distribution.
    X_mean : the desired expectation for random variable 'X' taking the value 1
    mu_num : the number of values taken at equal intervals between 0 and 1 to model the possibilities of 'mu', the probability of getting a head when a coin is tossed.
    a, b : the parameters of the beta distribution, which we take as the prior distribution.

     Returns
    ---------
    X : numpy array of data points (1 or 0) representing a random variable of the Bernoulli probability distribution, constructed according to the desired specifications (as explained above).
    mu : numpy array of 'mu_num' values between 0 and 1 modelling the possibilities of 'mu', a random variable representing the probability of getting a head when a coin is tossed.
    a, b : the parameters of the beta distribution, which we take as the prior distribution. Returned for the sake of clarity in concept, these are also parameters set by the user. 

    '''
    
    mu = np.linspace(0,1,mu_num)
    X = np.zeros((X_num,1))
    X[0 : int(X_mean * X_num), :] = 1
    np.random.shuffle(X)
    
    return X, mu, a, b



def analytical_method_seq (X, mu, a, b):
    '''
    This function uses the analytical method for sequential estimation of the posterior probability distribution of the mean of a
    Bernoulli distribution.
    The likelihood function in case of considering data points sequentially, is a binomial distribution with
    parameter ​ N ​ (number of Bernoulli trials) as 1. This is equivalent to a Bernoulli pdf for each data point.

    In the analytical method, it can be shown that the posterior probability after observation of each data point sequentially, is a 
    beta distribution with parameters as follows : 

        a = prev_a + data_point ​ 
        b = prev_b + 1 - data_point ​ 
        
    (at each stage).

     Arguments
    -----------
    X : numpy array of data points (1 or 0) representing a random variable of the Bernoulli probability distribution.
    mu : numpy array of values between 0 and 1 modelling the possibilities of 'mu', a random variable representing the probability of getting a head when a coin is tossed.
    a, b : parameters for the beta probability distribution, which we take as the prior distribution.

     Returns
    ---------
    posteriors : a Python list of numpy arrays, each representing the estimated posterior probability distribution, after the observation of each data point.
    means : a Python list of mean values of the posterior probability distribution estimated after the observation of each data point. Mean for a beta distribution of parameters a and b is calculated as a/(a+b).  

    '''
    
    posteriors = []
    means = []
    posterior = np.zeros (mu.shape)
    prev_a = a
    prev_b = b
    
    for data_point in X:
        
        new_a = prev_a + data_point
        new_b = prev_b + 1 - data_point
        mean = new_a / (new_a + new_b)
        
        posterior[:] = pdf_beta(mu, new_a, new_b)
        
        posteriors.append(posterior.copy())
        means.append(mean)
        
        prev_a = new_a
        prev_b = new_b
        
    return posteriors, means


def plot_sequential_posteriors (mu, posteriors, means = None, method = 'Numerical'):
    '''
    Plots and saves the estimated posterior probability distributions for the mean of the Bernoulli distribution, after the observation 
    of each data point, as per the sequential method for estimation.

     Arguments
    -----------
    mu : the random variable assumed to have a prior probability distrubution as beta, taking values between 0 and 1, for which we have estimated the posterior probability distribution. Occupies the x-axis on the plot.
    posteriors : a Python list of numpy arrays, each representing the estimated posterior probability distribution, after the observation of each data point. Each such array occupies the y-axis for its corresponding plot.
    means : a Python list of mean values of the posterior probability distribution estimated after the observation of each data point. Optional parameter. Defaults to None.
    method : specifies the technique used for estimation. Can take values 'Numerical' or 'Analytical'. Optional parameter. Defaults to 'Numerical'.

    '''
    
    
    save_figs_in  = "Sequential Posterior Distributions/" + method + "/"
    N = len(posteriors)
    
    for i in range(0,N):
        plt.figure()
        plt.plot(mu,posteriors[i])
        plt.ylim(0,12)
        plt.xlabel('mu (Mean of Bernoulli Distribution)')
        plt.ylabel('Beta(mu)')
        
        if method == 'Analytical' :
            plt.annotate ('Mean = %.5f'%means[i], (0.75, 10))
            
        plt.title('Sequential variation of Posterior Probability Distribution')
        digits = len(str(i))
        add = ''.join('0' for i in range(3-digits))
        # plt.savefig(save_figs_in+'Fig_{}{}'.format(add, i))



def analytical_method_entire (X, mu, a, b):
    '''
    This function uses the analytical method for estimating the posterior probability distribution of the mean of a Bernoulli 
    probability distribution, in one go, i.e., having observed the entire dataset at once.

    The likelihood function in case of considering all data points at once, is a binomial distribution with N as
    the number of data points, and m as the number of ones in the data. 

    In the analytical method, it can be shown that the posterior probability after observation the entire dataset, is a
    beta distribution with parameters as follows:
        
        a = prior_a + m ​ 
        b = prior_b + N - m 
        
     Arguments
    -----------
    X : numpy array of data points (1 or 0) representing a random variable of the Bernoulli probability distribution.
    mu : numpy array of values between 0 and 1 modelling the possibilities of 'mu', a random variable representing the probability of getting a head when a coin is tossed. 
    a, b : parameters for the beta probability distribution, which we take as the prior distribution.

     Returns
    ---------
    posterior_final : a numpy array representing the estimated posterior probability distribution for the mean, after the observation of the entire dataset at once.
    mean : the mean of the final estimated posterior probability distribution. If the beta distribution has parameters a and b, the mean is calculated as a/(a+b). 
    '''
    
    N = X.shape[0]
    m = sum(X)
    new_a = a + m
    new_b = b + N - m
    mean = (new_a) / (new_a + new_b)
    
    posterior_final = pdf_beta(mu, a + m, b + N - m)
    print(pdf_beta(mu, a + m, b + N - m).shape)
    return posterior_final, mean



def plot_final_posterior (mu, posterior_final, mean = None, method = 'Numerical'):
    '''
    Plots and saves the estimated posterior probability distribution for the mean of the Bernoulli distribution, after 
    observation of the entire dataset in a go (at once).

     Arguments
    -----------
    mu : the random variable assumed to have a prior probability distrubution as beta, taking values between 0 and 1, for which we have estimated the posterior probability distribution. Occupies the x-axis on the plot.
    posterior_final : a numpy array representing the estimated posterior probability distribution for the mean, after the observation of the entire dataset at once. Occupies the y-axis for its corresponding plot.
    mean : The mean value of the posterior probability distribution estimated after the observation of all data points at once. Optional parameter. Defaults to None.
    method : specifies the technique used for estimation. Can take values 'Numerical' or 'Analytical'. Optional parameter. Defaults to 'Numerical'.
    '''

    save_figs_in  = "Final Posterior Distributions/" + method + "/"
    
    plt.figure()
    plt.plot(mu, posterior_final)
    plt.xlabel('mu (Mean of Bernoulli Distribution)')
    plt.ylabel('Beta(mu)')
    
    if method == 'Analytical' :
        plt.annotate ('Mean = %.5f'%mean, (0.75, 10))
            
    plt.suptitle('Posterior Probability Distribution Calculated for Entire Dataset (at Once)', y = 1)
    plt.title(method + ' Method', fontsize = 14)
    # plt.savefig(save_figs_in+'Fig_Entire_Dataset_'+method)

def main():
    '''
    Driver function.
    '''

    # Construct dataset with 160 data points (1 or 0), having mean 0.7, with 10000 equally spaced values between 0 and 1
    # to model the possibilities for the mean, and with parameters 2 and 3 for the prior beta distribution for the mean.

    X, mu, a, b = initialize_data (160, 0.7, 10000, 2, 3)

    posteriors_seq_anal, means = analytical_method_seq (X, mu, a, b)
    plot_sequential_posteriors(mu, posteriors_seq_anal, means, 'Analytical')

    posterior_final_anal, mean = analytical_method_entire (X, mu, a, b)
    plot_final_posterior (mu, posterior_final_anal, mean, 'Analytical')


if __name__ == '__main__':
    
    main()




