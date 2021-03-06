
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.special



def C(n,r):
    '''
    Utility function to return the value of nCr, which represents the number of combinations/ways of choosing r objects, given n objects.
    Calculated as factorial(n) / (factorial(r) * factorial(n-r))

     Arguments
    -----------
    n : number of objects out of which to choose from
    r : number of objects to be chosen

     Returns
    ---------
    the value of nCr as explained above

    '''
    
    return math.factorial(n)/(math.factorial(r) * math.factorial(n-r))


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
    Constructs a dataset by random sampling from a binomial distribution with parameters p = 'X_mean' and n = 'X_num'.

    We assume that our prior distribution for the mean 'mu' of a Bernoulli distribution is a beta distribution (parameters: a and b), 
    as it is the conjugate prior for a binomial distribution. 
    
    'X' ​ is an array with elements representing a random variable of the Bernoulli probability distribution, each of which can take values 1 or 0 (head or tail). 
    We construct a dataset in which the expected value for ​the elements in 'X' ​ being 1 is 'X_mean'. We have 'X_num' such elements in 'X' (the set of observed data points). 
    Thus, we accordingly adjust the number of ones and zeros in the dataset to meet these specifications, and model sampling from a binomial distribution.

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
    X : numpy array of data points (1 or 0), each representing a random variable of the Bernoulli probability distribution, constructed according to the desired specifications (as explained above).
    mu : numpy array of 'mu_num' values between 0 and 1 modelling the possibilities of 'mu', a random variable representing the probability of getting a head when a coin is tossed.
    a, b : the parameters of the beta distribution, which we take as the prior distribution. Returned for the sake of clarity in concept, these are also parameters set by the user. 

    '''
    
    mu = np.linspace(0,1,mu_num)
    X = np.zeros((X_num,1))
    X[0 : int(X_mean * X_num), :] = 1
    np.random.shuffle(X)
    
    return X, mu, a, b


def numerical_method_seq (X, mu, a, b) :
    '''
    This function uses the numerical method for sequential estimation of the posterior probability distribution of the mean of a
    Bernoulli distribution.
    The likelihood function in case of considering data points sequentially, is a binomial distribution with
    parameter ​ N ​ (number of Bernoulli trials) as 1. This is equivalent to a Bernoulli pdf for each data point.

    In the numerical method, integration is approximated by taking ​'d_mu'​ as the space between consequent values
    in 'mu'​. Integration is therefore appropriated as the summation of function values at each point in ​'mu'
    multiplied by 'd_mu'.

     Arguments
    -----------
    X : numpy array of data points (1 or 0), each representing a random variable of the Bernoulli probability distribution.
    mu : numpy array of values between 0 and 1 modelling the possibilities of 'mu', a random variable representing the probability of getting a head when a coin is tossed.
    a, b : parameters for the beta probability distribution, which we take as the prior distribution.

     Returns
    ---------
    posteriors : a Python list of numpy arrays, each representing the estimated posterior probability distribution, after the observation of each data point.
    '''
    
    prior = pdf_beta(mu, a, b)
    posteriors = []
    likelihood = np.zeros(mu.shape)
    posterior_num = np.zeros(mu.shape)
    posterior = np.zeros(mu.shape)
    d_mu = 1 / mu.shape[0]
    
    for data_point in X:
    
        likelihood[:] = np.multiply((mu ** data_point), ((1 - mu) ** (1 - data_point)))
        posterior_num[:] = np.multiply(likelihood, prior)
        posterior[:] = posterior_num / (sum(posterior_num * d_mu))
        prior[:] = posterior

        posteriors.append(posterior.copy())
    
    return posteriors


def plot_sequential_posteriors (mu, posteriors, means = None, method = 'Numerical'):
    '''
    Plots and saves the estimated posterior probability distributions for the mean of the Bernoulli distribution, after the observation of each data point, as per 
    the sequential method for estimation.

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




def numerical_method_entire (X, mu, a, b):
    '''
    This function uses the numerical method for estimating the posterior probability distribution of the mean of a Bernoulli 
    probability distribution, in one go, i.e., having observed the entire dataset at once.

    The likelihood function in case of considering all data points at once, is a binomial distribution with N as
    the number of data points, and m as the number of ones in the data. 

    In the numerical method, again, integration is approximated by taking ​'d_mu'​ as the space between
    consequent ​'mu'​ values. Integration is therefore appropriated as the summation of function values at each
    'mu'​ multiplied by ​'d_mu'.

     Arguments
    -----------
    X : numpy array of data points (1 or 0), each representing a random variable of the Bernoulli probability distribution.
    mu : numpy array of values between 0 and 1 modelling the possibilities of 'mu', a random variable representing the probability of getting a head when a coin is tossed.
    a, b : parameters for the beta probability distribution, which we take as the prior distribution.

     Returns
    ---------
    posterior_final : a numpy array representing the estimated posterior probability distribution for the mean, after the observation of the entire dataset at once.
    '''
    
    N = int(X.shape[0])
    d_mu = 1 / mu.shape[0]
    
    prior = pdf_beta (mu, a, b)
    likelihood = C(N, sum(X)) * np.multiply(mu ** (sum(X)) , (1-mu) ** (N - sum(X))) 
    posterior_final_num = likelihood * prior
    posterior_final = posterior_final_num / (sum(posterior_final_num * d_mu))
    
    return posterior_final


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

    # Construct dataset with 160 data points (1 or 0), having mean 0.7, with 10000 equally spaced values between 0 and 1 to model the possibilities for the mean, and 
    # with parameters a = 2 and b = 3 for the prior beta distribution of the mean.

    X, mu, a, b = initialize_data (160, 0.7, 10000, 2, 3)

    posteriors_seq_num = numerical_method_seq (X, mu, a, b)
    plot_sequential_posteriors(mu, posteriors_seq_num)

    posterior_final_num = numerical_method_entire (X, mu, a, b)
    plot_final_posterior (mu, posterior_final_num)


if __name__ == '__main__':
    
    main()

