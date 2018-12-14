# -*- coding: utf-8 -*-

# probabilistic programming and inference with PyMC3 library

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pymc3 as pm

'''
Alice and Bob are trading on the market. Both of them are selling the Thing and 
want to get as high profit as possible. Every hour they check out with each other's 
prices and adjust their prices to compete on the market. Although they have different 
strategies for price setting.

Alice: takes Bob's price during the previous hour, multiply by 0.6, add 90\$, 
add Gaussian noise from $(0, 20^2).

Bob: takes Alice's price during the current hour, multiply by 1.2 and subtract 
20, add Gaussian noise from N(0, 10^2).

The problem is to find the joint distribution of Alice and Bob's prices after 
many hours of such an experiment.
'''

def run_simulation(alice_start_price=300.0, bob_start_price=300.0, seed=42, num_hours=10000, burnin=1000):
    """Simulates an evolution of prices set by Bob and Alice.
    
    The function simulate Alice and Bob behavior for `burnin' hours, then ignore the obtained
    simulation results, and then simulate it for `num_hours' more.
    The initial burnin (also sometimes called warmup) is done to make sure that the distribution stabilized.
    
    Returns:
        two lists, with Alice and with Bob prices. Both lists should be of length num_hours.
    """
    np.random.seed(seed)

    alice_prices = [alice_start_price]
    bob_prices = [bob_start_price]
    
    for i in range(burnin+num_hours-1):
        a_price = bob_prices[-1]*0.6+90+np.random.randn()*20
        b_price = a_price*1.2-20+np.random.randn()*10
        alice_prices.append(a_price)
        bob_prices.append(b_price)    
    
    return alice_prices[burnin:], bob_prices[burnin:]


# The following function visualize the sampling process.
def plot_traces(traces, burnin=2000):
    ''' 
    Convenience function:
    Plot traces with overlaid means and values
    '''
    
    ax = pm.traceplot(traces[burnin:], figsize=(12,len(traces.varnames)*1.5),
        lines={k: v['mean'] for k, v in pm.df_summary(traces[burnin:]).iterrows()})

    for i, mn in enumerate(pm.df_summary(traces[burnin:])['mean']):
        ax[i,0].annotate('{:.2f}'.format(mn), xy=(mn,0), xycoords='data'
                    ,xytext=(5,10), textcoords='offset points', rotation=90
                    ,va='bottom', fontsize='large', color='#AA0022')
                    

if __name__ == '__main__':
    alice_prices, bob_prices = run_simulation(alice_start_price=300, bob_start_price=300, seed=42, num_hours=3, burnin=1)
    if len(alice_prices) != 3:
        raise RuntimeError('Make sure that the function returns `num_hours` data points.')
    
    #the average prices for Alice and Bob after the burnin period
    alice_prices, bob_prices = run_simulation(alice_start_price=300.0, bob_start_price=10.0, seed=42, num_hours=10000, burnin=1000)
    average_alice_price = np.mean(alice_prices)
    average_bob_price = np.mean(bob_prices)
    
    # 2-d histogram of prices, computed using kernel density estimation
    data = np.array(run_simulation())
    sns.jointplot(data[0, :], data[1, :], stat_func=None, kind='kde')
    
    # Pearson correlation coefficient of Alice and Bob prices
    corrAliceBob  = np.corrcoef(data[0, :], data[1, :])
    correlation = corrAliceBob[0,1]

    # MAP inference
    with pm.Model() as manual_logistic_model:
        # Declare pymc random variables for logistic regression coefficients with uninformative 
        # prior distributions N(0, 100^2) on each weight using pm.Normal. 
        alpha = pm.Normal('alpha', 0, 100^2)
        beta1 = pm.Normal('beta1', 0, 100^2)
        beta2 = pm.Normal('beta2', 0, 100^2)
        
        # Transform these random variables into vector of probabilities p(y_i=1) using logistic regression model specified 
        # above. PyMC random variables are theano shared variables and support simple mathematical operations.
        z = alpha * np.ones(data.shape[0]) + beta1 * data['age'].values + beta2 * data['educ'].values
        pY = pm.invlogit(z)
        
        # Declare PyMC Bernoulli random vector with probability of success equal to the corresponding value
        # given by the sigmoid function.
        yObs = pm.Bernoulli('yObs', pY, observed=data['income_more_50K'].values)
        
        # Find the maximum a-posteriori estimate for the vector of logistic regression weights.
        map_estimate = pm.find_MAP()
        print(map_estimate)

    # use Metropolis-Hastings algorithm for finding the samples from the posterior distribution
    with pm.Model() as logistic_model:
        # Since it is unlikely that the dependency between the age and salary is linear, we will include age squared
        # into features so that we can model dependency that favors certain ages.
        # Train Bayesian logistic regression model on the following features: sex, age, age^2, educ, hours
        # Train the model for 400 samples.
        data['age_2'] = data['age']^2
        pm.glm.GLM.from_formula('income_more_50K ~ sex + age + educ + hours + age_2', data=data, family='binomial')
        trace = pm.sample(400, step=pm.Metropolis())
        
    plot_traces(trace, burnin=200)
    
    # use NUTS as sampling algorithm, which is a form of Hamiltonian Monte Carlo, in which parameters are tuned automatically
    with pm.Model() as logistic_model:
        # Train Bayesian logistic regression model on the following features: sex, age, age_squared, educ, hours
        # Use pm.sample to run MCMC to train this model.
        # Train the model for *4000* samples (ten times more than before).
        pm.glm.GLM.from_formula('income_more_50K ~ sex + age + educ + hours + age_2', data=data, family='binomial')
        trace = pm.sample(4000, step=pm.NUTS())
        
    plot_traces(trace)
     
    # We don't need to use a large burn-in here, since we initialize sampling
    # from a good point (from our approximation of the most probable
    # point (MAP) to be more precise).
    burnin = 100
    b = trace['sex[T. Male]'][burnin:]
    plt.hist(np.exp(b), bins=20, normed=True)
    plt.xlabel("Odds Ratio")
    plt.show()
    
    lb, ub = np.percentile(b, 2.5), np.percentile(b, 97.5)
    print("P(%.3f < Odds Ratio < %.3f) = 0.95" % (np.exp(lb), np.exp(ub)))
