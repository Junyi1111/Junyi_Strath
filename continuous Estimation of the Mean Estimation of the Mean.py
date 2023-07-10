#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np

class online_gaussian_estimator:
    def __init__(self,dim=2):
        self.dimension=dim
        self.M = np.zeros((self.dimension,self.dimension))
        self.mean = np.zeros((self.dimension,1))
        self.cov = np.zeros((self.dimension,self.dimension))
        self.runlength=0
        
    def online_estimate(self,x):
        delta = x - self.mean
        self.mean = self.mean + delta / (self.runlength + 1)
        self.M = self.M + delta.dot((x - self.mean).T)
        
        if self.runlength > 0:
            self.cov = M / (self.runlength + 1)
        else:
            self.M  # Adjust this line to compute population variance instead of sample variance
        
        self.runlength=self.runlength+1

    def reset_run_length(self):
        self.runlength=0

    def generate_sample(self):
        return np.mvn(self.mean,self.cov,1)


#the trouble with this is that the function is not online - it takes
#a whole data set and estimates from it, defeating the purpose...

#it also appears to be R code?
def online_estimator(X):
    n = X.shape[1]
    M = np.zeros((n, n))
    mean = np.zeros((n, 1))
    cov = np.zeros((n, n))
    all_means = []
    all_covs = []
    for i in range(X.shape[0]):
        x = X[i,:].reshape(n, 1)
        delta = x - mean
        mean = mean + delta / (i + 1)
        M = M + delta.dot((x - mean).T)
        cov = M / (i + 1) if i > 0 else M  # Adjust this line to compute population variance instead of sample variance
        all_means.append(mean.flatten())
        all_covs.append(cov.flatten())
    return {'mean_est': mean, 'cov_est': cov, 'all_means': all_means, 'all_covs': all_covs}
cov_mat = np.array([[1, -1, 0.5],
                    [-1, 3, -1],
                    [0.5, -1, 3]])
N = 5
mu = np.array([1, 10, 20])

# Sampling from the distribution
X = multivariate_normal.rvs(mean=mu, cov=cov_mat, size=N)

# Estimating the distribution's parameters
res = online_estimator(X)



# Creating a dataframe with all estimates
all_estimates = pd.DataFrame(res['all_means'], columns=[f'mean_{i+1}' for i in range(cov_mat.shape[0])])
all_estimates = pd.concat([all_estimates, pd.DataFrame(res['all_covs'], columns=[f'cov_{i+1}' for i in range(cov_mat.shape[0]**2)])], axis=1)
all_estimates['n'] = np.arange(1, N+1)

# Melting the dataframe for plotting
plot_df = pd.melt(all_estimates, id_vars='n')

# Defining the labels for the legend
legend_labels = [r'$\bar{x}_{1}$', r'$\bar{x}_{2}$', r'$\Sigma_{11}$', r'$\Sigma_{12}$', r'$\Sigma_{13}$', r'$\Sigma_{22}$']

# Creating the plot
sns.lineplot(data=plot_df, x='n', y='value', hue='variable', palette='Set1').legend(title='Variable', labels=legend_labels)
plt.show()

