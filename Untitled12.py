#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

