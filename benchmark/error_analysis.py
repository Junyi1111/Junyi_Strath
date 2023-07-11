def online_estimation(self):
    error = self.result.Actual - self.result.Predicted
    error = np.array(error)
    errors_reshaped = error.reshape(len(error) // 48, 48)
    n = errors_reshaped.shape[1]
    M = np.zeros((n, n))
    mean = np.zeros((n, 1))
    cov = np.zeros((n, n))
    all_means = []
    all_covs = []
    error_list = [48 + 48 * i for i in range(len(error) // 48 - 1)]
    observer_error = error[error_list]
    update_error_all = []
    for i in range(errors_reshaped.shape[0] - 1):
        column = []
        x = errors_reshaped[i, :].reshape(n, 1)
        delta = x - mean
        mean = mean + delta / (i + 1)
        M = M + delta.dot((x - mean).T)
        cov = M / (i + 1) if i > 0 else M
        for j in range(47):
            updated_value = mean[j + 1] + cov[0, j + 1] * (cov[0, 0] ** (-1)) * (observer_error[i] - mean[0])
            column.append(updated_value)
        column = np.hstack(column)
        update_error_all.append(column)
        all_means.append(mean.flatten())
        all_covs.append(cov.flatten())
    update_error_all = np.array(update_error_all)
    update_error_all = update_error_all.T
    updated_error = np.vstack((observer_error, update_error_all))
    update_error1 = updated_error.T
    return {'updated_error': update_error1}
