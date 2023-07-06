import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ErrorAnalysis:
    def __init__(self, result):
        self.result = result
        self.length = len(result)
        self.error = result.Actual - result.Predicted
        self.error = self.error.reset_index(drop=True)
        self.updated_error1 = None

    def calculate_errors(self):
        half_length = self.length // 2
        half_error = self.error[:-half_length]
        half_error = np.array(half_error)
        errors_reshaped = half_error.reshape((half_length // 48, 48))
        errors_reshaped = errors_reshaped.T

        means = np.mean(errors_reshaped, axis=1)
        covariances = np.cov(errors_reshaped, rowvar=True)

        error_list = [half_length + 48 * i for i in range(half_length // 48 )]
        observer_error = self.error[error_list]
        observer_error = observer_error.reset_index(drop=True)

        updated_error = []
        for i in range(47):
            column = []
            for j in range(half_length// 48):
                updated_value = means[i + 1] + covariances[0, i + 1] * (covariances[0, 0] ** (-1)) * (
                        observer_error[j] - means[0])
                column.append(updated_value)
            updated_error.append(column)

        updated_error = np.vstack((observer_error, updated_error))

        secondhalf_error = self.error[half_length:]
        secondhalf_error = np.array(secondhalf_error)
        secondhalf_error_reshaped = secondhalf_error.reshape((half_length // 48, 48))
        secondhalf_error = secondhalf_error_reshaped.T

        self.updated_error1 = updated_error.T.flatten()
        error_original = self.result.Actual[half_length:] - self.result.Predicted[half_length:]
        error_update = self.result.Actual[half_length:] - (self.result.Predicted[half_length:] + self.updated_error1)

        mae_original = mean_absolute_error(self.result.Actual[half_length:], self.result.Predicted[half_length:])
        mse_original = mean_squared_error(self.result.Actual[half_length:], self.result.Predicted[half_length:])

        mae_update = mean_absolute_error(self.result.Actual[half_length:], self.result.Predicted[half_length:] + self.updated_error1)
        mse_update = mean_squared_error(self.result.Actual[half_length:], self.result.Predicted[half_length:] + self.updated_error1)
        result_update = self.result.Predicted[half_length:] + self.updated_error1
        return {'mae_original': mae_original, 'mse_original': mse_original, 'mae_update': mae_update, 'mse_update': mse_update, 'result_update': result_update}

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
        update_errorl = updated_error.T
        return{'updated_error':update_errorl}
