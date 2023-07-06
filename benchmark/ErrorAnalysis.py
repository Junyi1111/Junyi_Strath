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
