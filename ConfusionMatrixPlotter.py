import matplotlib.pyplot as plt
import numpy as np

class ConfusionMatrixPlotter:
    def __init__(self, cm, true_labels):
        self.cm = cm
        self.classes = sorted(list(set(true_labels)))
    
    @staticmethod
    def _custom_colormap(cm):
        color_map = np.ones((cm.shape[0], cm.shape[1], 3))
        max_val = cm.max()
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i == j:
                    color_map[i, j] = [0.7, 0.7, 1]
                elif cm[i, j] != 0:
                    color_map[i, j] = [1, 1 - (cm[i, j] / max_val), 1 - (cm[i, j] / max_val)]
        return color_map

    def plot(self):
        plt.figure(figsize=(10, 8))
        plt.imshow(self._custom_colormap(self.cm), interpolation='nearest')
        plt.title('Confusion Matrix')
        
        tick_marks = range(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)

        for i in range(self.cm.shape[0]):
            for j in range(self.cm.shape[1]):
                plt.text(j, i, format(self.cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="black")

        plt.tight_layout()
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()



