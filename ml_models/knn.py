import numpy as np
from collections import Counter


class KNN():
    def __init__(self,k=3):
        self.k=k

    def fit(self,x,y):
        self.x_train=x
        self.y_train=y

    def predict(self,x):
        predictions=[self._predict(X) for X in x]
        return np.array(predictions)
    
    def _predict(self,x):
        distances=[np.linalg.norm(x-X_train) for X_train in self.x_train]
        k_index=np.argsort(distances[:self.k])
        kn_labels=[self.y_train[i] for i in  k_index]
        most_common = Counter(kn_labels).most_common(1)
        return most_common[0][0]

