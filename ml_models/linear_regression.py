import numpy as np 
# y = mx + c

class LinearRegression():
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)        
        self.bias = 0

        for _ in range(self.epochs):
            y_pred = np.dot(x, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(x.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

        
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, x):
        return np.dot(x, self.weights) + self.bias
    
    def score(self, x, y):
        y_pred = self.predict(x)
      
        u = np.sum((y - y_pred) ** 2)
        v = np.sum((y - np.mean(y)) ** 2)
        return 1 - (u / v)
