import numpy as np
class LogisticRegression():
    def __init__(self,lr,epochs=1000):
        self.lr=lr
        self.epochs=epochs
        self.weights=None
        self.bias=None

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def fit(self,x,y):
        n_samples,n_features=x.shape
        self.weights=np.zeros(n_features)
        self.bias=0

        for _ in range(self.epochs):
            linear_model=np.dot(x,self.weights)+self.bias
            y_predicted=self.sigmoid(linear_model)

            dw=(1/n_samples)*np.dot(x.T,(y_predicted-y))
            db=(1/n_samples)*np.sum(y_predicted-y)

            self.weights=dw*self.lr
            self.bias=db*self.lr
    def predictprob(self,x):
        linear_model=np.dot(x,self.weights)+self.bias
        return self.sigmoid(linear_model)
    
    def predict(self,x):
        y_prob=self.predictprob(x)
        return np.where(y_prob>=0.5,1,0)

        



            


        