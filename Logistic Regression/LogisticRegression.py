import numpy as np
from scipy.optimize import minimize

class LogisticRegression:

    def __init__(self,X,y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_observation = len(y)
        self.classes = np.unique(self.y)

        def find_neg_loglik(beta):
            z = beta[0] + np.sum(beta[1:] * X, axis=1)
            p = 1 / (1 + np.exp(-z))
            pi = np.where(y == self.y[0], p, 1 - p)
            loglik = np.sum(np.log(pi))
            return loglik * (-1)
        
        beta_guess = np.zeros(X.shape[1] + 1)
        min_results = minimize(find_neg_loglik, beta_guess) 
        self.coefficients = min_results.x 
        np.seterr(all='warn') 
        
        self.y_predicted = self.predict(self.X)
        
        
    def predict_proba(self,X):
        X = np.array(X)
        return 1 /(1- np.exp((-1*(self.coefficients[0]+np.sum(self.coefficients[1:]*X, axis=1)))))
    
    def predict(self,X,t):
        t=0.5
        X = np.array(X)
        p = self.predict_proba(X)
        return np.where(p>t,self.y[0],self.y[1])
    
    
    def score(self, X,y,t):
        t=0.5
        X = np.array(X)
        y = np.array(y)
        prediction = self.predict(X,t)
        return np.sum(prediction==y)/len(y)
        
        