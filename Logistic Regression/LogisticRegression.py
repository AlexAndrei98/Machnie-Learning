import numpy as np
import pandas as pd
from scipy.optimize import minimize

class LogisticRegression:

    def __init__(self,X,y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_observation = len(y)
        self.classes = np.unique(self.y)
        
        def find_neg_loglik(beta):
            z = beta[0] + np.sum(beta[1:] * self.X, axis=1)
            p = 1 / (1 + np.exp(-z))
            pi = np.where(self.y == self.classes[1], p, 1 - p)
            loglik = np.sum(np.log(pi))
            return loglik * (-1)
        
        np.seterr(all='ignore') 
        beta_guess = np.zeros(X.shape[1] + 1)
        min_results = minimize(find_neg_loglik, beta_guess) 
        self.coefficients = min_results.x 
        np.seterr(all='warn') 
        
        self.y_predicted = self.predict(self.X)
        self.loglik = find_neg_loglik(self.coefficients)*(-1)
        self.accuracy = self.score(self.X,self.y)
        
        
    def predict_proba(self,X):
        X = np.array(X)
        linear = self.coefficients[0]+np.sum(self.coefficients[1:]*X, axis=1)
        return 1 /(1 + np.exp((-1 * linear )))
    
    def predict(self,X,t=0.5):
        X = np.array(X)
        p = self.predict_proba(X)
        return np.where(p>t,self.classes[1],self.classes[0])
    
    
    def score(self, X,y,t=0.5):
        X = np.array(X)
        y = np.array(y)
        prediction = self.predict(X)
        return np.sum(prediction==y)/len(y)
    
    def confusion_matrix(self, X,y):
        X = np.array(X)
        y = np.array(y)
        class_0 = self.classes[0]
        class_1 = self.classes[1]
        TP = np.sum((self.y==class_0) & (self.y_predicted == class_0))
        FP = np.sum((self.y==class_0) & (self.y_predicted == class_1))
        TN = np.sum((self.y==class_1) & (self.y_predicted == class_0))
        FN = np.sum((self.y==class_1) & (self.y_predicted == class_1))
        cm = pd.DataFrame([[TP,FP],[TN, FN]])
        cm.columns = ['Pred_0', 'Pred_1'] 
        cm.index = ['True_0','True_1']
        print(cm)
    
    def summary(self):
        print('+----------------------------+')
        print('|Logistic Regression Summary |')
        print('+----------------------------+')
        print('Number of training observations:', str(self.n_observation))
        print('Coefficient Estimates:', str(self.coefficients))
        print('Log-Likelihood:', str(self.loglik))
        print('Accuracy:', str(self.accuracy))
        print('Class 0:',str(self.classes[0]))
        print('Class 1:',str(self.classes[1]))


width = [6.4, 7.7, 6.7, 7.4, 6.5, 6.9, 7.8, 7.6, 6.2, 7.4, 7.7, 6.8]
height = [8.2, 7.5, 6.6, 8.8, 6.8, 6.8, 7.6, 8.8, 8.4, 7.3, 7.4, 7.2]
X = pd.DataFrame({'x1':width, 'x2':height})
y = ['Lemon', 'Orange', 'Orange', 'Lemon', 'Orange', 'Lemon', 'Orange', 'Lemon', 'Lemon', 'Orange', 'Lemon', 'Lemon']
fruit_mod = LogisticRegression(X,y)
fruit_mod.summary()
fruit_mod.confusion_matrix(X,y)
X_test = pd.DataFrame({'x1':[7.4, 7.1, 6.4], 'x2':[7.2, 7.8, 6.8]})
y_test = ['Orange', 'Orange', 'Lemon']
print("Test Set Performance:")
print(fruit_mod.predict_proba(X_test))
print(fruit_mod.predict(X_test))
print(fruit_mod.score(X_test,y_test))

