import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import math as mt
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
class LinearRegression:

    def __init__(self,X,y):
        #initialize variables
        self.data = np.array(X)
        self.y = np.array(y)
        self.n_observation = len(y)
        
        def find_sse(beta):
            y_hat = beta[0] + np.sum(beta[1:] * self.data, axis=1)
            residuals = y - y_hat
            return np.sum(residuals**2)
            
        #initial guess is set to Zeros.
        beta_guess = np.zeros(self.data.shape[1] + 1)
        min_results = minimize(find_sse, beta_guess)
        #Coefficient arrays ( Betas )
        self.coefficients = min_results.x
        #array of best predicted y values 
        self.y_predicted = self.predict(self.data)
        #Residcals between y and the optimal y 
        self.residuals= y-self.y_predicted
        # The sum of squared errors objective function
        self.sse=np.sum(self.residuals **2)
        # the sum os squares 
        self.sst = np.sum((y-(np.mean(y)))**2)
        #r squared will be between 0-1 with one being best
        self.r_squared= 1 - (self.sse/self.sst)
 		#the residual standard error
        self.rse = mt.sqrt(self.sse/(self.n_observation-2))
    	#the log likelihood score for the optimal model.
        self.loglik = np.sum(np.log(norm.pdf(self.residuals,0,self.rse)))

    def predict(self,X):
        data = np.array(X)
        return self.coefficients[0]+np.sum(self.coefficients[1:]*data, axis=1)
        
    def score(self,X,y):
        values=np.array(X)
        labels = np.array(y)
        y_hat = self.predict(values)
        residuals = labels- y_hat
        sse = np.sum(residuals **2)
        sst = np.sum((labels-(np.mean(labels)))**2)
        return 1 - (sse/sst)

    def summary(self):
        print('+----------------------------+')
        print('| Linear Regression Summary  |')
        print('+----------------------------+')
        print('Number of training observations:', str(self.n_observation))
        print('Coefficient Estimates:', str(self.coefficients))
        print('Residual Standard Error:', str(self.rse))
        print('r-Squared:', str(self.r_squared))
        print('Log-Likelihood:', str(self.loglik))
        
#Testing algorithm with two features
np.random.seed(1)
X = np.random.uniform(0,10,100).reshape(50,2)
y = 3 + 1.3 * X[:,0] + 2.5 * X[:,1] + np.random.normal(0,3,50)
lm1 = LinearRegression(X,y)
lm1.summary()
X_test = np.random.uniform(0,10,40).reshape(20,2)
y_test = 3 + 1.3 * X_test[:,0] + 2.5 * X_test[:,1] + np.random.normal(0,3,20)
print("Testing r^2:", lm1.score(X_test, y_test), "\n")
X_new = np.array([[3,7], [6,1], [5,5]])
print("Predictions:", lm1.predict(X_new))

plt.close()
fig = plt.figure()
ax = fig.gca(projection='3d')


x1grid = np.arange(0, 10, 0.25)
x2grid = np.arange(0, 10, 0.25)
x1grid, x2grid = np.meshgrid(x1grid, x2grid)
ygrid = lm1.coefficients[0]+ lm1.coefficients[1] * x1grid + lm1.coefficients[2] * x2grid
ax.scatter(X[:,0], X[:,1],y, marker = 'o')
ax.plot_surface(x1grid, x2grid, ygrid, cmap=cm.coolwarm, alpha=0.8)


ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.show()





