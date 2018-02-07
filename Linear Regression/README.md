
# Code

## Simple Linear Regresion

In a simple linear regression task, we have a single feature, `x`, from which we wish to predict a continuous, real-valued label `y`.

* Assume that our training set has $n$ observations.
* Denote the observed values of the training feature (predictor) as $x_1, x_2, ..., x_n$.
* Denote the observed values of the label (response variable) as $y_1, y_2, ..., y_n$.
* We assume that our model has the following form: $\large \hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x$.
* In the model above, $\hat{\beta}_0$ and $\beta_1$ are model parameters that are learned from the training data. 
* $\hat{y}$ represents the predicted value of $y$ given some value of $x$. 

### Training the Model

* Let $b_0$ and $b_1$ be a pair of (not necessarily optimal) parameter values used to define a model $\large \hat{y} = {b}_0 + {b}_1 x$.
* For each training observation $(x_i, y_i)$, let $\large\hat{y}_i = b_0 + b_1 x_i$. 
* For each $i$, calculate the error (residual) $\large\hat{e}_i = \hat{y}_i - y_i$. 
* The goal of the algorithm is to find the parameter values that minimize the Sum of Squared Errors objective function, given by: $ \large SSE = \sum \hat{e}_i^2 $
* We will denote the optimal parameter values by $\hat{\beta}_0$ and $\hat{\beta}_1$.

## r-Squared

When working with linear regression, the optimal parameter values are determined by minimizing the objective function SSE. However, a different value is typically used to assess the quality of a regression model. This alternate scoring method is called the called the **r-squared value**. It is defined as follows:

* $ \large SST = \sum (y_i - \bar y ) ^2 $

* $ \large SSE = \sum \hat{e}_i^2 = \sum (y_i - \hat {y}_i ) ^2 $

* $ \large r^2 = 1 - \frac{SSE}{SST}$

Since SST is a constant for the supplied training data, minimizing SSE is equivalent to maximizing $r^2$. The score supplied by $r^2$ has two advantages over SSE:

1. The value $r^2$ is "normalized" to always be between 0 and 1. A value close to 1 indicates that our model provides a very good fit for the data. 
2. The $r^2$ value has a useful interpretation not present with SSE. We can think of $r^2$ as reporting the proportion of the variance in the training labels that has been accounted for by our model. 

We will now directly compute the $r^2$ value for our model. 

## Log-Likelihood

his is given bythe following formula: ln($L$)=ln[$f$($\small\hat{e}_1$))]+ln[$f$($\small\hat{e}_2$))]+⋯+lnln[$f$($\small\hat{e}_n$))],where $f$($x$)is the pdf for the normal distribution $N$(0,$rse$).

# Testing

## Multiple Regression

In a multiple linear regression task, we have several features, $X = [x^{(1)}, x^{(2)}, ..., x^{(p)}]$, from which we wish to predict a single continuous, real-valued label `y`.

* Assume that our training set has $n$ observations.
* Denote the values of the training features for observation number $i$ as $X_i = [x^{(1)}_i, x^{(2)}_i, ..., x^{(p)}_i]$.
* Denote the value of the label for observation $i$ as $y_i$. 
* We assume that our model has the following form: $\large \hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x^{(1)} + \hat{\beta}_2 x^{(2)} ... + \hat{\beta}_p x^{(p)}$.
* In the model above, $\hat{\beta}_0$, $\beta_1$, ..., $\beta_p$ are model parameters that are learned from the training data. 
* $\hat{y}$ represents the predicted value of $y$ given some input vector $X = [x^{(1)}, x^{(2)}, ..., x^{(p)}]$. 

### Training the Model

* Let $b_0, b_1, ..., b_p$ be a set (not necessarily optimal) parameter values used to define a model $\large \hat{y} = {b}_0 + {b}_1 x^{(1)} + {b}_2 x^{(2)} + ... + {b}_p x^{(p)}$.
* For each training observation $(x_i, y_i)$, let $\large\hat{y}_i = {b}_0 + {b}_1 x^{(1)}_i + {b}_2 x^{(2)}_i + ... + {b}_p x^{(p)}_i$.
* For each $i$, calculate the error (residual) $\large\hat{e}_i = \hat{y}_i - y_i$. 
* The goal of the algorithm is to find the parameter values that minimize the Sum of Squared Errors objective function, given by: $ \large SSE = \sum \hat{e}_i^2 $
* We will denote the optimal parameter values by $\hat{\beta}_0$, $\beta_1$, ..., $\beta_p$

### Results 

At the end of the program we will call the summary function that will give us the r-squared value and the log-likelihood for a certain regression
