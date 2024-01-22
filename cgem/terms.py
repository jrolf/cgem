
#############################################################################
#############################################################################

#############################################################################
#############################################################################

import numpy as np 
import pandas as pd
import pandas_ta as ta

from random import shuffle, choice
import random,time,os,io,requests,datetime
import json,hmac,hashlib,base64,pickle 
from collections import defaultdict as defd
from heapq import nlargest
from copy import deepcopy

from scipy import signal
from scipy.stats import entropy 
from scipy.constants import convert_temperature
from scipy.interpolate import interp1d
#from scipy.ndimage.filters import uniform_filter1d

#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
#https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#sphx-glr-auto-examples-ensemble-plot-forest-importances-py
from sklearn.ensemble       import ExtraTreesClassifier       as ETC
from sklearn.ensemble       import ExtraTreesRegressor        as ETR
from sklearn.ensemble       import BaggingClassifier          as BGC 
from sklearn.ensemble       import GradientBoostingClassifier as GBC 
from sklearn.ensemble       import GradientBoostingRegressor  as GBR 
from sklearn.neural_network import MLPRegressor               as MLP 
from sklearn.linear_model   import LinearRegression           as OLS
from sklearn.preprocessing  import LabelBinarizer             as LBZ 
from sklearn.decomposition  import PCA                        as PCA 

from sklearn.model_selection import cross_validate, ShuffleSplit, train_test_split
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline 
from sklearn.utils import check_array
from sklearn.preprocessing import * 
from sklearn.metrics import *

from sympy.solvers import solve
from sympy import Symbol,Eq,sympify
from sympy import log,ln,exp #,Wild,Mul,Add,sin,cos,tan

import statsmodels.formula.api as smf
from pygam import LinearGAM, GAM #, s, f, l, te

import xgboost as xgb
from xgboost import XGBRegressor as XGR


#############################################################################
#############################################################################


# Set print options: suppress scientific notation and set precision
np.set_printoptions(suppress=True, precision=8)
# Set Numpy error conditions: 
old_set = np.seterr(divide = 'ignore',invalid='ignore') 


#############################################################################
#############################################################################




def CalcCorr(x,y):
    return np.corrcoef(x,y)[0][1]

def PosCor(x,y):
    return max(0.0,CalcCorr(x,y)) 

def calc_rmse(actual, predictions):
    """
    Calculate the Root Mean Square Error (RMSE) between actual and predicted values.

    Parameters:
    actual (numpy array): The actual values.
    predictions (numpy array): The predicted values.

    Returns:
    float: The RMSE value.
    """
    # Calculate the square of differences
    differences = np.subtract(actual, predictions)
    squared_differences = np.square(differences)

    # Calculate the mean of squared differences
    mean_squared_differences = np.mean(squared_differences)

    # Calculate the square root of the mean squared differences (RMSE)
    rmse = np.sqrt(mean_squared_differences)
    return rmse


def calc_r2(actual, predictions):
    """
    Calculate the R-squared value between actual and predicted values.

    Parameters:
    actual (numpy array): The actual values.
    predictions (numpy array): The predicted values.

    Returns:
    float: The R-squared value.
    """
    # Calculate the mean of actual values
    mean_actual = np.mean(actual)

    # Calculate the total sum of squares (SST)
    sst = np.sum(np.square(np.subtract(actual, mean_actual)))

    # Calculate the residual sum of squares (SSR)
    ssr = np.sum(np.square(np.subtract(actual, predictions)))

    # Calculate R-squared
    r_squared = 1 - (ssr / sst)
    return r_squared


def robust_mean(distribution, center=0.7):
    """
    Calculate the mean of a distribution, excluding outliers.

    Parameters:
    distribution (array-like): The input distribution from which the mean is calculated.
    center (float): The central percentage of the distribution to consider. 
                    Default is 0.7, meaning the middle 70% is considered.

    Returns:
    float: The mean of the distribution after excluding outliers.
    """
    if not isinstance(distribution, np.ndarray):
        distribution = np.array(distribution)

    if distribution.size == 0 or not np.issubdtype(distribution.dtype, np.number):
        return np.nan

    margin = 100.0 * (1 - center) / 2.0
    min_val = np.percentile(distribution, margin)
    max_val = np.percentile(distribution, 100.0 - margin)

    filtered_dist = distribution[(distribution >= min_val) & (distribution <= max_val)]

    return np.mean(filtered_dist) if filtered_dist.size > 0 else np.nan


#############################################################################
#############################################################################

def OlsFromPoints(xvals, yvals):
    """
    Create an OLS model from given x and y values.

    Parameters:
    xvals (array-like): The x-values of the data points.
    yvals (array-like): The y-values of the data points.

    Returns:
    LinearRegression: A fitted OLS model.
    """
    xvals = np.array(xvals).reshape(-1, 1)
    yvals = np.array(yvals)
    
    if len(xvals) != len(yvals):
        raise ValueError("xvals and yvals must have the same length.")
    
    model = OLS() 
    model.fit(xvals, yvals)
    return model

def GetOlsParams(ols_model):
    """
    Extract the slope and intercept from an OLS model.

    Parameters:
    ols_model (LinearRegression): The OLS model.

    Returns:
    tuple: A tuple containing the slope (m) and intercept (b) of the model.
    """
    m = ols_model.coef_[0]
    b = ols_model.intercept_
    return m, b

def GenInverseOLS(normal_ols_model):
    """
    Generate an inverse OLS model from a given OLS model.

    Parameters:
    normal_ols_model (LinearRegression): The original OLS model.

    Returns:
    LinearRegression: The inverse OLS model.
    """
    m, b = GetOlsParams(normal_ols_model)
    if m == 0:
        raise ValueError("The slope of the OLS model is zero; inverse model cannot be generated.")
    
    inv_func = lambda y: (y - b) / m
    xvals = np.linspace(-100, 100, 1000)
    yvals = inv_func(xvals)
    
    return OlsFromPoints(yvals, xvals)  # Note the switch of xvals and yvals here

# Example usage:
# x_vals, y_vals = some_data_loading_function()
# ols_model = OlsFromPoints(x_vals, y_vals)
# inverse_ols_model = GenInverseOLS(ols_model)

#############################################################################
#############################################################################


def norm_flat(X):
    x_type = str(type(X)) 
    if ('DataFrame' in x_type) or ('Series' in x_type):
        X = X.values 
    return np.array(X).flatten() 


def clean_shape(X, y):
    """
    Reshapes the input features X and target y into shapes compatible with scikit-learn models.

    Parameters:
    X: array-like, list, DataFrame, or Series - input features
    y: array-like, list, DataFrame, or Series - target values

    Returns:
    X2, y2: reshaped versions of X and y, suitable for use with scikit-learn models
    """
    # Ensure X is a 2D array-like structure
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X2 = X.values
    else:
        X2 = np.array(X)
    # Reshape X to 2D if it's 1D, assuming each element is a single feature
    if X2.ndim == 1:
        X2 = X2.reshape(-1, 1)
    # Ensure y is a 1D array-like structure
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y2 = y.values.ravel()  # Flatten to 1D
    else:
        y2 = np.array(y).ravel()
    # Check if X2 and y2 are in acceptable shape for sklearn models
    X2 = check_array(X2)
    y2 = check_array(y2, ensure_2d=False)

    return X2, y2


Usage = '''

# Example usage:
X = [1,2,3,4,3,4,5,6,4,6,7,8]
y = [2,3,4,3,4,5,6,5,6,7,8,9]
X2, y2 = clean_shape(X, y)

print("X2 shape:", X2.shape)
print("y2 shape:", y2.shape)
print() 

from sklearn.linear_model import LinearRegression as OLS

model = OLS() 
model.fit(X2, y2) 
yhat = model.predict(X2) 

for y_1, y_hat in zip(y,yhat):
    print(y_1, y_hat) 
    
print()
print(y2.mean())
print(yhat.mean())

print()
print(y2.std())
print(yhat.std())
    
'''

#############################################################################
#############################################################################


class PolyFit:
    def __init__(self, poly=[2, 3, 4, 5]):
        """
        Initialize the PolyFit class with specified polynomial degrees.

        Parameters:
        poly (list or int): Polynomial degrees to fit. If an integer is provided, it's converted to a list.
        """
        self.poly = np.atleast_1d(poly).tolist()
        self.models = {}

    def _validate_and_reshape_input(self, X):
        """Validates and reshapes the input to a 1D numpy array."""
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.ndim > 1:
            if X.shape[1] != 1:
                raise ValueError("X needs to be a 1D array or 2D array with one feature.")
            X = X.ravel()

        return X

    def fit(self, x_train, y_train, poly=[]):
        """
        Fit polynomial models to the training data.

        Parameters:
        x_train (array-like): Training data features.
        y_train (array-like): Training data targets.
        poly (list or int, optional): Polynomial degrees to fit. If specified, overrides the instance's poly attribute.
        """
        if poly:
            self.poly = np.atleast_1d(poly).tolist()

        x = self._validate_and_reshape_input(x_train)
        y = self._validate_and_reshape_input(y_train)

        for deg in self.poly:
            params = np.polyfit(x, y, deg)
            self.models[deg] = params

    def predict(self, x_test):
        """
        Predict using the polynomial models on the test data.

        Parameters:
        x_test (array-like): Test data features.

        Returns:
        numpy.ndarray: Mean predictions from all polynomial models.
        """
        x = self._validate_and_reshape_input(x_test)
        predictions = [np.polyval(self.models[deg], x) for deg in self.poly]
        return np.mean(predictions, axis=0)

# Example usage:
# model = PolyFit()
# model.fit(x_train, y_train)
# preds = model.predict(x_test)



class MedianModel:
    def __init__(self, samples=1000, portion=0.05, radius=0, middle=0.2):
        """
        Initialize the MedianModel class.

        Parameters:
        samples (int): Number of samples to consider.
        portion (float): Portion of the range to consider for radius calculation.
        radius (float): Radius around each point to consider for median calculation.
        middle (float): Parameter for the robust mean calculation.
        """
        self.n = samples
        self.p = portion 
        self.r = radius 
        self.m = middle

    def _validate_and_reshape_input(self, X):
        """Validates and reshapes the input to a 1D numpy array."""
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.ndim > 1:
            if X.shape[1] != 1:
                raise ValueError("X needs to be a 1D array or 2D array with one feature.")
            X = X.ravel()

        return X

    def fit(self, x_train, y_train):
        """
        Fit the model using the training data.

        Parameters:
        x_train (array-like): Training data features.
        y_train (array-like): Training data targets.
        """
        x = self._validate_and_reshape_input(x_train)
        y = self._validate_and_reshape_input(y_train)
        self.x, self.y = x, y 

        xmin, xmax = x.min(), x.max() 
        if not self.r: 
            self.r = (xmax - xmin) * self.p

        yvals = []
        xvals = np.linspace(xmin, xmax, self.n)
        for xval in xvals: 
            xlo, xhi = xval - self.r, xval + self.r
            mask = (x >= xlo) & (x <= xhi) 
            if np.any(mask):
                med = RobustMean(y[mask], self.m) 
                yvals.append(med)
            else:
                yvals.append(np.nan)
        
        self.xv, self.yv = xvals, np.array(yvals)

    def predict(self, x_test):
        """
        Predict using the model on the test data.

        Parameters:
        x_test (array-like): Test data features.

        Returns:
        numpy.ndarray: Predictions for each test data point.
        """
        x = self._validate_and_reshape_input(x_test)
        preds = []
        for xval in x: 
            xlo, xhi = xval - self.r, xval + self.r
            mask = (self.x >= xlo) & (self.x <= xhi) 
            if np.any(mask):
                med = RobustMean(self.y[mask], self.m) 
                preds.append(med)
            else:
                preds.append(np.nan)

        return np.array(preds)

# Example usage:
# model = MedianModel()
# model.fit(x_train, y_train)
# predictions = model.predict(x_test)


class InterpModel:
    def __init__(self):
        """
        Initialize the InterpModel class. This class provides methods for fitting 
        and predicting using linear and cubic interpolation.
        """
        self.lin_predict = None
        self.cub_predict = None
        self.xmin = None
        self.xmax = None

    def _validate_and_reshape_input(self, X):
        """Validates and reshapes the input to a 1D numpy array."""
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.ndim > 1:
            if X.shape[1] != 1:
                raise ValueError("X needs to be a 1D array or 2D array with one feature.")
            X = X.ravel()

        return X

    def fit(self, x_train, y_train):
        """
        Fit the interpolation model using the training data.

        Parameters:
        x_train (array-like): Training data features.
        y_train (array-like): Training data targets.
        """
        x = self._validate_and_reshape_input(x_train)
        y = self._validate_and_reshape_input(y_train)

        if x.size < 2:
            raise ValueError("At least two data points are required for interpolation.")

        self.lin_predict = interp1d(x, y, kind='linear', fill_value='extrapolate')
        self.cub_predict = interp1d(x, y, kind='cubic', fill_value='extrapolate')
        self.xmin = x.min()
        self.xmax = x.max()

    def predict(self, x_test, kind='linear'):
        """
        Predict using the interpolation model on the test data.

        Parameters:
        x_test (array-like): Test data features.
        kind (str): Type of interpolation ('linear' or 'cubic').

        Returns:
        numpy.ndarray: Predictions for each test data point.
        """
        x = self._validate_and_reshape_input(x_test)
        x_clipped = np.clip(x, self.xmin, self.xmax)

        if kind not in ['linear', 'cubic']:
            raise ValueError("Interpolation kind must be either 'linear' or 'cubic'.")

        predictor = self.lin_predict if kind == 'linear' else self.cub_predict
        return predictor(x_clipped)

# Example usage:
# model = InterpModel()
# model.fit(x_train, y_train)
# predictions = model.predict(x_test, kind='linear')


#############################################################################
#############################################################################
#############################################################################

### CLEAN ACCOUNTING OF THE TERM CLASSES ###

Notes = '''

TODO:
————————
CGEM Terms
————————
ScalarTerm  # Unconstrained Scalar   #LimitTerm   # Constrained Scalar
FactorTerm

CatEffectTerm 
CatFactorTerm

LinearTerm
PolyTerm
SplineTerm

ProphetTerm
RForestTerm
XGBoostTerm
CatBoostTerm
MlpTerm

————————
MultiLevel Random Effects
Bivariate spline interaction effect
Random Effects FUNCTIONS
Monotonic functions
Interpolation Model
————————

'''

### VALIDATED ###
class DirectVar:
    
    def __init__(self):
        pass
    
    def fit(self,X=[],y=[]): 
        pass
    
    def predict(self,X=[]):
        return norm_flat(X) 


### VALIDATED ###
class ScalarTerm:
    
    def __init__(self):
        pass

    def fit(self,X=[],y=[]): 
        # Only "y" is required here.
        # X should be initialized to ensure len(X)=0
        self.scalar = np.array(y).mean() 

    def predict(self,X=[]):  
        # We "predict" a single number for every row of X
        return self.scalar * np.ones(len(X)) 


### VALIDATED ###
class CatRegModel:
    def __init__(self):
        """
        Initialize the CatRegModel. This model encodes categorical variables 
        and fits a linear regression model.
        """
        self.encoder = LBZ()
        self.model = OLS()

    def fit(self, X, y):
        """
        Fit the model with the encoded features.

        Parameters:
        X (array-like): Feature variable.
        y (array-like): Target variable.
        """
        X_encoded = self.encoder.fit_transform(X)
        self.model.fit(X_encoded, y)

    def predict(self, X):
        """
        Predict using the fitted model.

        Parameters:
        X (array-like): Feature variable.

        Returns:
        numpy.ndarray: Predicted values.
        """
        X_encoded = self.encoder.transform(X)
        return self.model.predict(X_encoded)



#############################################################################
#############################################################################
#############################################################################



































#############################################################################
#############################################################################
#############################################################################


Notes = '''



class InterceptModel:
    def __init__(self):
        """
        Initialize the InterceptModel. This model predicts a constant value 
        based on the mean of the target variable.
        """
        self.expected_value = 0

    def fit(self, y):
        """
        Fit the model by calculating the mean of the target variable.

        Parameters:
        y (array-like): Target variable.
        """
        self.expected_value = np.mean(y)

    def predict(self):
        """
        Predict using the calculated mean.

        Returns:
        float: The expected value.
        """
        return self.expected_value



class RandomEffectsModel:
    def __init__(self, group_var='COUNTRY'):
        """
        Initialize the RandomEffectsModel. This model fits a mixed linear 
        model with random effects.

        Parameters:
        group_var (str): Variable name for grouping.
        """
        self.group_var = group_var
        self.model = None
        self.result = None
        self.intercept = None
        self.group_names = None
        self.effect = None
        self.preds = None

    def fit(self, X, y):
        """
        Fit the mixed linear model.

        Parameters:
        X (pandas.DataFrame): Feature variables.
        y (array-like): Target variable.
        """
        tdf = pd.DataFrame(X)
        tdf['Y'] = y
        fixed_model = "Y ~ 1"
        self.model = smf.mixedlm(fixed_model, tdf, groups=tdf[self.group_var])
        self.result = self.model.fit()
        self.intercept = self.result.fe_params['Intercept']
        self.group_names = list(self.result.random_effects)
        self.effect = {group: round(float(self.result.random_effects[group]), 9)
                       for group in self.group_names}
        self.preds = self.result.fittedvalues

    def predict(self, X=None, y=None):
        """
        Predict using the fitted model.

        Parameters:
        X (pandas.DataFrame, optional): Feature variables.
        y (array-like, optional): Target variable.

        Returns:
        pandas.Series: Predicted values.
        """
        if X is not None and y is not None:
            self.fit(X, y)
        return self.preds




class CatEffectTerm: ### V1 !!! 

    def __init__(self):
        pass

    def fit(self,X,y): 
        # X is the list of category names, per record in the training set.
        # y is the target effect we are converging on.
          # "x" is now a 1-D array of category names.
        # Calculate the average values per category and return a dict: 
        self.cat_vals = self.calc_cat_means(x,y)  

    def predict(self,X):
        # We "predict" a single number for every row of X
        X2 = self.norm_flat(X) 
        map_cats_to_vals

        return self.scalar * np.ones(len(X)) 


    def calc_cat_means(self,x,y):
        # Find the unique categories and their counts
        categories, counts = np.unique(x, return_counts=True)
        # Sum the observations for each category
        sums = np.bincount(x, weights=y)
        # Compute averages, avoiding division by zero for any category not in x
        averages = sums[categories] / counts
        return dict(zip(categories, averages))

    def map_cats_to_vals(self,cats2vals,new_cats):
        # Create a mapping from category IDs to indices
        unique_ids = np.unique(new_cats)
        id2index = {id_: i for i, id_ in enumerate(unique_ids)} 
        # Create an array of averages using this mapping
        averages_array = np.array([cats2vals.get(id_, 0) for id_ in unique_ids])
        # Map the averages to the new IDs in array 'a' using the mapping
        idx = np.vectorize(id2index.get)(a) 
        return averages_array[idx]
 
    def norm_flat(self,X):
        x_type = str(type(X)) 
        if ('DataFrame' in x_type) or ('Series' in x_type):
            X = X.values 
        return np.array(X).flatten() 




class CatEffectTerm:   ### V2 !!!

    def __init__(self):
        pass

    def fit(self,X,y): 
        # X is the list of category names, per record in the training set.
        # y is the target effect we are converging on.
          # "x" is now a 1-D array of category names.
        # Calculate the average values per category and return a dict: 
        self.cat_vals = self.calc_cat_means(x,y)  

    def predict(self,X):
        # We "predict" a single number for every row of X
        X2 = norm_flat(X) 
        preds = self.map_cat_vals(X2) 
        return preds

    def calc_cat_means(self,x,y):
        # Find the unique categories and their counts
        categories, counts = np.unique(x, return_counts=True)
        # Sum the observations for each category
        sums = np.bincount(x, weights=y)
        # Compute averages, avoiding division by zero for any category not in x
        averages = sums[categories] / counts
        return dict(zip(categories, averages))

    def map_cat_vals(self,cats2vals,new_cats):
        # Create a mapping from category IDs to indices
        unique_ids = np.unique(new_cats)
        id2index = {id_: i for i, id_ in enumerate(unique_ids)} 
        # Create an array of averages using this mapping
        averages_array = np.array([cats2vals.get(id_, 0) for id_ in unique_ids])
        # Map the averages to the new IDs in array 'a' using the mapping
        idx = np.vectorize(id2index.get)(a) 
        return averages_array[idx] 







from sklearn.linear_model import LinearRegression

class LinearTerm:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

class PolynomialTerm:
    def __init__(self, degree=2):
        self.model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


import numpy as np
from scipy.interpolate import UnivariateSpline

class UnivariateSplineTerm:
    def __init__(self, k=3, s=0):
        self.k = k
        self.s = s
        self.spline = None

    def fit(self, X, y):
        X = np.ravel(X)
        self.spline = UnivariateSpline(X, y, k=self.k, s=self.s)

    def predict(self, X):
        X = np.ravel(X)
        return self.spline(X)


import pandas as pd
import numpy as np

class RandomEffect:
    def __init__(self):
        self.group_means = {}

    def fit(self, X, y):
        # Assuming X is a DataFrame with the first column as the group identifier
        grouped = pd.DataFrame({'X': X.iloc[:, 0], 'y': y}).groupby('X')
        self.group_means = grouped.mean().to_dict()['y']

    def predict(self, X):
        # Return the group mean for each entry
        return X.iloc[:, 0].map(self.group_means).fillna(np.mean(list(self.group_means.values())))


import numpy as np
from patsy import dmatrix

class FactorSmoother:
    def __init__(self, smoother_type='cr'):
        self.smoother_type = smoother_type
        self.design_matrix = None
        self.coef_ = None

    def fit(self, X, y):
        X = np.asarray(X).ravel()
        self.design_matrix = dmatrix(f"C(X, {self.smoother_type})")
        self.coef_, _, _, _ = np.linalg.lstsq(self.design_matrix, y, rcond=None)

    def predict(self, X):
        X = np.asarray(X).ravel()
        design_matrix = dmatrix(f"C(X, {self.smoother_type})")
        return design_matrix @ self.coef_


import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class TensorProductSplines:
    def __init__(self, degree=3):
        self.degree = degree
        self.model = None

    def fit(self, X, y):
        poly = PolynomialFeatures(self.degree)
        X_poly = poly.fit_transform(X)
        self.model = LinearRegression().fit(X_poly, y)

    def predict(self, X):
        poly = PolynomialFeatures(self.degree)
        X_poly = poly.transform(X)
        return self.model.predict(X_poly)


import numpy as np
from sklearn.isotonic import IsotonicRegression

class AdaptiveSpline:
    def __init__(self):
        self.model = IsotonicRegression()

    def fit(self, X, y):
        X = np.asarray(X).ravel()
        self.model.fit(X, y)

    def predict(self, X):
        X = np.asarray(X).ravel()
        return self.model.predict(X)


import numpy as np
from scipy.interpolate import SmoothBivariateSpline

class BivariateSplineTerm:
    def __init__(self, kx=3, ky=3, s=0):
        self.kx = kx
        self.ky = ky
        self.s = s
        self.spline = None

    def fit(self, X, y):
        self.spline = SmoothBivariateSpline(X[:, 0], X[:, 1], y, kx=self.kx, ky=self.ky, s=self.s)

    def predict(self, X):
        return self.spline.ev(X[:, 0], X[:, 1])


import numpy as np
from scipy.interpolate import UnivariateSpline

class ShrinkageSmoother:
    def __init__(self, smoothing_factor=0.5):
        self.smoothing_factor = smoothing_factor
        self.spline = None

    def fit(self, X, y):
        X = np.ravel(X)
        y = np.ravel(y)
        # Adjust smoothing_factor for the amount of regularization
        self.spline = UnivariateSpline(X, y, s=self.smoothing_factor)

    def predict(self, X):
        X = np.ravel(X)
        return self.spline(X)


import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

class StructuredAdditiveRegressionTerm:
    def __init__(self, kernel=None):
        if kernel is None:
            # Default to a Radial-basis function (RBF) kernel
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1)
        self.model = GaussianProcessRegressor(kernel=kernel)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X, return_std=False)



'''

#############################################################################
#############################################################################
















# . 

