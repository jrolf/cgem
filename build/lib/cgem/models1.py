
#############################################################################
#############################################################################

Notes = '''

**Collaborative Generalized Effects Modeling (CGEM): A Comprehensive Overview**

### What is CGEM?

Collaborative Generalized Effects Modeling (CGEM) is an advanced statistical modeling framework that marks a significant evolution in the realm of data analysis and predictive modeling. It stands out in its ability to handle complex, real-world scenarios that are often encountered in business analytics, scientific research, and other domains where data relationships are intricate and multifaceted. CGEM's main strength lies in its innovative approach to model construction, which blends traditional statistical methods with modern machine learning techniques.

### Defining Characteristics of CGEM

1. **Formulaic Flexibility**: CGEM is characterized by its unparalleled formulaic freedom. Unlike conventional models constrained by linear or additive structures, CGEM allows for the creation of models with any mathematical form. This includes linear, non-linear, multiplicative, exponential, and more intricate relationships, providing a canvas for data scientists to model the real complexity found in data.

2. **Generalization of Effects**: In CGEM, the concept of an 'effect' is broadly defined. An effect can be as straightforward as a constant or a linear term, or as complex as the output from a machine learning algorithm like a neural network or a random forest. This generalization enables the seamless integration of diverse methodologies within a single coherent model, offering a more holistic view of the data.

3. **Iterative Convergence and Refinement**: The methodology operates through an iterative process, focusing on achieving a natural and efficient convergence of terms. This iterative refinement ensures that each effect in the model is appropriately calibrated, thus avoiding common pitfalls like overfitting or the disproportionate influence of particular variables.

4. **Causal Coherence**: CGEM places a strong emphasis on maintaining causally coherent relationships. This principle ensures that the model's outputs are not just statistically significant but also meaningful and interpretable in the context of real-world scenarios. It is a crucial aspect that distinguishes CGEM from many other data modeling approaches.

5. **Integration with Machine Learning**: Uniquely, CGEM is designed to incorporate machine learning models as effects within its framework. This integration allows for leveraging the predictive power of machine learning while maintaining the interpretability and structural integrity of traditional statistical models.

### Underlying Principles Making CGEM Uniquely Powerful

- **Versatility in Model Design**: CGEM's formulaic flexibility allows it to adapt to various data types and relationships, making it applicable in diverse fields from marketing to environmental science.

- **Holistic Data Representation**: By allowing for a wide range of effects, CGEM can represent complex datasets more completely, capturing nuances that simpler models might miss.

- **Balanced Complexity and Interpretability**: While it can incorporate complex machine learning models, CGEM also maintains a level of interpretability that is often lost in more black-box approaches.

- **Focus on Causality**: By ensuring that models are causally coherent, CGEM bridges the gap between correlation and causation, a critical factor in making sound decisions based on model outputs.

- **Adaptive Learning and Refinement**: The iterative nature of CGEM enables it to refine its parameters continually, leading to models that are both robust and finely tuned to the data.

### Conclusion

CGEM represents a significant leap in statistical modeling, offering a sophisticated, flexible, and powerful tool for understanding and predicting complex data relationships. Its unique blend of formulaic freedom, generalization of effects, and focus on causal coherence makes it an invaluable resource in the data scientist's toolkit, particularly in an era where data complexity and volume are ever-increasing.

'''

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

### FILE I/O OPERATIONS: 

def get_files(folder_path):
    """
    List all non-hidden files in a given folder.

    Parameters:
    folder_path (str): Path to the folder.

    Returns:
    list: A list of filenames in the folder.
    """
    if folder_path[-1] != '/':
        folder_path += '/'
    return [file for file in os.listdir(folder_path) if not file.startswith('.')]

def get_path(string):
    """
    Extract the path from a string representing a file or folder.

    Parameters:
    string (str): The input string.

    Returns:
    str: The extracted path.
    """
    if '/' not in string:
        return '' if '.' in string else string
    parts = string.split('/')
    if '.' not in parts[-1]:
        return string if string.endswith('/') else string + '/'
    return '/'.join(parts[:-1]) + '/'

def ensure_path(path):
    """
    Create a path if it doesn't already exist.

    Parameters:
    path (str): The path to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def read_json(filename):
    """
    Read a JSON file and return its content.

    Parameters:
    filename (str): The name of the JSON file.

    Returns:
    dict: The content of the JSON file.
    """
    with open(filename, 'r') as file:
        return json.load(file)

def write_json(filename, obj, pretty=True):
    """
    Write an object to a JSON file.

    Parameters:
    filename (str): The name of the JSON file.
    obj (object): The Python object to write.
    pretty (bool): Whether to write the JSON in a pretty format.
    """
    path = get_path(filename)
    if path:
        ensure_path(path)
    with open(filename, 'w') as file:
        if pretty:
            json.dump(obj, file, sort_keys=True, indent=2, separators=(',', ': '))
        else:
            json.dump(obj, file, sort_keys=True)

def export_model(filename, model_object):
    """
    Export a fitted model to a file.

    Parameters:
    filename (str): The name of the file to save the model to.
    model_object (object): The model object to save.
    """
    with open(filename, 'wb') as file:
        pickle.dump(model_object, file)

def import_model(filename):
    """
    Import a fitted model from a file.

    Parameters:
    filename (str): The name of the file to load the model from.

    Returns:
    object: The loaded model object.
    """
    with open(filename, 'rb') as file:
        return pickle.load(file)

def read_github_csv(csv_url):
    """
    Read a CSV file from a GitHub URL.

    Parameters:
    csv_url (str): The URL of the CSV file.

    Returns:
    DataFrame: The content of the CSV file as a pandas DataFrame.
    """
    response = requests.get(csv_url)
    return pd.read_csv(io.StringIO(response.content.decode('utf-8')))

def try_get_default(dictionary, key, default=0.0):
    """
    Try to get a value from a dictionary; return a default value if the key is not found.

    Parameters:
    dictionary (dict): The dictionary to search.
    key (object): The key to look for.
    default (object): The default value to return if the key is not found.

    Returns:
    object: The value associated with the key or the default value.
    """
    return dictionary.get(key, default)


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

#############################################################################
#############################################################################

Notes = '''

def Reform(eq_str1="y=m*x+b",solve_for='x'):
    global left,right,sleft,sright,maker_symbol,atoms,atom,EqString,resolve_str,eq2
    eq_str1 = eq_str1.replace(' ','').replace('==','=').replace('~','=') 
    left,right = tuple(eq_str1.split('='))
    sleft,sright = sympify(left,evaluate=False),sympify(right,evaluate=False) 
    atoms = list(sleft.atoms())+list(sright.atoms())
    for atom in atoms:
        try:
            maker_symbol = "ting=Symbol('ting')".replace('ting',str(atom))
            exec(maker_symbol) 
        except: pass 
    EqString = "Eq(LeftPart,RightPart)".replace('LeftPart',left).replace('RightPart',right) 
    eq1 = eval(EqString)   
    resolve_str = "Eq(var,solve(eq1,var)[0])".replace('var',solve_for) 
    eq2 = eval(resolve_str) 
    eq_str2 = str(eq2)[3:-1].replace(' ','').replace(',',' = ') 
    return eq_str2 

def GetVars(eq_str1="y=m*x+b",side='both'): 
    eq_str1 = eq_str1.replace(' ','').replace('==','=').replace('~','=')
    left,right = tuple(eq_str1.split('='))
    sleft,sright = sympify(left,evaluate=False),sympify(right,evaluate=False) 
    if side=='both': atoms = list(sleft.atoms())+list(sright.atoms())
    elif side=='right': atoms = list(sright.atoms()) 
    elif side=='left' : atoms = list(sleft.atoms())  
    found_vars = []
    for atom in atoms:
        try:
            maker_symbol = "ting=Symbol('ting')".replace('ting',str(atom))
            exec(maker_symbol) 
            found_vars.append(atom) 
        except: pass  
    found_vars = [str(a) for a in found_vars] 
    return sorted(set(found_vars)) 

eq_conv = [
    ['log(','np.log('],
    ['exp(','np.exp('],
]
# Equation String to Numpy String Conversion:
def eq2np(eq_str):
    for r in eq_conv:
        a,b = tuple(r)
        eq_str = eq_str.replace(a,b) 
    return eq_str

# Equation String to Numpy String Conversion:
def np2eq(eq_str):
    for r in eq_conv:
        a,b = tuple(r) 
        eq_str = eq_str.replace(b,a)  
    return eq_str

def EvaluationString(eq_str1="y=m*x+b",solve_for='x',dfname='RDF',tvars=[]):
    eq_str2 = Reform(eq_str1,solve_for) 
    NumpyForm = str(eq_str2).split('=')[1].strip() 
    NumpyForm = eq2np(NumpyForm) 
    for tvar in tvars: 
        new_name = "DFN['VAR']".replace('DFN',dfname).replace('VAR',tvar)
        NumpyForm = NumpyForm.replace(tvar,new_name) 
    return NumpyForm 

def Evaluation(eq_str1="y=m*x+b",solve_for='x',dfname='RDF',tvars=[]): 
    es = EvaluationString(eq_str1,solve_for,dfname,tvars=tvars)  
    return eval(es)   

'''

#############################################################################
#############################################################################
#############################################################################
#############################################################################

class CGEM:
    
    def __init__(self):
        self.df1 = None
        self.YVar = None
        self.TermList = None
        self.TrueForm = None
        self.tparams = None
        self.target_ival = None
        self.epoch_logs = [] 

    def load_df(self, df, df_name='df'):
        self.df1 = df.copy()

    def define_form(self, formula="TGT_Z = CAT_D_EFF * LIN_REG_EFF"):
        self.YVar     = self.get_vars(formula, side='left')[0]
        self.TrueForm = self.reform(formula, self.YVar)
        self.TermList = self.get_vars(formula, side='right')
        
        # Initializing the Maximum Learning Rate:
        self.TermLR = {} 
        for tvar in self.TermList: 
            self.TermLR[tvar] = 0.1   
        # Special Provision:
        self.TermLR['const'] = 0.3  
        
    def define_terms(self, terms_params):
        self.tparams = dict(terms_params)
        self.target_ival = eval(f'self.df1["{self.YVar}"].mean()')

    def reform(self, eq_str1="y=m*x+b", solve_for='x'):
        eq_str1 = eq_str1.replace(' ', '').replace('==', '=').replace('~', '=')
        left, right = tuple(eq_str1.split('='))
        sleft, sright = sympify(left, evaluate=False), sympify(right, evaluate=False)
        atoms = list(sleft.atoms()) + list(sright.atoms())
        for atom in atoms:
            try:
                exec(f"{atom}=Symbol('{atom}')")
            except:
                pass
        eq1 = eval(f"Eq({left}, {right})")
        eq2 = eval(f"Eq({solve_for}, solve(eq1, {solve_for})[0])")
        eq_str2 = str(eq2)[3:-1].replace(' ', '').replace(',', ' = ')
        return eq_str2
    
    def get_vars(self, eq_str1="y=m*x+b", side='both'):
        eq_str1 = eq_str1.replace(' ', '').replace('==', '=').replace('~', '=')
        left, right = tuple(eq_str1.split('='))
        sleft, sright = sympify(left, evaluate=False), sympify(right, evaluate=False)

        if side == 'both':    atoms = list(sleft.atoms()) + list(sright.atoms())
        elif side == 'right': atoms = list(sright.atoms())
        elif side == 'left':  atoms = list(sleft.atoms())

        # Filter out non-symbol atoms and sort them
        found_vars = sorted(str(atom) for atom in atoms if atom.is_Symbol)
        return found_vars 

    def eq2np(self, eq_str):
        eq_conv = [['log(', 'np.log('], ['exp(', 'np.exp(']]
        for a, b in eq_conv:
            eq_str = eq_str.replace(a, b)
        return eq_str

    def np2eq(self, eq_str):
        eq_conv = [['log(', 'np.log('], ['exp(', 'np.exp(']]
        for a, b in eq_conv:
            eq_str = eq_str.replace(b, a)
        return eq_str

    def evaluation_string(self, eq_str1="y=m*x+b", solve_for='x', dfname='df1', tvars=[]):
        eq_str2 = self.reform(eq_str1, solve_for)
        numpy_form = eq_str2.split('=')[1].strip()
        numpy_form = self.eq2np(numpy_form)
        for tvar in tvars:
            numpy_form = numpy_form.replace(tvar, f"{dfname}['{tvar}']")
        return numpy_form

    def evaluation(self, eq_str1="y=m*x+b", solve_for='x', dfname='df1', tvars=[]):
        es = self.evaluation_string(eq_str1, solve_for, dfname, tvars)
        return eval(es)

    def fit(self, n_epochs=50):
        # Creates the initial version of the Transient Effects DataFrame: 
        self.initialize_tdf() # << self.TDF is created.

        # Preserve the values of the Target Variable for later evaluation
        TrueVals = self.TDF[self.YVar].values 

        # Initial Evaluation of Predictions
        #preds = self.evaluation(self.TrueForm, self.YVar, 'TDF', tvars=self.TermList)
        #actuals = TrueVals
        #R2 = max(round(r2_score(actuals, preds), 5), 0.00001)

        for epoch_num in range(1,n_epochs+1):
            if epoch_num % 1 == 0:  # Adjust this condition for controlling the print frequency
                print(f"\n{'#' * 50}\nLearning Epoch: {epoch_num + 1}")

            # Initial Evaluation
            yhat1 = self.evaluation(self.TrueForm, self.YVar, 'self.TDF', tvars=self.TermList + [self.YVar])
            rmse1 = self.calc_rmse(TrueVals, yhat1)
            rsq1  = self.calc_r2(TrueVals, yhat1)

            model_log = {}
            NewEffects = {}
            for tvar in self.TermList: 
                self.term_tdf1 = self.TDF[[self.YVar] + self.TermList].copy()
                self.term_tdf2 = self.term_tdf1.copy()

                # Old Effects
                old_effects = self.term_tdf1[tvar].values
                
                # Implied Effects
                implied_effects = self.evaluation(self.TrueForm, tvar, 'self.term_tdf1', tvars=self.TermList + [self.YVar])
                
                # Fit a new model
                y = implied_effects
                X = self.df1[self.tparams[tvar]['xvars']].values
                model = eval(self.tparams[tvar]['model'])
                model.fit(X, y)

                # Predict new effects
                new_effects = model.predict(X)
                self.term_tdf2[tvar] = new_effects

                # Evaluate performance after learning new effects
                yhat2 = self.evaluation(self.TrueForm, self.YVar, 'self.term_tdf2', tvars=self.TermList + [self.YVar])
                rmse2 = self.calc_rmse(TrueVals, yhat2) 
                rsq2 = self.calc_r2(TrueVals, yhat2)

                # Update effects
                LRate = self.TermLR[tvar]
                deltas = new_effects - old_effects
                learned_effects = old_effects + (LRate * deltas)
                NewEffects[tvar] = learned_effects

                model_log[tvar] = {
                    'm_str':self.tparams[tvar]['model'], 
                    'xvars':self.tparams[tvar]['xvars'],
                    'model':model, 
                    'LRate':LRate, 
                    'rmse1':rmse1,
                    'rmse2':rmse2,
                    'rsq1' :rsq1 ,
                    'rsq2' :rsq2 ,
                }

            # Update TDF with new effects
            for tvar in self.TermList:
                self.TDF[tvar] = NewEffects[tvar]

            # Final evaluation for this iteration
            yhat2 = self.evaluation(self.TrueForm, self.YVar, 'self.TDF', tvars=self.TermList + [self.YVar])
            rmse2 = self.calc_rmse(TrueVals, yhat2) 
            rsq2  = self.calc_r2(TrueVals, yhat2)

            elog = {
                'epoch' : epoch_num,
                'models': model_log,  
            }
            self.epoch_logs.append(elog) 

            if epoch_num % 1 == 0:  # Adjust this condition for controlling the print frequency
                print(f"{'-' * 50}\nRMSE 1: {rmse1}\nRMSE 2: {rmse2}\nDELTA: {rmse2 - rmse1}")
                print(f"RSQ 1: {rsq1}\nRSQ 2: {rsq2}\nDELTA: {rsq2 - rsq1}\n{'-' * 50}")

        print('Done.')

    def initialize_tdf(self):
        """
        Initialize the Transient DataFrame (TDF) that holds all the currently learned effect values.
        """
        self.RDF = pd.DataFrame()
        self.RDF[self.YVar] = self.df1[self.YVar].values

        for term in self.TermList:
            if term==self.YVar: continue
            
            form2 = str(self.TrueForm)
            for term2 in self.TermList:
                if term2 == term or term2 == self.YVar: continue
                null_val = self.tparams[term2]['ival']
                form2 = form2.replace(term2, str(null_val))

            self.form3 = self.reform(form2, term)
            self.form3 = self.eq2np(self.form3).replace(term, 'term_vals')

            yvar_vals = f"self.df1['{self.YVar}'].values"
            self.form3 = self.form3.replace(self.YVar, yvar_vals) 
            expr = self.form3.split(' = ')[1]

            #exec(self.form3) 
            term_vals = eval(expr) 
            term_vals = list(term_vals)
            for _ in range(5): shuffle(term_vals)
            self.RDF[term] = term_vals

            # Special Provision
            if term == 'const': self.RDF[term] = 1.0
            else: self.RDF[term] = self.tparams[term]['ival']

        if 'const' in self.TermList: self.RDF['const'] = self.RDF['const'].mean()

        self.TDF = self.RDF.copy() 
        #print('Done Initializing Effects.') 

    def predict(self, X):
        """
        Predict using the CGEM model.

        Parameters:
        X (pandas.DataFrame): Input features for making predictions.

        Returns:
        numpy.ndarray: Predicted values.
        """
        # Create a DataFrame for storing the predictions:
        self.PDF = X.copy() 
        self.last_log = self.epoch_logs[-1]
        
        NewEffects = {}
        # Apply the learned effects to the prediction DataFrame
        for term in self.TermList:
            if term == self.YVar or term == 'const': continue            
            # Load the last available effects model for the given term: 
            self.last_model = deepcopy(self.last_log['models'][term]['model'])
            # Predict new effects
            self.X2 = X[self.tparams[term]['xvars']].values
            pred_effects = self.last_model.predict(self.X2)
            self.PDF[term] = pred_effects
            
        yhat2 = self.evaluation(
            self.TrueForm,
            self.YVar,
            'self.PDF',
            tvars=self.TermList+[self.YVar]
        )
        return yhat2

    def calc_r2(self,actual, predictions):
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

    def calc_rmse(self,actual, predictions):
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



Notes = '''

#------------------------------------------------

Formula = "TGT_Z = CAT_D_EFF * LIN_REG_EFF"

# Terms Model Parameters:
tparams = {
    "CAT_D_EFF": {
        'model': "CatRegModel()", 
        'xvars': ['CAT_D'],
        'ival' : 10,
    },
    "LIN_REG_EFF": {
        'model': "OLS()", 
        'xvars': ['REG_A','REG_B','REG_C'],
        'ival' : 10,
    } 
}   

#------------------------------------------------

model = CGEM() 
model.load_df(DF1)  
model.define_form(Formula) 
model.define_terms(tparams)  

model.fit(25);

preds = model.predict(DF2) 
actuals = DF2['TGT_Z'].values
r2 = model.calc_r2(actuals,preds)  
print('CrosVal R-Squared:',round(r2,5)) 

#------------------------------------------------

'''


#############################################################################
#############################################################################
#############################################################################
#############################################################################

Notes = '''

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







