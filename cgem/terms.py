
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


















#############################################################################
#############################################################################





