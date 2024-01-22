
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

from .terms  import * 

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

    def load_df(self, df):
        self.df1 = df.copy()
        self.train_len = len(df)

    def define_form(self, formula="TGT_Z = CAT_D_EFF * LIN_REG_EFF"):
        self.YVar     = self.get_vars(formula, side='left')[0]
        self.TrueForm = self.reform(formula, self.YVar)
        self.TermList = self.get_vars(formula, side='right')
        # Initializing the Maximum Learning Rate:
        self.TermLR = {} 
        for term in self.TermList: 
            self.TermLR[term] = 0.1   
        
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
        
        #---------------------------------------------------------
        self.left, self.right = left, right
        self.sleft, self.sright = sleft, sright
        self.eq1 = eq1 
        #---------------------------------------------------------
        
        eq2 = eval(f"Eq({solve_for}, solve(eq1, {solve_for})[0])")
        eq_str2 = str(eq2)[3:-1].replace(' ', '').replace(',', ' = ')
        
        #---------------------------------------------------------
        self.last_left, self.last_right = left, right
        self.last_sleft, self.last_sright = sleft, sright
        self.last_eq1 = eq1 
        self.last_eq2 = eq2
        self.last_eq_str2 = eq_str2
        #---------------------------------------------------------
        
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
        self.es = es
        return eval(es) 

    def fit(self, n_epochs=50,verbose=False):
        # Creates the initial version of the Transient Effects DataFrame: 
        self.initialize_tdf() # << self.TDF is created.
        self.FunctionalTerms = [a for a in self.TermList if "DirectVar" not in tparams[a]['model']]  
        self.DirectVarTerms  = [a for a in self.TermList if "DirectVar" in tparams[a]['model']]  

        # Preserve the values of the Target Variable for later evaluation
        TrueVals = self.TDF[self.YVar].values 
        
        for epoch_num in range(1,n_epochs+1):
            if verbose==True and epoch_num % 1 == 0:  # Adjust this condition for controlling the print frequency
                print(f"\n{'#' * 50}\nLearning Epoch: {epoch_num}")

            # Initial Evaluation
            yhat1 = self.evaluation(self.TrueForm, self.YVar, 'self.TDF', tvars=self.TermList + [self.YVar])
            rmse1 = self.calc_rmse(TrueVals, yhat1)
            rsq1  = self.calc_r2(TrueVals, yhat1)

            model_log = {}
            NewEffects = {}
            
            for term in self.FunctionalTerms: 
                
                if 'lr' in self.tparams[term]: 
                    self.TermLR[term] = self.tparams[term]['lr'] 
                
                self.term_tdf1 = self.TDF[[self.YVar] + self.TermList].copy()
                self.term_tdf2 = self.term_tdf1.copy() 

                # Old Effects
                old_effects = self.term_tdf1[term].values 
                
                # Implied Effects
                implied_effects = self.evaluation(self.TrueForm, term, 'self.term_tdf1', tvars=self.TermList + [self.YVar])
                
                # Fit a new model
                y = implied_effects
                xvars = [xvar for xvar in self.tparams[term]['xvars'] if xvar!=''] 
                if len(xvars)>0: X = self.df1[xvars].values 
                else: X = np.ones(self.train_len) 
                model = eval(self.tparams[term]['model'])
                model.fit(X, y) 

                # Predict new effects
                new_effects = model.predict(X) 
                self.new_effects = new_effects      ## DEBUG
                self.term_tdf2[term] = new_effects

                # Evaluate performance after learning new effects
                yhat2 = self.evaluation(self.TrueForm, self.YVar, 'self.term_tdf2', tvars=self.TermList + [self.YVar])
                rmse2 = self.calc_rmse(TrueVals, yhat2) 
                rsq2 = self.calc_r2(TrueVals, yhat2)

                # Update effects
                LRate = self.TermLR[term]
                deltas = new_effects - old_effects
                learned_effects = old_effects + (LRate * deltas)
                NewEffects[term] = learned_effects

                model_log[term] = {
                    'm_str':self.tparams[term]['model'], 
                    'xvars':self.tparams[term]['xvars'],
                    'model':model, 
                    'LRate':LRate, 
                    'rmse1':rmse1,
                    'rmse2':rmse2,
                    'rsq1' :rsq1 ,
                    'rsq2' :rsq2 ,
                }

            # Update TDF with new effects
            for term in list(NewEffects): 
                self.TDF[term] = NewEffects[term]

            # Final evaluation for this iteration
            yhat2 = self.evaluation(self.TrueForm, self.YVar, 'self.TDF', tvars=self.TermList + [self.YVar])
            rmse2 = self.calc_rmse(TrueVals, yhat2) 
            rsq2  = self.calc_r2(TrueVals, yhat2)

            elog = {
                'epoch' : epoch_num,
                'models': model_log,  
            }
            self.epoch_logs.append(elog) 

            if verbose==True and epoch_num % 1 == 0:  # Adjust this condition for controlling the print frequency
                print(f"{'-' * 50}\nRMSE 1: {rmse1}\nRMSE 2: {rmse2}\nDELTA: {rmse2 - rmse1}")
                print(f"RSQ 1: {rsq1}\nRSQ 2: {rsq2}\nDELTA: {rsq2 - rsq1}\n{'-' * 50}")

        print('CGEM model fitting complete. ('+str(epoch_num)+' epochs)')  
    
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
        self.pred_len = len(X) 
        
        for term in self.DirectVarTerms:
            var = tparams[term]['xvars'][0] 
            self.PDF[term] = X[var].values 
        
        # Apply the learned effects to the prediction DataFrame
        for term in self.FunctionalTerms:
            if term == self.YVar: continue           
            # Load the last available effects model for the given term: 
            self.last_model = deepcopy(self.last_log['models'][term]['model'])
            
            if "ScalarTerm" in str(self.last_model):
                self.PDF[term] = self.last_model.scalar 
                continue 

            # Predict new effects
            #----------------------------------------------------------
            xvars = [xvar for xvar in self.tparams[term]['xvars'] if xvar!=''] 
            if len(xvars)>0: self.X2 = X[xvars].values 
            else: self.X2 = np.array([np.ones(self.pred_len)]) 
            #----------------------------------------------------------
            pred_effects = self.last_model.predict(self.X2)
            self.pred_effects = pred_effects
            self.PDF[term] = pred_effects
            
        yhat2 = self.evaluation(
            self.TrueForm,
            self.YVar,
            'self.PDF',
            tvars=self.TermList+[self.YVar]
        )
        return yhat2
    
    
    def initialize_tdf(self):
        """
        Initialize the Transient DataFrame (TDF) that holds all the currently learned effect values.
        """
        
        #self.RDF = pd.DataFrame()
        #self.RDF[self.YVar] = self.df1[self.YVar].values
        
        self.RDF = self.df1.copy()
        for term in self.TermList:
            try: self.rdf[term] = self.tparams[term]['ival'] 
            except: pass
            if "DirectVar" in tparams[term]['model']:
                var = tparams[term]['xvars'][0] 
                self.RDF[term] = self.df1[var] 
            else:
                ival = tparams[term]['ival']
                self.RDF[term] = ival
        
        TermValsDict = dict() 
        for term in self.TermList:
            if term==self.YVar: continue
            if "DirectVar" in tparams[term]['model']: continue 
            self.rdf = self.RDF.copy() 
            form2 = str(self.TrueForm) 
            
            #--------------------------------
            self.term = term 
            self.form2 = form2
            #--------------------------------
            
            self.form3a = self.reform(form2, term)
            self.form3 = self.eq2np(self.form3a).replace(term, 'term_vals')
            
            #--------------------------------
            self.last_term = term 
            self.last_form2 = form2
            #--------------------------------
            
            df_vals_base = "self.rdf['TERM'].values"
            
            df_vals = df_vals_base.replace('TERM',self.YVar) 
            self.form3 = self.form3.replace(self.YVar, df_vals) 
            for term2 in self.TermList:
                if term2 not in self.form3: continue
                df_vals = df_vals_base.replace('TERM',term2) 
                self.form3 = self.form3.replace(term2, df_vals) 
            
            expr = self.form3.split(' = ')[1]
            self.expr = expr
            
            term_vals = eval(expr) 
            term_vals = list(term_vals)
            TermValsDict[term] = term_vals
            
        for term in list(TermValsDict):
            self.RDF[term] = TermValsDict[term] 
            
        self.TDF = self.RDF.copy() 
        

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




#############################################################################



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
























# .

