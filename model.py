
import pickle
import os
import optuna
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, confusion_matrix, accuracy_score, classification_report, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel, RFECV, mutual_info_classif, VarianceThreshold
os.chdir('D:\Shrey\Python\SPYDER\Datasets')

data = pd.read_csv('income(1).csv', na_values = [' ?'])
data

data1 = data.copy(deep = True)
data1

data1.drop_duplicates(keep = 'first', inplace = True)

r_list = ['race', 'nativecountry']
data1=data1.drop(r_list, axis = 1)

v1 = data1['SalStat'].unique()[0]
v2 = data1['SalStat'].unique()[1]
data1['SalStat'] = data1['SalStat'].map({v1:0, v2:1})

y = data1['SalStat']
data1.drop(['SalStat'], axis = 1, inplace = True)

num_cols = [cname for cname in data1.columns if data1[cname].dtype in ['int64', 'float64']]
cat_cols = [cname for cname in data1.columns if data1[cname].dtype == 'object']

scaler = StandardScaler(with_mean=False)

num_trans = Pipeline(steps = [('impute', SimpleImputer(strategy = 'mean')),
                              ('scale', scaler)])

cat_trans = Pipeline(steps = [('impute', SimpleImputer(strategy = 'most_frequent')),
                              ('onehot', OneHotEncoder(handle_unknown = 'ignore')),
                              ('scale', scaler)])

preproc = ColumnTransformer(transformers = [('num', num_trans, num_cols),
                                            ('cat', cat_trans, cat_cols)])

xgbc_model = XGBClassifier(random_state = 69)

xgbc_pipe = Pipeline(steps = [('preproc', preproc), ('model', xgbc_model)])

train_x, test_x, train_y, test_y = train_test_split(data1, y, test_size = 0.2, random_state = 69)


#OPTUNA NOT EFFECTIVE! 
def objective(trial):
    model__learning_rate = trial.suggest_float('model__learning_rate', 0.001, 0.01)
    model__n_estimators = trial.suggest_int('model__n_estimators', 10, 500)
    model__sub_sample = trial.suggest_float('model__sub_sample', 0.0, 1.0)
    model__max_depth = trial.suggest_int('model__max_depth', 1, 20)
    
    params = {'model__max_depth' : model__max_depth,
              'model__n_estimators' : model__n_estimators,
              'model__sub_sample' : model__sub_sample,
              'model__learning_rate' : model__learning_rate}
    
    xgbc_pipe.set_params(**params)
    
    return np.mean(-1 * cross_val_score(xgbc_pipe, train_x, train_y,
                                        cv = 5, n_jobs = -1, scoring = 'neg_mean_squared_error'))

xgbc_study = optuna.create_study(direction = 'minimize')
xgbc_study.optimize(objective, n_trials = 10)

xgbc_pipe.set_params(**xgbc_study.best_params)
xgbc_pipe.fit(data1, y)


pickle.dump(xgbc_pipe, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))