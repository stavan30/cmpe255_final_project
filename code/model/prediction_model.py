import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor


# Reading the training data
train_sf = pd.read_csv('data/train_main_sf.csv').values
train_mv = pd.read_csv('data/train_main_mv.csv').values
train_sj = pd.read_csv('data/train_main_sj.csv').values
train_re = pd.read_csv('data/train_main_re.csv').values


# Convert all column datatype to float
train_sf = train_sf.astype(float)
train_mv = train_mv.astype(float)
train_sj = train_sj.astype(float)
train_re = train_re.astype(float)


# Loading the labels from csv
labels_sf = pd.read_csv('data/labels_sf.csv')
labels_mv = pd.read_csv('data/labels_mv.csv')
labels_sj = pd.read_csv('data/labels_sj.csv')
labels_re = pd.read_csv('data/labels_re.csv')

labels_sf = labels_sf.values.reshape(-1)
labels_mv = labels_mv.values.reshape(-1)
labels_sj = labels_sj.values.reshape(-1)
labels_re = labels_re.values.reshape(-1)

# SF model
pipe_sf = Pipeline([('scalar', MinMaxScaler()), ('model',
                                                 GradientBoostingRegressor(n_estimators=50, random_state=42))])
pipe_sf.fit(train_sf, labels_sf)
with open('models/sf.pkl', 'wb') as f:
    pickle.dump(pipe_sf, f)

# MV model
pipe_mv = Pipeline([('scalar', MinMaxScaler()), ('model',
                                                 GradientBoostingRegressor(n_estimators=50, random_state=42))])
pipe_mv.fit(train_mv, labels_mv)
with open('models/mv.pkl', 'wb') as f:
    pickle.dump(pipe_mv, f)

# SJ model
pipe_sj = Pipeline([('scalar', MinMaxScaler()), ('model',
                                                 GradientBoostingRegressor(n_estimators=50, random_state=42))])
pipe_sj.fit(train_sj, labels_sj)
with open('models/sj.pkl', 'wb') as f:
    pickle.dump(pipe_sj, f)

# RE model
pipe_re = Pipeline([('scalar', MinMaxScaler()), ('model',
                                                 GradientBoostingRegressor(n_estimators=50, random_state=42))])
pipe_re.fit(train_re, labels_re)
with open('models/re.pkl', 'wb') as f:
    pickle.dump(pipe_re, f)
