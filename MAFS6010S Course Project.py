# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 23:40:28 2020

@author: Kwanyuen
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier,VotingClassifier
from IPython.display import display
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import re
from sklearn.metrics import precision_recall_fscore_support as score
import lightgbm as lgb
import datetime 
import math 
import gc 
from sklearn.pipeline import Pipeline 
import graphviz
import itertools
import random 
import warnings
warnings.filterwarnings('ignore')

path=r'D:/FM/2020Spring/machine learning/proj/kkbox-music-recommendation-challenge/'

# =============================================================================
# 1. Data Description
# =============================================================================
train = pd.read_csv(path+'train.csv')
members = pd.read_csv(path+'members.csv')
songs = pd.read_csv(path+'songs.csv')
song_extra_info = pd.read_csv(path+'song_extra_info.csv')
#test = pd.read_csv(path+'test.csv')
#balance
np.mean(train.target)
#NA
def na_num(df):
    na_info=pd.DataFrame(df.isnull().astype('int').sum(axis=0),columns=["NA_num"])
    na_info['Percentage']=df.isnull().astype('int').sum(axis=0)*100/len(df)
    return na_info

na_num(train)
na_num(members)
na_num(songs)
na_num(song_extra_info)

#distribution
def cat_distr(data,feature):
    cat_distr=data.groupby(feature).size()
    cat_distr = pd.DataFrame(cat_distr)
    cat_distr.reset_index(level=0, inplace=True)
    cat_distr.columns = [feature, 'Count']
    cat_distr = cat_distr.sort_values(by='Count', ascending=False)
    plt.rcParams["axes.labelsize"] = 15
    ax = sns.catplot(x=feature, y='Count', kind='bar',
                 data=cat_distr, height=6, palette='Blues_d', aspect=1.5)
    ax.fig.suptitle('Distribution of '+feature, fontsize=15)
    ax.fig.subplots_adjust(top=.9)
    for ax in ax.axes.flat:
        plt.setp(ax.get_xticklabels(), rotation=90)
    return cat_distr

cat_distr(train, 'source_system_tab')
cat_distr(train, 'source_screen_name')
cat_distr(train, 'source_type')
cat_distr(members, 'city')
cat_distr(members, 'gender')
cat_distr(members, 'registered_via')
cat_distr(songs, 'language')

def value_distr(data,feature):
    df = data[feature].value_counts()
    df.columns = [feature, 'Count']
    plt.subplots(figsize=(10, 7))
    plt.rcParams["axes.labelsize"] = 15
    ax = sns.lineplot(data=df, color='blue', size=20)
    ax.set(xlabel=feature, ylabel='Count')
    ax.set_title('Distribution of '+ feature, fontsize=15)
    plt.show()
    return df
    
value_distr(members,'registration_init_time')
value_distr(members,'expiration_date')

length = songs.song_length/60000
ax = sns.distplot(length, color='blue')
ax.set_title('Distribution of song_length', fontsize=15)
ax.set(xlabel='song_length (min)', ylabel='Count')
plt.show()

# =============================================================================
# 2. Data Processing
# =============================================================================
#type transfer
train.dtypes
members.dtypes
songs.dtypes
song_extra_info.dtypes

train['target']=train['target'].astype(np.uint8)
members['bd']=members['bd'].astype(np.uint32)
members['city']=members['city'].astype('object')
members['registered_via']=members['registered_via'].astype('object')

def to_cat(data):
    object_cols = list(data.select_dtypes(include=['object']).columns)
    for col in object_cols:
        data[col]=data[col].astype('category')
to_cat(train)
to_cat(songs)
to_cat(members)

# extracting date_member
members['registration_init_time'] = pd.to_datetime(members['registration_init_time'], errors='coerce')
members['expiration_date'] = pd.to_datetime(members['expiration_date'], errors='coerce')

members['membership_days'] = members['expiration_date'].subtract(members['registration_init_time']).dt.days.astype(int)
members['registration_year'] = members['registration_init_time'].dt.year
members['registration_month'] = members['registration_init_time'].dt.month
members['registration_date'] = members['registration_init_time'].dt.day
members['expiration_year'] = members['expiration_date'].dt.year
members['expiration_month'] = members['expiration_date'].dt.month
members['expiration_date'] = members['expiration_date'].dt.day
members = members.drop(['registration_init_time'], axis=1)

# extracting date_song_extra_info
def year_from_isrc(data):
    if type(data) == str:
        if int(data[5:7]) > 17:
            return 1900 + int(data[5:7])
        else:
            return 2000 + int(data[5:7])
    else:
        return np.nan
           
song_extra_info['song_year'] = song_extra_info['isrc'].apply(year_from_isrc)
song_extra_info.drop(['isrc', 'name'], axis = 1, inplace = True)

# merge dataset
train = train.merge(songs, on='song_id', how='left')
train = train.merge(members, on='msno', how='left')
train.dtypes
train['msno']=train['msno'].astype('category')
train = train.merge(song_extra_info, on = 'song_id', how = 'left')

train.song_length.fillna(200000,inplace=True)
train['song_length']=train['song_length'].astype(np.uint32)
train['song_id']=train['song_id'].astype('category')

## feature generating 
# counting genre_id
def genre_id_num(data):
    if data == 'no_genre_id':
        return 0
    else:
        return data.count('|') + 1

train['genre_ids']=train['genre_ids'].cat.add_categories('no_genre_id').fillna('no_genre_id')
train['genre_ids_num'] = train['genre_ids'].apply(genre_id_num).astype(np.int8)


# splitting the lyricists by ['|', '/', '\\', ';'] and counting the number of Lyricists
def lyricist_num(data):
    if data == 'no_lyricist':
        return 0
    else:
        return sum(map(data.count, ['|', '/', '\\', ';'])) + 1
    return sum(map(data.count, ['|', '/', '\\', ';']))

train['lyricist']=train['lyricist'].cat.add_categories('no_lyricist').fillna('no_lyricist')
train['lyricists_num'] = train['lyricist'].apply(lyricist_num).astype(np.int8)



# Splitting the comoser by ['|', '/', '\\', ';'] and counting the number of Lyricists
def composer_num(data):
    if data == 'no_composer':
        return 0
    else:
        return sum(map(data.count, ['|', '/', '\\', ';'])) + 1

train['composer']=train['composer'].cat.add_categories('no_composer').fillna('no_composer')
train['composer_num'] = train['composer'].apply(composer_num).astype(np.int8)


# Checking for feat in the column value
def is_featured(data):
    if 'feat' in str(data) :
        return 1
    return 0

train['artist_name']=train['artist_name'].cat.add_categories('no_artist').fillna('no_artist')
train['is_featured'] = train['artist_name'].apply(is_featured).astype(np.int8)


# Splitting the artists by [and, ',', feat, &] and counting the number of artists
def artist_num(data):
    if data == 'no_artist':
        return 0
    else:
        return data.count('and') + data.count(',') + data.count('feat') + data.count('&')

train['artist_num'] = train['artist_name'].apply(artist_num).astype(np.int8)


def short_song(data):
    if data < _mean_song_length:
        return 1
    return 0
_mean_song_length = np.mean(train['song_length'])
train['short_song'] = train['song_length'].apply(short_song).astype(np.int8)



# number of times a song has been played before
_dict_count_song_played_train = {k: v for k, v in train['song_id'].value_counts().iteritems()}
def song_played_num(data):
    try:
        return _dict_count_song_played_train[data]
    except KeyError:
        try:
            return _dict_count_song_played_test[data]
        except KeyError:
            return 0
    
train['song_played_num'] = train['song_id'].apply(song_played_num).astype(np.int64)


# number of times the artist has been played
_dict_count_artist_played_train = {k: v for k, v in train['artist_name'].value_counts().iteritems()}
def artist_played_num(data):
    try:
        return _dict_count_artist_played_train[data]
    except KeyError:
        try:
            return _dict_count_artist_played_test[data]
        except KeyError:
            return 0

train['artist_played_num'] = train['artist_name'].apply(artist_played_num).astype(np.int64)

train.to_csv(path+"processed_train1.csv")


# =============================================================================
# 3.Model Building
# =============================================================================
train1 = pd.read_csv(path+"processed_train1.csv")
train1 =train1.drop(train1.columns[0],axis=1)   
to_cat(train1)
train1.dtypes

y_train = train1['target'].values
X_train = train1.drop(['target'], axis=1)


X_train_all, X_test, y_train_all, y_test = train_test_split(X_train, y_train, test_size=0.3)
X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all,test_size=0.3)


# Sampled train data
idx = random.sample(range(0,X_train.shape[0]), 500000)
y_train = pd.DataFrame(y_train)
X_train_sampled = X_train.iloc[idx]
y_train_sampled = y_train.iloc[idx]


#Fitting a model on sampled data
model_feature = lgb.sklearn.LGBMClassifier(
         objective='binary',
         eval_metric='binary_logloss',
         boosting='gbdt',
         learning_rate=0.3 ,
         verbose=0,
         num_leaves=600,
         bagging_freq= 1,
         feature_fraction= 0.9,
         max_bin= 256,
         max_depth= 300,
         num_rounds= 200,
)


model_feature.fit(X_train_sampled, y_train_sampled)


predicted = model_feature.predict(X_val)
accuracy = accuracy_score(y_val, predicted)
print(f'Mean accuracy score: {accuracy:.3}')


### Feature selection

def lgb_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)

fi = lgb_feat_importance(model_feature, X_train_sampled )
print(fi[:10])

def plot_fi(fi):
    return fi.plot('cols', 'imp', 'barh',figsize=(12,7), legend=False)
plot_fi(fi[:30])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# =============================================================================
# # Pipeline and grid search
# pipe_dt = Pipeline([('clf', lgb.sklearn.LGBMClassifier())]) # Estimator: ML algorithm
# 
# grid_params = dict(
#          clf__objective=['binary'],
#          clf__eval_metric=['binary_logloss'],
#          clf__boosting_type=['gbdt'],
#          clf__learning_rate=[0.3] ,
#          clf__verbose=[0],
#          clf__num_leaves=[300, 350, 250],
#          clf__feature_fraction= [0.9],
#          clf__max_bin= [256],
#          clf__max_depth= [200, 150, 250],
#          clf__num_rounds= [200])
# gs = GridSearchCV(estimator=pipe_dt,  
#                   param_grid=grid_params,
#                   scoring='accuracy',
#                   cv=5)
# 
# gs.fit(X_train_sampled, y_train_sampled)
# f"{gs.score(X_test, y_test):.4f}"
# 
# # Best algorithm with best hyperparameters 
# # (need to fit it to find specific model parameters)
# print(gs.best_estimator_)
# 
# # Best model with specific model parameters
# gs.best_estimator_.get_params()['clf']
# =============================================================================

#Fitting the best model on all the data
model = lgb.sklearn.LGBMClassifier(objective='binary',
                                         eval_metric='binary_logloss',
                                         boosting='gbdt',
                                         learning_rate=0.3 ,
                                         verbose=0,
                                         num_leaves=600,
                                         bagging_freq= 1,
                                         feature_fraction= 0.9,
                                         max_bin= 256,
                                         max_depth= 300,
                                         num_rounds= 200)

# =============================================================================
# 4.Performance
# =============================================================================
#Validation accuracy
model.fit(X_train, y_train)
predicted = model.predict(X_val)
accuracy = accuracy_score(y_val, predicted)
print(f'Mean accuracy score on validation: {accuracy:.4}')

precision, recall, fscore, support = score(y_val, predicted)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))
  


# Test accuracy
predicted1 = model.predict(X_test)
prob =model.predict_proba(X_test)
accuracy = accuracy_score(y_test, predicted1)
print(f'Mean accuracy score on validation: {accuracy:.4}')

precision, recall, fscore, support = score(y_test, predicted1)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))



def plot_cmf_matrix(y_test, y_pred, title, classes):
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    fig.savefig(title + '.png')
    plt.show()

classes = ['0', '1']
title = 'Matrix'
plot_cmf_matrix(y_test,predicted1, title, classes)


def plot_ROC(y_test, y_pred):
    false_positive_rate, recall, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, recall)
    fig = plt.gcf()
    plt.title('Receiver Operating Characteristic', fontsize=15)
    plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)  # AUC
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('Recall', fontsize=15)
    plt.xlabel('Fall-out', fontsize=15)
    fig.set_size_inches(12, 7)
    plt.show()
    fig.savefig('ROC.png', dpi=100)
    print(roc_auc)
plot_ROC(y_test, prob[:,1])
