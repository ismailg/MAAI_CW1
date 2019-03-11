
'''
This code was ran on Google Colab's GPU

'''

import numpy as np
import pandas as pd
with open('/content/drive/My Drive/Multiagent/train.csv') as csv_file: 
    df_train = pd.read_csv(csv_file)

with open('/content/drive/My Drive/Multiagent/validation.csv') as csv_file: 
    df_validation = pd.read_csv(csv_file)

with open('/content/drive/My Drive/Multiagent/test.csv') as csv_file: 
    df_test = pd.read_csv(csv_file)
    
   
      
# Prepare train set
train = df_train.click.copy()
for variable in ['weekday', 'hour', 'useragent', 'region', 'adexchange', 'slotvisibility', 'slotformat', 'advertiser']:
    tmp = pd.get_dummies(df_train[variable], prefix=variable)
    train = pd.concat([train, tmp], axis=1)

tmp = pd.Series([df_train.slotheight.apply(str) + '_' + df_train.slotwidth.apply(str)])
tmp = pd.get_dummies(tmp[0], prefix='slotsize')
train = pd.concat([train, tmp], axis=1)
train.rename(columns={0:'slotsize'}, inplace=True)

train = pd.concat([train, df_train.slotprice, df_train.bidprice, df_train.payprice], axis=1)
    

# Prepare validation set
validation = df_validation.click.copy()
for variable in ['weekday', 'hour', 'useragent', 'region', 'adexchange', 'slotvisibility', 'slotformat', 'advertiser']:
    tmp = pd.get_dummies(df_validation[variable], prefix=variable)
    validation = pd.concat([validation, tmp], axis=1)

tmp = pd.Series([df_validation.slotheight.apply(str) + '_' + df_validation.slotwidth.apply(str)])
tmp = pd.get_dummies(tmp[0], prefix='slotsize')
validation = pd.concat([validation, tmp], axis=1)
validation.rename(columns={0:'slotsize'}, inplace=True)

validation = pd.concat([validation, df_validation.slotprice, df_validation.bidprice, df_validation.payprice], axis=1)
  

# Prepare test set
test = df_test.bidid.copy()
for variable in ['weekday', 'hour', 'useragent', 'region', 'adexchange', 'slotvisibility', 'slotformat', 'advertiser']:
    tmp = pd.get_dummies(df_test[variable], prefix=variable)
    test = pd.concat([test, tmp], axis=1)

tmp = pd.Series([df_test.slotheight.apply(str) + '_' + df_test.slotwidth.apply(str)])
tmp = pd.get_dummies(tmp[0], prefix='slotsize')
test = pd.concat([test, tmp], axis=1)
test.rename(columns={0:'slotsize'}, inplace=True)

test = pd.concat([test, df_test.slotprice], axis=1)  


'''

Data pre-processing (elimiating some missing variables)

'''

train = train.drop(['useragent_android_ie', 'useragent_android_maxthon', 'useragent_other_firefox',
                    'useragent_linux_ie','useragent_mac_maxthon', 'useragent_mac_sogou'], axis=1)
test = test.drop(['useragent_android_maxthon'], axis=1)
validation = validation.drop(['useragent_mac_sogou', 'useragent_mac_maxthon', 'useragent_linux_ie'], axis=1)


'''
CTR estimation
 1. Feature engineering

 1.1. Exclude certain features, see iPinYou paper (quote below)
  - exclude  Bid ID, Log Type, iPinYou ID, URL, Anonymous URL ID, Bidding Price, Paying Price, Key Page URL 
    because they are either almost unique for each case or meaningless to be added to model training

'''

X_train = train.iloc[:, 1:-2].as_matrix()
y_train = train.iloc[:, 0].as_matrix()

X_selection = validation.iloc[:, 1:-2].as_matrix()
y_selection = validation.iloc[:, 0].as_matrix()

X_test = test.iloc[:, 1:].as_matrix()

# 1.2. Reduce dimensionality of sparse input matrix
#  - ideas: PCA, autoencoder, feature ranking with recursive feature elimination (RFECV sklearn)


# 1.3. Models considered
#  a) logistic and xgboost from iPinYou paper (quote below)
#       we set the max tree depth as 5 and train 50 trees with 0.05 learning rate.
#  * PROBLEM: they don't tune the models, we improve that w/ GridSearchCV
#  * Above works faster as the single-model training on a single fold job is paralellised
#
#  b) iPinYou paper uses xgboost, we can use catboost since it takes care of categorical & continuous inputs
#  We don't need to preprocess data like in the paper. See https://github.com/catboost/catboost



'''
 2. Training and Tuning models on the training set. The following models are tried:
     a) logistic regression
     b) Neural network
     c) CatBoost random forest
     d) CatBoost with log-loss weighting of the minority class
     e) Stacking the Neural Net and CatBoost with another CatBoost Random Forest meta-learner

'''

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier

# CV folds and tol
cv = 2
tol = 1e-4


# a) Logistic regression
param_grid_log = {'loss': ['log'], 'penalty': ['l1', 'l2'],
                  'alpha': [10 ** x for x in range(-4, 3)]
                  }
clf_log = SGDClassifier(random_state=0, class_weight='balanced', n_jobs=-1, tol=tol)
clf_grid_log = GridSearchCV(clf_log, param_grid_log, scoring='roc_auc', cv=cv)
clf_grid_log.fit(X_train, y_train)
# best params: {'alpha': 0.001, 'loss': 'log', 'penalty': 'l1'}
  

# b) Neural Net (or MLP in sklearn)
param_grid_mlp = {'hidden_layer_sizes': [(100,200,20), (100,50, 10), (50,100,50)], 
                  'alpha': [10 ** x for x in range(-4, 3)],
                  'activation': ["relu", "tanh"]
                 }
clf_mlp = MLPClassifier(random_state=0, tol=tol,solver='adam', learning_rate='adaptive', verbose=2)
clf_grid_mlp = GridSearchCV(clf_mlp, param_grid_mlp, scoring='roc_auc', cv=cv)
clf_grid_mlp.fit(X_train, y_train)
# best params: {'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': (50, 100, 50)}    
 
   
# c) catboost
params = {
          'iterations': [50, 100, 150, 200],
          "depth":[5, 8, 10, 12, 16],
          'loss_function': ['Logloss'], 
          'verbose':[True],
          'task_type':['GPU'],
          'l2_leaf_reg': [3]
          } 
cb = CatBoostClassifier(random_state=0, verbose= True)
cb_model = GridSearchCV(cb, params, scoring="roc_auc", cv = cv)
cb_model.fit(X_train, y_train)
# best params: {'depth': 16,'iterations': 100,'l2_leaf_reg': 3,'loss_function': 'Logloss','task_type': 'GPU','verbose': True}



# d) catboost with weighting the positive class loss
# weights defined from CatBoost developer's recommandation (from training set)
scale_pos_weight = sum(train['click']==0)/sum(train['click']==1)
# define same param grid as before
params = {
          'iterations': [50, 100, 150, 200],
          "depth":[5, 8, 10, 12, 16],
          'loss_function': ['Logloss'], 
          'verbose':[True],
          'task_type':['GPU'],
          'l2_leaf_reg': [3]
          } 
cb = CatBoostClassifier(random_state=0, verbose=True, scale_pos_weight = scale_pos_weight)
cb_grid = GridSearchCV(cb, params, scoring="roc_auc", cv = cv)
cb_grid.fit(X_train, y_train)

# e) Stacking the Neural Net and CatBoost with another CatBoost Random Forest meta-learner
# Step 1: obtain pctr predictions of MLP and cb on the validation set
# Then stack them in an array
y_hat_mlp = clf_grid_mlp.predict_proba(X_selection)[:,1]
y_hat_cb = cb_model.predict_proba(X_selection)[:,1]
x_selection_pctr = np.array((y_hat_mlp, y_hat_cb)).T
# Step 2: train another CatBoost learner with the pctrs as input and the validation clicks as output
# Weight loss of pozitive class as part d) but weights from the validation set this time 
scale_pos_weight = sum(validation['click']==0)/sum(validation['click']==1)
# define same param grid as before
params = {
          'iterations': [50, 100, 150, 200],
          "depth":[5, 8, 10, 12, 16],
          'loss_function': ['Logloss'], 
          'verbose':[True],
          'task_type':['GPU'],
          'l2_leaf_reg': [3]
          } 
cb_meta = CatBoostClassifier(random_state=0, verbose=True, scale_pos_weight = scale_pos_weight)
cb_grid_meta = GridSearchCV(cb_meta, params, scoring="roc_auc", cv = cv)
cb_grid_meta.fit(x_selection_pctr, y_selection)


'''
 3. After all models are trained and tuned, evaluate them on the validation set
 This is an out-of-sample model selection method (except for the meta-learner)

'''

# some standard machine learning metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# 1. obtain all models' predictions on the validation set
y_hat_logistic = clf_grid_log.predict(X_selection)
y_hat_mlp = clf_grid_mlp.predict(X_selection)
y_hat_cb = cb_model.predict(X_selection)
y_hat_cb_weighted = cb_grid.predict(X_selection)
y_hat_stacker = cb_grid_meta.predict(X_selection)
list_y_hats = [y_hat_logistic, y_hat_mlp, y_hat_cb, y_hat_cb_weighted, y_hat_stacker]
# 2. compute the metrics
print("==== ROC-AUC scores ====")
print("all printed in order: logistic, mlp, cb, cb weighted, stacked learner")
for model in list_y_hats:
    print(roc_auc_score(y_selection, model))
    
print("==== classification reports ====")
for model in list_y_hats:
    print(classification_report(y_selection, model))

print("==== confusion matrices ====")
for model in list_y_hats:
    print(confusion_matrix(y_selection, model))

print("==== unbalanced accuracy ====")
for model in list_y_hats:
    print(accuracy_score(y_selection, model))
'''
Analysis of above results is in the report. The model chosen to perform pCTR estimation is the 
CatBoost model with unweighted loss of positive examples.

'''







'''

3. Linear and Non-linear bid estimation (in this order)

'''


# 1. LINEAR BIDDING MODEL
# Use catboost to form CTR on validation set e.g. P(click=1)
# best params: {'depth': 16,'iterations': 100,'l2_leaf_reg': 3,'loss_function': 'Logloss','task_type': 'GPU','verbose': True}
# roc-auc: 0.7027
''' 
The bid value is linearly proportional to the pCTR (predicted CTR). The formula can be generally written as

                        bid = base_bid* pCTR/avgCTR
                        
where the tuning parameter base bid is the bid price for the average CTR cases. Opti-
mise the base bid and the CTR estimation ON THE VALIDATION SET (budget not valid on training set)

'''
# import random shuffling of array
from sklearn.utils import shuffle

# compute predicted CTR on validation set p(click = 1)
p_ctr = cb_model.predict_proba(X_selection)[:, 1]
# avg CTR on training set per TA email
avg_ctr = sum(train['click'])/len(train['click']) 
budget = 6250
data_payprice = np.array(validation['payprice'].values).reshape(-1,1)

# 1.1. optimising to find base_bid
base_bid_grid = np.linspace(25, 350, 400)
best_expected_clicks = 0
trials = 500 # the number of random samples drawn from winnable ads

for base_bid in base_bid_grid:

    skip = 1
    # formula from assignament
    current_bid = (base_bid * p_ctr/avg_ctr).reshape(-1, 1)
    # ads won by bid i
    current_ads_won_bool = (current_bid-data_payprice > 0).reshape(len(data_payprice), )  
    current_clicks = validation['click'].iloc[current_ads_won_bool]
    current_cost = validation['payprice'].iloc[current_ads_won_bool]

    # if don't win any click skip base_bid guess
    if sum(current_clicks) == 0:
        continue
    
    # if can afford all winnable ads
    if sum(current_cost)/1000 <= budget:
        expected_clicks = sum(current_clicks)
        expected_cost = sum(current_cost)
        optimal_base_bid_no_clicks = base_bid
        optimal_bids_bool_no_clicks = current_ads_won_bool
        skip = 0
    
    # if can win some ads but can't afford all of them, draw random samples since we don't
    # know which ads we'll win in a competition
    if skip:
        indicator = np.array(range(len(current_clicks)))
        shuffled_clicks_won = 0
        shuffled_cost_won = 0
        shuffled_sample_size = 0
        
        for j in range(trials):
            
            random_indicator = shuffle(indicator)
            
            shuffled_clicks = current_clicks.iloc[random_indicator]
            shuffled_cost = current_cost.iloc[random_indicator]
            
            shuffled_affordability = (shuffled_cost.cumsum()/1000 <= budget).values
            
            shuffled_clicks_won += sum(shuffled_clicks.iloc[shuffled_affordability])
            shuffled_cost_won += sum(shuffled_cost.iloc[shuffled_affordability])/1000
            shuffled_sample_size += sum(shuffled_affordability)
            
        expected_clicks = shuffled_clicks_won/trials
        expected_cost = shuffled_cost_won/trials
    
    if expected_clicks > best_expected_clicks:
        best_expected_clicks = expected_clicks
        print("Best exp:", best_expected_clicks)
        best_expected_cost = expected_cost
        exp_sample_size = shuffled_sample_size/trials
        
        optimal_base_bid_no_clicks = base_bid
        optimal_bids_bool_no_clicks = current_ads_won_bool
# Linear bids COMPUTED with the optimal parameters
print(optimal_base_bid_no_clicks)
print(best_expected_cost)
print(best_expected_clicks)
optimal_linear_bid = (optimal_base_bid_no_clicks * p_ctr/avg_ctr).reshape(-1, 1)
print(best_expected_clicks/int(exp_sample_size))







# 2. NONLINEAR BIDDING MODEL
'''
ORBT1 model: Zhang et al. Optimal real-time bidding for display advertising. KDD 14

                bid = sqrt(pCTR * c/lambda +c^2) - c
                
'''
# the 2 parameter grids to optimize over
lambda_grid = np.linspace(1e-7, 1e-5, 100)
c_grid = np.linspace(50, 100, 50)
best_expected_clicks_ortb = 0
trials = 500 # the number of random samples drawn from winnable ads

for c in c_grid:
    for lmb in lambda_grid:
    
        skip = 1
        # formula from assignament
        current_bid = (np.sqrt(p_ctr * c/lmb + c*c)-c).reshape(-1, 1)
        # ads won by bid i
        current_ads_won_bool = (current_bid-data_payprice > 0).reshape(len(data_payprice), )  
        current_clicks = validation['click'].iloc[current_ads_won_bool]
        current_cost = validation['payprice'].iloc[current_ads_won_bool]
        
        # if don't win any clicks skip base_bid
        if sum(current_clicks) == 0:
            continue
        
        # if can afford all ads
        if sum(current_cost)/1000 <= budget:
            expected_clicks = sum(current_clicks)
            expected_cost = sum(current_cost)
            optimal_base_bid_no_clicks = base_bid
            optimal_bids_bool_no_clicks = current_ads_won_bool
            skip = 0
            
        # if can't afford all ads
        if skip:
            indicator = np.array(range(len(current_clicks)))
            shuffled_clicks_won = 0
            shuffled_cost_won = 0
            shuffled_sample_size = 0
            
            for j in range(trials):
                
                random_indicator = shuffle(indicator)
                
                shuffled_clicks = current_clicks.iloc[random_indicator]
                shuffled_cost = current_cost.iloc[random_indicator]
                
                shuffled_affordability = (shuffled_cost.cumsum()/1000 <= budget).values
                
                shuffled_clicks_won += sum(shuffled_clicks.iloc[shuffled_affordability])
                shuffled_cost_won += sum(shuffled_cost.iloc[shuffled_affordability])/1000
                shuffled_sample_size += sum(shuffled_affordability)
                
            expected_clicks = shuffled_clicks_won/trials
            expected_cost = shuffled_cost_won/trials
        
        if expected_clicks > best_expected_clicks_ortb:
            best_expected_clicks_ortb = expected_clicks
            print("Best exp:", best_expected_clicks_ortb)
            best_expected_cost_ortb = expected_cost
            optimal_c = c
            optimal_lmb = lmb
            optimal_bids_bool_no_clicks_ortb = current_ads_won_bool
            exp_sample_size_ortb = shuffled_sample_size/trials


print("Best exp:", best_expected_clicks_ortb)
print(            best_expected_cost_ortb/1000)
print(            optimal_c )
print(            optimal_lmb)
print(sum(best_expected_clicks_ortb)/len(current_clicks))
# Calculate the optimal bids
optimal_nonlinear_bid = (np.sqrt(p_ctr * optimal_c/optimal_lmb + optimal_c*optimal_c)-optimal_c).reshape(-1, 1)


'''

Linear vs Non-linear bidding: Figure 4 in report

'''
import matplotlib.pyplot as plt
# linear and non-linear bids from [0, 0.002] interval, 1000 points
pctr = np.linspace(0, 0.002, 1000)
lin_bids = (optimal_base_bid_no_clicks * pctr/avg_ctr).reshape(-1, 1)
non_lin_bids = (np.sqrt(pctr * 50.0/4e-06 + 50.0*50.0)-50.0).reshape(-1, 1)
# plot
plt.plot(p_ctr, lin_bids, label='Linear')
plt.plot(pctr, non_lin_bids, label='Non-linear')
plt.title("Linear vs Non-linear bidding strategies")
plt.legend()
plt.xlabel("eCTR")
plt.ylabel("bid")
plt.savefig("/content/drive/My Drive/Multiagent/comparison.jpeg")


'''

4. Export linear and non-linear bids
Use calibrated parameters from above and test pCTRs from the catboost model

'''
p_ctr_test = cb_model.predict_proba(X_test)[:, 1]
test_bids_linear = (optimal_base_bid_no_clicks * p_ctr_test/avg_ctr).reshape(-1, 1)
test_bids_nonlinear = (np.sqrt(p_ctr_test * optimal_c/optimal_lmb + optimal_c*optimal_c)-optimal_c).reshape(-1, 1)

pd.DataFrame(test_bids_linear).to_csv("/content/drive/My Drive/Multiagent/test_bids_linear.csv")
pd.DataFrame(test_bids_nonlinear).to_csv("/content/drive/My Drive/Multiagent/test_bids_nonlinear.csv")

