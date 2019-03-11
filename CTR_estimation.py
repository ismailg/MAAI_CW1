'''

TIM DATA LOADING        
                        
'''

import numpy as np
import pandas as pd
with open('./data/train.csv') as csv_file: 
    df_train = pd.read_csv(csv_file)

with open('./data/validation.csv') as csv_file: 
    df_validation = pd.read_csv(csv_file)

with open('./data/test.csv') as csv_file: 
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

Notes on above:
    
Some vars from train missing in validation (see below)

useragent_android_ie
useragent_android_maxthon
useragent_other_firefox
                        
'''

for i in train.keys():
    if i not in validation.keys(): 
        print(i)

train = train.drop(['useragent_android_ie', 'useragent_android_maxthon', 'useragent_other_firefox'], axis=1)

# delete useless df and release ram
import gc
gc.collect()
del [[tmp, df_train, df_validation, df_test]]
gc.collect()


'''

CTR estimation

'''
# 1. Feature engineering

# 1.1. Exclude certain features, see iPinYou paper (quote below)
#  - exclude  Bid ID, Log Type, iPinYou ID, URL, Anonymous URL ID, Bidding Price, Paying Price, Key Page URL 
#    because they are either almost unique for each case or meaningless to be added to model training
X_train = train.iloc[:, 1:-2].as_matrix()
y_train = train.iloc[:, 0].as_matrix()

X_selection = validation.iloc[:, 1:-2].as_matrix()
y_selection = validation.iloc[:, 0].as_matrix()
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


# Save model to disk:
import pickle
# load the model from disk. Example:
#filename = 'logistic.sav'
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier


# CV folds
cv = 2
tol = 1e-4


# Logistic
param_grid_log = {'loss': ['log'], 'penalty': ['l1', 'l2'],
                  'alpha': [10 ** x for x in range(-4, 3)]
                  }
clf_log = SGDClassifier(random_state=0, class_weight='balanced', n_jobs=-1, tol=tol)
clf_grid_log = GridSearchCV(clf_log, param_grid_log, scoring='roc_auc', cv=cv)
clf_grid_log.fit(X_train, y_train)
# save model to disk
filename = 'logistic.sav'
pickle.dump(clf_grid_log, open(filename, 'wb'))
# best params: {'alpha': 0.001, 'loss': 'log', 'penalty': 'l1'}
# roc auc: 0.6186
  
# MLP
param_grid_mlp = {'hidden_layer_sizes': [(100,200,20), (100,50, 10), (50,100,50)], 
                  'alpha': [10 ** x for x in range(-4, 3)],
                  'activation': ["relu"]
                 }
clf_mlp = MLPClassifier(random_state=0, tol=tol,solver='adam', learning_rate='adaptive', verbose=2)
clf_grid_mlp = GridSearchCV(clf_mlp, param_grid_mlp, scoring='roc_auc', cv=cv)
clf_grid_mlp.fit(X_train, y_train)
# save model to disk
filename = 'mlp.sav'
pickle.dump(clf_grid_mlp, open(filename, 'wb'))
# best params: {'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': (50, 100, 50)}    
# roc auc: 0.7001
    

# catboost
from catboost import CatBoostClassifier, Pool, cv

params = {
          'iterations': [50, 100, 150, 200],
          "depth":[5, 8, 10, 12, 16],
          'loss_function': ['Logloss'], 
          'verbose':[True],
          'task_type':['GPU'],
          'l2_leaf_reg': [3]
          } 
# cb support
#pool = Pool(X_train, y_train)
#scores = cv(pool, params)

 # grid search
cb = CatBoostClassifier(random_state=0)
cb_model = GridSearchCV(cb, params, scoring="roc_auc", cv = 2)
cb_model.fit(X_train, y_train)
# best params: {'depth': 16,'iterations': 100,'l2_leaf_reg': 3,'loss_function': 'Logloss','task_type': 'GPU','verbose': True}
# roc auc: 0.7027
# save model to disk
filename = 'cb.sav'
pickle.dump(clf_grid_mlp, open(filename, 'wb'))    
    
    
# model selection on validation set (binary prediction)
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score

# logistic - 1/3 of dataset is false positive
y_pred_logistic = clf_grid_log.predict(X_selection)
logistic_roc = roc_auc_score(y_selection, y_pred_logistic)
logistic_clas_report = classification_report(y_selection, y_pred_logistic)
confusion_matrix(y_selection, y_pred_logistic)
accuracy_score(y_selection, y_pred_logistic)
# mlp - always predicts 0, but 99% accuracy
y_pred_mlp = clf_grid_mlp.predict(X_selection)  
mlp_roc = roc_auc_score(y_selection, y_pred_mlp)
mlp_clas_report = classification_report(y_selection, y_pred_mlp)
confusion_matrix(y_selection, y_pred_mlp) 
accuracy_score(y_selection, y_pred_mlp)
# catboost - same as mlp. recall for class 0 is 100%
y_pred_cb = cb_model.predict(X_selection)    
cb_roc = roc_auc_score(y_selection, y_pred_cb)
cb_clas_report = classification_report(y_selection, y_pred_cb)
confusion_matrix(y_selection, y_pred_cb) 
accuracy_score(y_selection, y_pred_cb)   
    
  



'''

own BID estimation

'''


# 1. LINEAR BIDDING MODEL
# Use catboost to form CTR on validation set e.g. P(click=1)
''' The bid value is linearly proportional to the pCTR (predicted CTR). The formula can be generally written as

                        bid = base_bid* pCTR/avgCTR
                        
where the tuning parameter base bid is the bid price for the average CTR cases. Opti-
mise the base bid and the CTR estimation

SEE BELOW COMMENTED BLOCK
'''

#p_ctr = cb_model.predict_proba(X_selection)[:,1]
#avg_ctr = np.mean(p_ctr)    
#bid = validation['bidprice']
#
#from sklearn.linear_model import LinearRegression
#reg = LinearRegression(fit_intercept=True).fit((p_ctr/avg_ctr).reshape(-1, 1), bid)
#base_bid = reg.coef_
#predicted_bid = reg.predict((p_ctr/avg_ctr).reshape(-1, 1)).reshape(-1,1)
#
## checking the quality of the model
#from sklearn.metrics import r2_score
#r2_score(bid, predicted_bid)  
#import matplotlib.pyplot as plt
#plt.scatter(p_ctr/avg_ctr, bid,color='g')
#plt.scatter(p_ctr/avg_ctr, predicted_bid,color='k')
#plt.show()
#from scipy import stats
#stats.describe(predicted_bid)
#stats.describe(bid)
#
## check total cost of the winning ads are in budget
#ads_won = predicted_bid.reshape(-1,1) - np.array(validation['payprice'].as_matrix()).reshape(-1,1)>0
#ads_won = ads_won.reshape(len(validation['payprice']),)
#
#total_cost = sum(validation['payprice'].iloc[ads_won])/1000 # divide total cost by 1000
#budget = 6250
#if total_cost>budget:
#    print("Out of budget")
#else:
#    print("In budget")
## clicks won
#clicks_won = sum(validation['click'].iloc[ads_won])
## clicks won / total clicks
#proportion_won = clicks_won/sum(validation['click'])



# 1* NEW LINEAR BIDDING MODEL (ON VALIDATION SET)
p_ctr = cb_model.predict_proba(X_selection)[:,1]
avg_ctr = np.mean(p_ctr)  
budget = 6250
data_payprice = np.array(validation['payprice'].as_matrix()).reshape(-1,1)
base_bid_grid = np.linspace(0, 200, 500)

# 1.1. optimising to find base_bid
optimal_clicks = 0
optimal_ctr = 0

for i in base_bid_grid:
    
    # formula from assignament pdf
    current_bid = (i*p_ctr/avg_ctr).reshape(-1, 1)
    # ads won by bid i
    current_ads_won_bool = (current_bid-data_payprice > 0).reshape(len(data_payprice),)
    # total cost implied by bid i (divide by 1000)
    current_total_cost = sum(validation['payprice'].iloc[current_ads_won_bool])/1000 
    # current number of clicks won
    current_clicks = sum(validation['click'].iloc[current_ads_won_bool])
    # Def: Click-Through Rate = Num. of Clicks / Winning Impressions
    current_no_of_winning_ads = sum(current_ads_won_bool)
    
    if current_no_of_winning_ads > 0:
        current_ctr = current_clicks / current_no_of_winning_ads
    else:
        current_ctr = 0
    
    # if bid i => total cost > budget skip guess i from the grid
    if current_total_cost > budget:
        continue
    
    
    # criteria 1: no of clicks
    # if total cost is affordable, optimize base bid by the number of clicks
    if current_clicks > optimal_clicks:  
        optimal_clicks = current_clicks
        optimal_base_bid_no_clicks = i
        
        # additional stats of interest
        optimal_clicks_to_total_clicks_no_clicks = optimal_clicks/sum(validation['click'])
        optimal_bids_bool_no_clicks = current_ads_won_bool
        total_cost_no_clicks = current_total_cost
     
        
    
    # criteria 2: ctr
    # if total cost is affordable, optimize base bid by ctr
    if current_ctr > optimal_ctr:  
        optimal_ctr = current_ctr
        optimal_base_bid_ctr = i
    
        # additional stats of interest
        optimal_clicks_ctr = current_clicks
        optimal_bids_bool_ctr = current_ads_won_bool
        total_cost_ctr = current_total_cost
        optimal_clicks_to_total_clicks_ctr = optimal_clicks_ctr/sum(validation['click'])



    # criteria 3: spend as close as possible to the budget
    lim = 10
    if budget - current_total_cost <= lim:
        optimal_clicks_budget = current_clicks
        optimal_bids_bool_budg = current_ads_won_bool
        total_cost_budg = current_total_cost




# 2. NONLINEAR BIDDING MODEL
# ORBT model: Zhang et al. Optimal real-time bidding for display advertising. KDD 14
# bid = sqrt(pCTR * c/lambda +c^2) - c
        
# the 2 parameter grids to optimize over
lambda_grid = np.linspace(1e-7, 1e-5, 100)
c_grid = np.linspace(1, 100, 500)
optimal_clk = 0
it = 0

for c in c_grid:
    for lmb in lambda_grid:
    
        # formula from assignament pdf
        current_bid = (np.sqrt(p_ctr * c/lmb + c*c)-c).reshape(-1, 1)
        # ads won by bid i
        current_ads_won_bool = (current_bid-data_payprice > 0).reshape(len(data_payprice),)
        # total cost implied by bid i (divide by 1000)
        current_total_cost = sum(validation['payprice'].iloc[current_ads_won_bool])/1000 
        # current number of clicks won
        current_clicks = sum(validation['click'].iloc[current_ads_won_bool])
        # Def: Click-Through Rate = Num. of Clicks / Winning Impressions
        current_no_of_winning_ads = sum(current_ads_won_bool)
            
        
        # if bid i => total cost > budget skip guess i from the grid
        if current_total_cost > budget:
            continue

        # criteria 1: no of clicks
        # if total cost is affordable, optimize base bid by the number of clicks
        if current_clicks > optimal_clicks:  
            optimal_clk = current_clicks
            optimal_base_bid_clk = i
            
            # additional stats of interest
            optimal_clicks_to_total_clicks_clk = optimal_clk/sum(validation['click'])
            optimal_bids_bool_clk = current_ads_won_bool
            total_cost_clk = current_total_cost
         
    
        it += 1
        if it % 100 == 0:
            print("Iter:", it)





    
    