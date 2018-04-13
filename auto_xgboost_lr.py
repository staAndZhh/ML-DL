import numpy as np
import pandas as pd
import warnings
from sklearn import cross_validation
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

def metrics_spec(actual_data, predict_data, cutoff=0.5):
    actual_data = np.array(actual_data)
    predict_data = np.array(predict_data)
    bind_data = np.c_[actual_data, predict_data]
    dr = 1.0 * (bind_data[bind_data[:, 0] == 1][:, 1] >= cutoff).sum() / bind_data[bind_data[:, 0] == 1].shape[0]
    ppv = 1.0 * (
        (bind_data[bind_data[:, 0] == 1][:, 1] >= cutoff).sum() + (
            bind_data[bind_data[:, 0] == 0][:, 1] < cutoff).sum()) / \
           bind_data.shape[0]
    fpr = 1.0 * (bind_data[bind_data[:, 0] == 0][:, 1] >= cutoff).sum() / bind_data.shape[0]
    print("模型检出率:%.3f" % dr, "\t模型假阳性率:%.3f" % fpr, "\t模型阳性预测值:%.3f" % ppv)

def auto_xgboost_lr(df_train, f_target = 'type', scoring_grid = 'roc_auc', random_seed = 1301, cv_fold = 5, unbalance = True):
    y = df_train[f_target]
    X = df_train.drop(f_target, axis = 1, inplace = False)
    
    if unbalance == True:
        ratio = y.sum()/y.shape[0]
    else:
        ratio = 1        
    
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, y, random_state = random_seed, stratify = y, test_size = 0.4)
    param_test = {
        'learning_rate':[0.1, 0.05, 0.02, 0.01, 0.005],
    }
    gsearch0 = GridSearchCV(
        estimator=XGBClassifier(
            learning_rate = 0.1,
            n_estimators = 1000,
            max_depth = 4,
            min_child_weight = 1,
            gamma = 0,
            colsample_bytree = 0.8,
            subsample = 0.8,
            objective='binary:logistic',
            scale_pos_weight = 1/ratio,
            reg_alpha = 0.01,
            seed=random_seed),
        param_grid=param_test,
        scoring = scoring_grid,
        n_jobs=-1,
        iid=False,
        cv = cv_fold)
    gsearch0.fit(X_train, Y_train)
    print('learning_rate: ',list(gsearch0.best_params_.values())[0])

    param_test = {
        'n_estimators': list(range(100,3000,100))
    }
    gsearch1 = GridSearchCV(
        estimator=XGBClassifier(
            learning_rate = gsearch0.best_params_['learning_rate'],
            n_estimators = 100,
            max_depth = 4,
            min_child_weight = 1,
            gamma = 0,
            colsample_bytree = 0.8,
            subsample = 0.8,
            objective='binary:logistic',
            scale_pos_weight = 1/ratio,
            reg_alpha = 0.01,
            seed=random_seed),
        param_grid=param_test,
        scoring = scoring_grid,
        n_jobs=-1,
        iid=False,
        cv = cv_fold)
    gsearch1.fit(X_train, Y_train)

    param_test = {
        'n_estimators': list(range(gsearch1.best_params_['n_estimators']-100,gsearch1.best_params_['n_estimators']+100,10))
    }
    gsearch1_0 = GridSearchCV(
        estimator=XGBClassifier(
            learning_rate = gsearch0.best_params_['learning_rate'],
            n_estimators = 100,
            max_depth = 4,
            min_child_weight = 1,
            gamma = 0,
            colsample_bytree = 0.8,
            subsample = 0.8,
            objective='binary:logistic',
            scale_pos_weight = 1/ratio,
            reg_alpha = 0.01,
            seed=random_seed),
        param_grid=param_test,
        scoring = scoring_grid,
        n_jobs=-1,
        iid=False,
        cv = cv_fold)
    gsearch1_0.fit(X_train, Y_train)
    print('n_estimators: ',gsearch1_0.best_params_['n_estimators'])

    param_test = {
        'max_depth': list(range(3,9,1)),
        'min_child_weight': list(range(2,20,2))
    }
    gsearch2 = GridSearchCV(
        estimator=XGBClassifier(
            learning_rate =  gsearch0.best_params_['learning_rate'],
            n_estimators = gsearch1_0.best_params_['n_estimators'],
            max_depth = 4,
            min_child_weight = 1,
            gamma = 0,
            colsample_bytree = 0.8,
            subsample = 0.8,
            objective='binary:logistic',
            scale_pos_weight = 1/ratio,
            reg_alpha = 0.01,
            seed=random_seed),
        param_grid=param_test,
        scoring = scoring_grid,
        n_jobs=-1,
        iid=False,
        cv = cv_fold)
    gsearch2.fit(X_train, Y_train)
    print('max_depth: %d'%gsearch2.best_params_['max_depth'],'\tmin_child_weight: %d'%gsearch2.best_params_['min_child_weight'])

    param_test = {
        'gamma': np.array(list(range(1,30,2)))/100,
    }
    gsearch3 = GridSearchCV(
        estimator=XGBClassifier(
            learning_rate = gsearch0.best_params_['learning_rate'],
            n_estimators =  gsearch1_0.best_params_['n_estimators'],
            max_depth = gsearch2.best_params_['max_depth'],
            min_child_weight = gsearch2.best_params_['min_child_weight'],
            gamma = 0,
            colsample_bytree = 0.8,
            subsample = 0.8,
            objective='binary:logistic',
            scale_pos_weight = 1/ratio,
            reg_alpha = 0.01,
            seed=random_seed),
        param_grid=param_test,
        scoring = scoring_grid,
        n_jobs=-1,
        iid=False,
        cv = cv_fold)
    gsearch3.fit(X_train, Y_train)
    print('gamma:', gsearch3.best_params_['gamma'])

    param_test = {
        'subsample': np.array(list(range(5,11,1)))/10,
        'colsample_bytree': np.array(list(range(1,11,1)))/10
    }
    gsearch4 = GridSearchCV(
        estimator=XGBClassifier(
            learning_rate = gsearch0.best_params_['learning_rate'],
            n_estimators = gsearch1_0.best_params_['n_estimators'],
            max_depth = gsearch2.best_params_['max_depth'],
            min_child_weight = gsearch2.best_params_['min_child_weight'],
            gamma =  gsearch3.best_params_['gamma'],
            colsample_bytree = 0.8,
            subsample = 0.8,
            objective='binary:logistic',
            scale_pos_weight = 1/ratio,
            reg_alpha = 0.01,
            seed=random_seed),
        param_grid=param_test,
        scoring = scoring_grid,
        n_jobs=-1,
        iid=False,
        cv = cv_fold)
    gsearch4.fit(X_train, Y_train)

    param_test = {
        'subsample': np.array(list(range(int(gsearch4.best_params_['subsample']*100-10),int(gsearch4.best_params_['subsample']*100+10),1)))/100,
        'colsample_bytree': np.array(list(range(int(gsearch4.best_params_['colsample_bytree']*100-10),int(gsearch4.best_params_['colsample_bytree']*100+10),1)))/100
    }
    gsearch4_0 = GridSearchCV(
        estimator=XGBClassifier(
            learning_rate = gsearch0.best_params_['learning_rate'],
            n_estimators = gsearch1_0.best_params_['n_estimators'],
            max_depth = gsearch2.best_params_['max_depth'],
            min_child_weight = gsearch2.best_params_['min_child_weight'],
            gamma =  gsearch3.best_params_['gamma'],
            colsample_bytree = 0.8,
            subsample = 0.8,
            objective='binary:logistic',
            scale_pos_weight = 1/ratio,
            reg_alpha = 0.01,
            seed=random_seed),
        param_grid=param_test,
        scoring = scoring_grid,
        n_jobs=-1,
        iid=False,
        cv = cv_fold)
    gsearch4_0.fit(X_train, Y_train)
    print('subsample: ',gsearch4_0.best_params_['subsample'],'\tcolsample_bytree: ',gsearch4_0.best_params_['colsample_bytree'])

    param_test = {
        'reg_alpha': [1e-5, 1e-2, 0.1,1,100],
    }
    gsearch5 = GridSearchCV(
        estimator=XGBClassifier(
            learning_rate = gsearch0.best_params_['learning_rate'],
            n_estimators = gsearch1_0.best_params_['n_estimators'],
            max_depth = gsearch2.best_params_['max_depth'],
            min_child_weight = gsearch2.best_params_['min_child_weight'],
            gamma =  gsearch3.best_params_['gamma'],
            colsample_bytree = gsearch4_0.best_params_['colsample_bytree'],
            subsample = gsearch4_0.best_params_['subsample'],
            objective='binary:logistic',
            scale_pos_weight = 1/ratio,
            reg_alpha = 0.01,
            seed=random_seed),
        param_grid=param_test,
        scoring = scoring_grid,
        n_jobs=-1,
        iid=False,
        cv = cv_fold)
    gsearch5.fit(X_train, Y_train)
    print('reg_alpha: ',gsearch5.best_params_['reg_alpha'])

    xgb_0 = XGBClassifier(
            learning_rate = gsearch0.best_params_['learning_rate'],
            n_estimators = gsearch1_0.best_params_['n_estimators'],
            max_depth = gsearch2.best_params_['max_depth'],
            min_child_weight = gsearch2.best_params_['min_child_weight'],
            gamma =  gsearch3.best_params_['gamma'],
            colsample_bytree = gsearch4_0.best_params_['colsample_bytree'],
            subsample = gsearch4_0.best_params_['subsample'],
            objective='binary:logistic',
            scale_pos_weight = 1/ratio,
            reg_alpha = gsearch5.best_params_['reg_alpha'],
            seed=random_seed
    )
    model_xgb = xgb_0.fit(X_train, Y_train)
    print('xgboost模型效果:训练集')
    metrics_spec(Y_train, model_xgb.predict_proba(X_train)[:, 1])
    print('xgboost模型效果:预测集')
    metrics_spec(Y_test, model_xgb.predict_proba(X_test)[:, 1])

    # 叶子结点获取
    train_new_feature = model_xgb.apply(X_train)
    test_new_feature = model_xgb.apply(X_test)
    # enhotcoding
    enc = OneHotEncoder()
    enc.fit(train_new_feature)
    train_new_feature2 = np.array(enc.transform(train_new_feature).toarray())
    test_new_feature2 = np.array(enc.transform(test_new_feature).toarray())
    res_data = pd.DataFrame(np.c_[Y_train, train_new_feature2])
    res_data.columns = ['f' + str(x) for x in range(res_data.shape[1])]
    res_test = pd.DataFrame(np.c_[Y_test, test_new_feature2])
    res_test.columns = ['f' + str(x) for x in range(res_test.shape[1])]

    param_test = {
        'C' : list(range(1,11,1)),
        'penalty': ['l1','l2']
    }
    gsearch6 = GridSearchCV(
        estimator=LogisticRegression(
            C = 2,
            penalty = 'l2',
            max_iter=1000, 
            solver='liblinear', 
            multi_class='ovr', 
            class_weight = None),
        param_grid=param_test,
        scoring=scoring_grid,
        n_jobs=1,
        iid=False,
        cv = cv_fold)
    gsearch6.fit(res_data.iloc[:,1:], res_data['f0'])
    print('C:',gsearch6.best_params_['C'],'\tpenalty:',gsearch6.best_params_['penalty'])

    param_test = {
        'class_weight':[{0: w} for w in np.array(list(range(5,100,5)))/100] 
    }
    gsearch7 = GridSearchCV(
        estimator=LogisticRegression(
            C = gsearch6.best_params_['C'],
            penalty = gsearch6.best_params_['penalty'],
            max_iter=1000, 
            solver='liblinear', 
            multi_class='ovr', 
            class_weight = None),
        param_grid=param_test,
        scoring=scoring_grid,
        n_jobs=1,
        iid=False,
        cv = cv_fold)
    gsearch7.fit(res_data.iloc[:,1:], res_data['f0'])
    print('class_weight',gsearch7.best_params_['class_weight'])

    lr = LogisticRegression(
        C = gsearch6.best_params_['C'], 
        penalty=gsearch6.best_params_['penalty'], 
        max_iter=1000, 
        solver='liblinear', 
        multi_class='ovr', 
        class_weight = gsearch7.best_params_['class_weight']
    )
    model_lr = lr.fit(res_data.iloc[:,1:], res_data['f0'])
    y_train_lr = model_lr.predict_proba(res_data.iloc[:,1:])[:, 1]
    y_test_lr = model_lr.predict_proba(res_test.iloc[:,1:])[:, 1]
    print('xgboost+lr模型效果:训练集')
    metrics_spec(Y_train, y_train_lr)
    print('xgboost+lr模型效果:预测集')
    metrics_spec(Y_test, y_test_lr)
    
    return model_xgb, model_lr, train_new_feature, y_train_lr, y_test_lr

    #res_data.to_csv('编码原始数据集.csv', encoding = 'utf-8')

def predict(model_xgb, model_lr, leaves_feature, data_predict, target_feature = 'type'):
    
    y = data_predict[target_feature]
    X = data_predict.drop(target_feature, axis = 1, inplace = False)
    predict_new_feature = model_xgb.apply(X)
    xgbenc = OneHotEncoder()
    xgbenc.fit(leaves_feature)
    train_new_feature2 = np.array(xgbenc.transform(leaves_feature).toarray())
    predict_new_feature2 = np.array(xgbenc.transform(predict_new_feature).toarray())
    y_pred = model_lr.predict_proba(predict_new_feature2)[:, 1]
   
    return y_pred

import matplotlib.pyplot as plt

def cutoff_plot(y_actual, y_pred, col = 'b-'):
    
    cut_off = []
    dr = []
    fpr = []
    for cutoff in np.array(list(range(1,100,1)))/100:
        cut_off.append(cutoff)
        dr.append(1.0 * (y_pred[y_actual == 1] >= cutoff).sum() / y_actual[y_actual == 1].sum())
        fpr.append(1.0 * (y_pred[y_actual == 0] >= cutoff).sum() / len(y_actual))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Cutoff + DR')
    plt.plot(cut_off, dr, 'b-',
             label='all_data')
    plt.legend(loc='upper right')
    plt.xlabel('Cutoff')
    plt.ylabel('Detection Rate')

    plt.subplot(1, 2, 2)
    plt.title('Cutoff + FPR')
    plt.plot(cut_off, fpr, 'r-',
             label='all_data')
    plt.legend(loc='upper right')
    plt.xlabel('Cutoff')
    plt.ylabel('False positive rate')
    
    res = pd.DataFrame(np.empty((99,3)))
    res.iloc[:,0] = cut_off
    res.iloc[:,1] = dr
    res.iloc[:,2] = fpr
    res.columns = ['cut_off', 'DR','FPR']
    
    return res   

"""
df_train = {
    'fj': train_fj,
    'sd': train_sd,
    'zj': train_zj
}

xgb_model = {}
lr_model ={}
train_new_feature = {}
y_train_pred = {}
y_test_pred = {}

for region in ['fj', 'sd', 'zj']:
    (xgb_model[region], lr_model[region], train_new_feature[region], y_train_pred[region], y_test_pred[region]) = \
        auto_xgboost_lr(df_train[region], scoring_grid = 'f1_weighted', unbalance = True)

region_0 = 'sd'
y_preds = predict(xgb_model[region_0], lr_model[region_0], train_new_feature[region_0], df_train[region_0])
metrics_spec(df_train[region_0]['type'], y_preds)
cutoff_plot(df_train[region_0]['type'], y_preds)

"""

def region_stacking(model_xgb, model_lr, leaves_feature, df, region_cal, 
                    region_stack = ['fj', 'sd', 'zj'], f_target = 'type', weighted = False, r_w = 0.8):    
    y_preds = {}
    for i in region_stack:
        y_preds[i] = predict(model_xgb[i], model_lr[i], leaves_feature[i], df[region_cal])
    if weighted == 'sample_weight':
        n = len(df_train.keys())
        w_0 = []
        for i in range(0,n):
            w_0.append(df[region_stack[i]].loc[df[region_stack[i]][f_target]==1,:].shape[0])
        w = np.array(w_0)/sum(w_0)
        y_result = (pd.DataFrame(y_preds) * w).sum(axis = 1).values 
    elif weighted == 'region_weight':
        n = len(df_train.keys())
        w_0 = []
        for i in range(0,n):
            w_0.append((1-r_w)/(n-1))
        w_0 = np.array(w_0)
        w_0[region_stack.index(region_cal)] = r_w
        w = w_0/sum(w_0)
        y_result = (pd.DataFrame(y_preds) * w).sum(axis = 1).values 
    else:
        y_result = pd.DataFrame(y_preds).sum(axis =1).values
    metrics_spec(df_train[region_cal][f_target], y_result)
    res = cutoff_plot(df_train[region_cal][f_target], y_result)
    
    return res

# region_stacking(xgb_model, lr_model, train_new_feature, df_train, 'zj', weighted = 'region_weight', r_w = 0.8) 

def region_stacking_predict(model_xgb, model_lr, leaves_feature, predict_data, region_cal, 
                    region_stack = ['fj', 'sd', 'zj'], f_target = 'type', weighted = False, r_w = 0.8):    
    y_preds = {}
    for i in region_stack:
        y_preds[i] = predict(model_xgb[i], model_lr[i], leaves_feature[i], predict_data)
    if weighted == 'sample_weight':
        w_sample = {
            'fj': 0.63098592,
            'sd': 0.12394366,
            'zj': 0.24507042
        }
        w = []
        for i in region_stack:
            w.append(w_sample[i])
        y_result = (pd.DataFrame(y_preds) * w).sum(axis = 1).values 
    elif weighted == 'region_weight':
    	n = len(region_stack)
        w_0 = []
        for i in range(0,n):
            w_0.append((1-r_w)/(n-1))
        w_0 = np.array(w_0)
        w_0[region_stack.index(region_cal)] = r_w
        w = w_0/sum(w_0)
        y_result = (pd.DataFrame(y_preds) * w).sum(axis = 1).values 
    else:
        y_result = pd.DataFrame(y_preds).sum(axis =1).values
    metrics_spec(predict_data[f_target], y_result)
    res = cutoff_plot(predict_data[f_target], y_result)
    
    return res

# region_stacking_predict(xgb_model, lr_model, train_new_feature, test, 'zj', weighted = 'region_weight', r_w = 0.8) 