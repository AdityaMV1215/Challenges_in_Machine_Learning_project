import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score

bank = pd.read_csv('data/Bank_data_to_class.csv', sep=',')

y = bank['y']
X = bank.drop('y', axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=9)
model = GradientBoostingClassifier(random_state=9)
# Write your solution here :
def pipeline(X_train,X_test,y_train,y_test,model):
    bank = pd.read_csv('data/Bank_data_to_class.csv', sep=',')
    y = bank['y']
    X = bank.drop('y', axis=1)
    param_grid = {'n_estimators':[10,20,30], 'max_features':[2,4,6,8,12,16], 'max_depth':[2,4,6,8]}
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num_features = X_train.select_dtypes(include=numerics).columns.tolist()
    cat_features = list(set(X_train.columns.tolist()) - set(num_features))
    onehot_en = ['marital','contact','month','job']
    label_en = list(set(cat_features) - set(onehot_en))
    for val in cat_features:
        temp = X_train[X_train[val] != 'unknown'][val].mode()[0]
        X_train.loc[X_train[val] == 'unknown',val] = temp
        X_test.loc[X_test[val] == 'unknown',val] = temp

    encoded_df_train = X_train
    encoded_df_test = X_test
    le = LabelEncoder()
    for val in cat_features:
        encoded_df_train[val] = le.fit_transform(encoded_df_train[val])
        encoded_df_test[val] = le.fit_transform(encoded_df_test[val])
    y_train1 = le.fit_transform(y_train)
    y_test1 = le.fit_transform(y_test)
    sm = SMOTE(random_state=9)
    X_res_train, y_res_train = sm.fit_sample(encoded_df_train,y_train1)
    X_res_test, y_res_test = sm.fit_sample(encoded_df_test,y_test1)
    #model.fit(X_res_train,y_res_train)
    search = GridSearchCV(model, param_grid)
    search.fit(X_res_train,y_res_train)
    y_pred = search.predict_proba(X_res_test)[:, 1]
    auc = roc_auc_score(y_res_test,y_pred)
    return search, auc

#print(pipeline(X_train,X_test,y_train,y_test,model))
