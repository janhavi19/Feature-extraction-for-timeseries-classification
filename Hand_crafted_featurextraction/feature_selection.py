import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.svm import SVC # SVM model algorithm
from sklearn.metrics import accuracy_score # evalution metric
from sklearn.metrics import confusion_matrix 
import numpy as np
# import library
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def get_model_details(df,cmodel):
    
    X = df.loc[(df['Model'] == cmodel)]
    y= X['Status']
    
    X = X.drop(['Status','Model'], axis=1)
    
    y= LabelEncoder().fit_transform(y)
    return X,y

def data_arrange(X,y):
    
    col = list(X.columns.values)
    col = np.asarray(col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
    return col,X_train, X_test, y_train, y_test

def balance_data(X_train, y_train):
    rus = RandomUnderSampler(random_state=42, replacement=True)# fit predictor and target variable
    x_t_n, y_t_n = rus.fit_resample(X_train, y_train)
    return x_t_n, y_t_n

def slight_high(X_train, y_train):
    sampling_strategy = {0: 103, 1: 206}
    rus1 = RandomUnderSampler(sampling_strategy=sampling_strategy)
    x_n, y_n = rus1.fit_resample(X_train, y_train)
    return x_n, y_n
    
def feature_relavance(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)
    sorted_idx = rf.feature_importances_.argsort()
    plt.barh(col[sorted_idx], rf.feature_importances_[sorted_idx])
    plt.xlabel("Random Forest Feature Importance")
    plt.rcParams["figure.figsize"] = (35,30)
    
def data_modification_relavance(X_train, X_test,col,setf):
   
    if setf=='F1':
        features_selected = ['median_V', 'max_P', 'median_P', 'f_max_A', 'mean_A', 'median_A',
       'DevationMedian_A', 'min_A', 'Closing speed', 'Penetration']
        
        X_s_train = X_train.reindex(index=features_selected)
   
        X_s_test = X_test.reindex(index=features_selected)
        
        return X_s_train,X_s_test
    
    elif setf=='F2':
        features_selected = ['Closing speed', 'Penetration']
        X_s_train = X_train.reindex(index=features_selected)
        X_s_test = X_test.reindex(index=features_selected)
        
        return X_s_train,X_s_test
        
    else:
        print("wrong input")
        
    
    
def classification(X_selected_train,y_train,X_selected_test,y_test):
    model =  RandomForestClassifier(n_estimators=100)
    model.fit(X_selected_train, y_train)
    yhat = model.predict(X_selected_test)
    print(('Accuracy score of our model is {}'.format(accuracy_score(y_test, yhat))))
    error_df = pd.DataFrame({'Predictions': yhat,
                        'True_class': y_test})
    
    LABELS = ["NOK","OK"]

    conf_matrix = confusion_matrix(error_df.True_class, error_df.Predictions)
    plt.figure(figsize=(4, 4))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.save("")
    
def perf_measure(y_true, y_pred):
    CM = confusion_matrix(y_true, y_pred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]

    return(TP, FP, TN, FN)

def multi_algorithm(X_train, X_test, y_train, y_test):
    models = [
          ('Logistic Regression', LogisticRegression()), 
          ('Random Forest', RandomForestClassifier(n_estimators=100)),
          ('SVM', SVC()),
          ('Naive Bayes',GaussianNB())
          
          ]
    col = ['Classifiers','Accuracy','AUC score','True positive','False positive','True negative','False negative']

    auc_score = 0
    model_results =[]
    for name,model in models:
        result = []
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        TP, FP, TN, FN = perf_measure(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred)
        result = [name,accuracy,auc_score,TP, FP, TN, FN]
        model_results.append(result)

        model_results = np.array(model_results)
        df_results = pd.DataFrame(model_results,columns=col)
        df_results[['Accuracy','AUC score','True positive','False positive','True negative','False negative']] = df_results[['Accuracy','AUC score','True positive','False positive','True negative','False negative']].apply(pd.to_numeric)
        df_results
def process_balnced(X, y,col,cat):
    
    col,X_train, X_test, y_train, y_test = data_arrange(X,y)
    X_b_t,y_b_t = balance_data(X_train, y_train)
    feature_relavance(X_b_t,y_b_t)
    X_s_train,X_s_test = data_modification_relavance(X_b_t,y_b_t,col,cat)
    classification(X_s_train,y_train,X_s_test,y_test)

def process_slight(X, y,col,cat):

    col,X_train, X_test, y_train, y_test = data_arrange(X,y)
    X_b_t,y_b_t = balance_data(X_train, y_train)
    feature_relavance(X_b_t,y_b_t)
    X_s_train,X_s_test = data_modification_relavance(X_b_t,y_b_t,col,cat)
    classification(X_s_train,y_train,X_s_test,y_test)


if __name__ == "__main__":
    cmodel,catagory = str(sys.argv[1]), str(sys.argv[2])
    if catagory == "F1":
        process_balnced(X, y,col,catagory)

    elif catagory == "F2":
        process_slight( (X, y, col, catagory)