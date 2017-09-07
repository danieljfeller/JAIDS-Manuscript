import pandas
import sys
import re
from time import sleep
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print "USAGE: python build_models.py [LDA features file] [bootstrap confidence interval?]"
sleep(2)

# import features from structured EHR data
structured_features = pandas.read_csv("structured_features.csv")
print str(len(structured_features['MRN'])) + " MRN in the structured EHR data"

# structuredHIVcohort = structured_features[structured_features.study_outcome == True]['MRN']
# RedLDAfeatures = pandas.read_csv(sys.argv[2])

# Load vanilla LDA features
LDAdistribution = pandas.read_csv(sys.argv[1])
topic_columns = [e for e in LDAdistribution.columns.values if e.isdigit()]
topic_columns.append(str('mrn'))
# Take maximum topic proportion X1...Xn for each MRN
LDAfeatures = LDAdistribution[topic_columns].groupby(str('mrn'), as_index=True).aggregate('max')#.aggregate(max)# take maximum of topic column for each MRN
LDAfeatures['mrn'] = LDAfeatures.index
print str(len(set(LDAfeatures['mrn']))) + " MRN in the topic features"

# Format Data for sklearn
features = LDAfeatures.merge(structured_features, left_on='mrn', right_on='MRN', how = 'inner')
print str(features.shape[0]) + " MRN in feature matrix"

# perform one-hot encoding of feature matrix
def encode_onehot(df, cols):
    vec = DictVectorizer()
    vec_data = pandas.DataFrame(vec.fit_transform(df[cols].to_dict(orient='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index

    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df

features = features.iloc[:,1:features.shape[1]-1]
X0 = features.iloc[:,1:features.shape[1]-1] # feature matrix
cols_to_transform = ['sex', 'race', 'ethnicity', 'marital_status', 'plan']
X1 = encode_onehot(X0, cols_to_transform)
X1[['priorHIVtest', 'anyGC', 'anySyphillis', 'anyHBV', 'anyHCV', 'anyDrugs']] = X1[['priorHIVtest', 'anyGC', 'anySyphillis', 'anyHBV', 'anyHCV', 'anyDrugs']].astype(int)
X2 = pandas.DataFrame(X1).fillna(0)
X2.to_csv("X2.csv")
X = preprocessing.scale(X2)
print "The dimensions of the feature matrix is: " + str(X.shape)

# retrieve class labels (HIV status)
print "Class labels in feature matrix:"
Y = np.array(features.iloc[:,features.shape[1]-1]) # response variable; study_outcome from structured_features
print Y

X_MI = SelectKBest(mutual_info_classif, k=150).fit_transform(X, Y)
X_names_MI = X2.columns.values[SelectKBest(mutual_info_classif, k=150).fit(X, Y).get_support()]

RandomForest = RandomForestClassifier(n_estimators = 1500, max_features = None, n_jobs = -1)

Ypredict_probs = pandas.DataFrame(cross_val_predict(RandomForest, X_MI, Y, cv=10, method = 'predict_proba'))
predictions = pandas.concat([pandas.DataFrame(Y), Ypredict_probs], axis=1)
predictions.columns = ['class', 'predict0', 'predict1']
predictions.to_csv("predictions/LDA_pred_probs.csv")

if str(sys.argv[2]) != 'True':
    trainX, testX, trainY, testY = train_test_split(X_MI, Y, test_size=0.20)  # partition training & testing set
    RandomForest.fit(trainX, trainY)
    Ypredict = cross_val_predict(RandomForest, X_MI, Y, cv=10)

    print "Random Forest Variable Importance"
    sleep(1)
    var_importance = zip(X_names_MI, RandomForest.feature_importances_)
    counter = 0
    for feature, value in sorted(var_importance, key=lambda x: x[1], reverse=True):
        if value > 0.0:
            print feature, value
        counter += 1
        if counter > 20:
            break

    print "Random Forest Performance:"
    sleep(1)
    print "Recall: " + str(metrics.recall_score(Y,Ypredict, pos_label = True, average = 'binary'))
    print "Precision: " + str(metrics.precision_score(Y,Ypredict, pos_label = True, average = 'binary'))
    print "F-measure: " + str(metrics.f1_score(Y,Ypredict, pos_label = True, average = 'binary'))
    sleep(1)


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

if str(sys.argv[2]) == 'True':
    print "bootstrapping 100 iterations"
    precision, recall, f1 = [], [], []
    for i in range(1,30):
        trainX, testX, trainY, testY = train_test_split(X_MI, Y, test_size=0.20)  # partition training & testing set
        RandomForest.fit(trainX, trainY)
        Ypredict = cross_val_predict(RandomForest, X_MI, Y, cv=10)
        recall.append(metrics.recall_score(Y, Ypredict, pos_label=True, average='binary'))
        precision.append(metrics.precision_score(Y, Ypredict, pos_label=True, average='binary'))
        f1.append(metrics.f1_score(Y, Ypredict, pos_label=True, average='binary'))

    print "F1: Mean/Upper/Lower" + str(mean(f1)) + str(mean(f1) + 2*np.std(f1)) + str(mean(f1) - 2*np.std(f1))
    print "Precision: Mean/Upper/Lower" + str(mean(precision)) + str(mean(precision) + 2*np.std(precision)) + str(mean(precision) - 2*np.std(precision))
    print "Recall: Mean/Upper/Lower" + str(mean(recall)) + str(mean(recall) + 2*np.std(recall)) + str(mean(recall) - 2*np.std(recall))


"""
## perform 10-fold cross validation with a 80/20 split
# partition data into training and testing sets 80/20
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.20) # partition training & testing set
print str(len(trainY)) + " MRN in training set"
print str(len(testY)) + " MRN in testing set"

print "Random Forest Performance:"
sleep(1)
print "Recall: " + str(metrics.recall_score(testY,testYpredict))
print "Precision: " + str(metrics.precision_score(testY,testYpredict))
print "F-measure: " + str(metrics.f1_score(testY,testYpredict))
sleep(1)

print "Random Forest Variable Importance"
sleep(1)
var_importance = zip(X1.columns.values, RandomForest.feature_importances_)
counter = 0
for feature, value in sorted(var_importance, key=lambda x: x[1], reverse=True):
    if value > 0.0:
        print feature, value
    counter += 1
    if counter > 5:
        break
"""
"""
AdaBoost = AdaBoostClassifier(DecisionTreeClassifier(),
                         algorithm="SAMME",
                         n_estimators=1000)
AdaBoost.fit(trainX, trainY)

testYpredict = AdaBoost.predict(testX)
print "AdaBoost Performance:"
sleep(1)
print "Recall: " + str(metrics.recall_score(testY,testYpredict))
print "Precision: " + str(metrics.precision_score(testY,testYpredict))
print "F-measure: " + str(metrics.f1_score(testY,testYpredict))
sleep(1)

# train the model
LogisticRegression = LogisticRegression()
LogisticRegression.fit(trainX, trainY)

print "L2 Logistic Regression Performance:"
sleep(1)
testYpredict = LogisticRegression.predict(testX)
print "Recall: " + str(metrics.recall_score(testY,testYpredict))
print "Precision: " + str(metrics.precision_score(testY,testYpredict))
print "F-measure: " + str(metrics.f1_score(testY,testYpredict))
sleep(1)

# coefficient multiplied by standard deviation of column vector
print(np.std(trainX, 0)*LogisticRegression.coef_)
"""

'''
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict

k = KFold(n_splits=10)
testYpredictCV = cross_val_predict(RandomForest, X, Y, cv = k)
fpr, tpr, thresholds = metrics.roc_curve(Y, testYpredictCV)
print "Cross-Validation AUC: " + str(metrics.auc(fpr, tpr))
'''
















