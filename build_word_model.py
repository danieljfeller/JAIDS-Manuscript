import pandas
import sys
import csv
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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

print "USAGE: python build_models.py [list of unigram tokens] [bootstrap confidence interval?]"
sleep(2)

# perform one-hot encoding of feature matrix
def encode_onehot(df, cols):
    vec = DictVectorizer()
    vec_data = pandas.DataFrame(vec.fit_transform(df[cols].to_dict(orient='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index

    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df

# import features from structured EHR data
structured_features = pandas.read_csv("structured_features.csv")
print str(len(structured_features['MRN'])) + " MRN in the structured EHR data"

# format feature matrix
X0 = structured_features # feature matrix
X0 = X0.drop('ethnicity', 1)
cols_to_transform = ['sex', 'race', 'marital_status', 'plan']
X1 = encode_onehot(X0, cols_to_transform)
X1[['priorHIVtest', 'anyGC', 'anySyphillis', 'anyHBV', 'anyHCV', 'anyDrugs']] = X1[['priorHIVtest', 'anyGC', 'anySyphillis', 'anyHBV', 'anyHCV', 'anyDrugs']].astype(int)
features = pandas.DataFrame(X1).fillna(0)

# import unigram features
unigrams = open(sys.argv[1]).read().split() # get words
tokens = []
for word in unigrams:
    tokens.append(re.sub("u\'|\',|\]|\[|\'", "", word))

csv.field_size_limit(sys.maxsize) # unlimited size of text fields
notes = pandas.read_csv("V2refined.csv", engine='python', converters={'note': str, 'mrn': str}) # read data
cohort = np.array(pandas.read_csv("../propensity_matched_cohort.csv", converters={'x':str})['x'])
notes = notes[notes['mrn'].isin(cohort)]
print str(notes.shape[0]) + " notes being modeled"

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english', vocabulary = tokens) # creates document x term matrix w/ TF-IDF weights
dtm = vectorizer.fit_transform(notes['note'].values.astype('U')) # convert to unicode string
dtm = pandas.DataFrame(data = dtm.toarray())
dtm.columns = vectorizer.get_feature_names()
columns = [col for col in dtm.columns]
columns.append('mrn')
dtm = pandas.concat([dtm, notes['mrn']], axis=1) # get MRNs
dtm.columns = columns
dtm.to_csv("dtm.csv")
unigram_features = dtm.groupby('mrn', as_index=True).aggregate('max')# take maximum of each unigram column for each MRN
print unigram_features.columns.values
unigram_features['mrn'] = unigram_features.index
structured_features.MRN = structured_features.MRN.astype(int)
unigram_features.mrn = unigram_features.mrn.astype(int)

# merge unigram features & structured features
features = unigram_features.merge(features, left_on='mrn', right_on='MRN', how = 'inner')

# retrieve class labels (HIV status)
print "Class labels in feature matrix:"
Y = np.array(features['study_outcome']) # response variable; study_outcome from structured_features
print Y

# FINALIZE FEATURE MATRI
features = features.drop('study_outcome', 1)

features = features.fillna('0')# fill NAN with 0s
print features.shape
X = preprocessing.scale(features) # feature matrix
X_MI = SelectKBest(mutual_info_classif, k=150).fit_transform(X, Y)
X_names_MI = features.columns.values[SelectKBest(mutual_info_classif, k=150).fit(X, Y).get_support()]
print "The dimensions of the feature matrix is: " + str(X_MI.shape)


RandomForest = RandomForestClassifier(n_estimators = 1500, max_features = None, n_jobs = -1)

Ypredict_probs = pandas.DataFrame(cross_val_predict(RandomForest, X_MI, Y, cv=10, method = 'predict_proba'))
predictions = pandas.concat([pandas.DataFrame(Y), Ypredict_probs], axis=1)
predictions.columns = ['class', 'predict0', 'predict1']
predictions.to_csv("predictions/Unigram_pred_probs.csv")

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

    print f1
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
















