import pandas
import sys
import re
from time import sleep
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import warnings
import csv
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

print "USAGE: python vanillaLDA_findK.py [none - must have files 100 - 350 in directory]"


# perform one-hot encoding
def encode_onehot(df, cols):
    vec = DictVectorizer()
    vec_data = pandas.DataFrame(vec.fit_transform(df[cols].to_dict(orient='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index

    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df
# format feature matrix
def preprocess_features(data):
    data = data.iloc[:, 1:data.shape[1] - 1]
    X0 = features.iloc[:, 1:features.shape[1] - 1]  # feature matrix
    cols_to_transform = ['sex', 'race', 'ethnicity', 'marital_status', 'plan']
    X1 = encode_onehot(X0, cols_to_transform)
    X1[['priorHIVtest', 'anyGC', 'anySyphillis', 'anyHBV', 'anyHCV', 'anyDrugs']] = X1[
        ['priorHIVtest', 'anyGC', 'anySyphillis', 'anyHBV', 'anyHCV', 'anyDrugs']].astype(int)
    X2 = pandas.DataFrame(X1).fillna(0)
    X2.to_csv("X2.csv")
    return (preprocessing.scale(X2))
    print
    "The dimensions of the feature matrix is: " + str(X.shape)

print "importing structured EHR features"
structured_features = pandas.read_csv("structured_features.csv")
print str(len(structured_features['MRN'])) + " MRN in the structured EHR data"


K150 = pandas.read_csv("vanillaLDA_150_features.csv")
K200 = pandas.read_csv("vanillaLDA_200_features.csv")
K250 = pandas.read_csv("vanillaLDA_250_features.csv")
K300 = pandas.read_csv("vanillaLDA_300_features.csv")
#K350 = pandas.read_csv("")

iteration = 1
precision, recall, F1 = [], [], []
for Kdata in [K150, K200, K250, K300]: # ,RedLDA
    # Load Vanilla LDA features
    topic_columns = [e for e in Kdata.columns.values if e.isdigit()]
    topic_columns.append(str('mrn'))
    # Take maximum topic proportion X1...Xn for each MRN
    LDAfeatures = Kdata[topic_columns].groupby(str('mrn'), as_index=True).aggregate('max') # take maximum of topic column for each MRN
    LDAfeatures['mrn'] = LDAfeatures.index
    features = LDAfeatures.merge(structured_features, left_on='mrn', right_on='MRN', how='inner')
    print str(len(features)) + " MRN in the feature matrix"

    X = preprocess_features(features.iloc[:, 0:features.shape[1] - 2])
    Y = np.array(features.iloc[:, features.shape[1] - 1])  # response variable; study_outcome from structured_features
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.25)

    RandomForest = RandomForestClassifier(n_estimators=750, max_features=None, n_jobs=-1)
    RandomForest.fit(Xtrain, Ytrain)
    Ypredict = RandomForest.predict(Xtest)

    print "Random Forest Performance using features #" + str(iteration)
    iteration += 1

    print "F1-measure: " + str(metrics.f1_score(Ytest, Ypredict, pos_label=True, average='binary'))
    print "Precision: " + str(metrics.precision_score(Ytest, Ypredict, pos_label=True, average='binary'))
    print "Recall: " + str(metrics.recall_score(Ytest, Ypredict, pos_label=True, average='binary'))


    recall.append(metrics.recall_score(Y, Ypredict, pos_label=True, average='binary'))
    precision.append(metrics.precision_score(Y, Ypredict, pos_label=True, average='binary'))
    F1.append(str(metrics.f1_score(Y, Ypredict, pos_label=True, average='binary')))

for i in range(len(iterations)):
    print "Model #" + str(iteration)
    print "F1: " + str(F1[i])
    print "Precision: " + str(precision[i])
    print "Recall: " + str(recall[i])

"""
print "Usage: python vanillaLDA_findK.py"
print "Output: An array of topic models and F1-score for V1sample.csv"

# read data
print "load data"
csv.field_size_limit(sys.maxsize) # unlimited size of text fields
try:
    notes = pandas.read_csv("V1_random_sample.csv",engine='python', converters={'note': str, 'mrn':str},
    names = ["X", "X", "file", "MRN", "note", "status", "sample"])
except:
    notes = pandas.read_csv("V1sample.csv", engine='python', converters={'note': str, 'mrn': str})

documents = notes['note']
HIVstatus = notes['status']

# obtain the frequency of each word
print "identify high-frequency words"
frequency = collections.defaultdict(int)
for note in documents:
    for token in str(note).split():
        frequency[token] += 1

# obtain the frequency of the words as a numpy array
n_most_common = 30
np_freq = numpy.zeros(len(frequency))
count = 0
for token in frequency:
    np_freq[count] = frequency[token]
    count += 1
# sort the frequencies
np_freq_sorted = numpy.sort(np_freq)
# obtain the maximum allowed frequency
max_freq = np_freq_sorted[-n_most_common]

high_freq_words = []
for token in frequency.keys():
    if frequency[token] > max_freq:
        high_freq_words.append(token)

# remove words that appear only once or more than M times
notes_noLowFreq = [[token for token in str(note).split() if frequency[token] > 3 and token not in high_freq_words]
                      for note in documents]

# create the dictionary
print "create dictionary & vector-space representation"
dictionary = gensim.corpora.Dictionary(notes_noLowFreq)  # dictionary
corpus = [dictionary.doc2bow(doc) for doc in notes_noLowFreq]  # vector representation

avg_results, k = [], []
print "train many topic models where K= " + str([150,200,250,300])
# loop over different # of topics
for K in [150,200,250,300]:
    for i in range(1,6):
        print str(K) + " topics - iteration #" + str(i)
        results = []

        # generate topic distributions for the testing set
        LDA = gensim.models.LdaModel(corpus, id2word=dictionary, num_topics=K)
        posterior = LDA[corpus]

        # create note x topic matrix
        topic_matrix = numpy.zeros((len(posterior), K))
        note_index = 0
        for note in posterior:
            for topic in note:
                topic_matrix[note_index, topic[0]] = topic[1]  # add topic proportions to matrix
            note_index += 1

        trainX, testX, trainY, testY = train_test_split(topic_matrix, HIVstatus, test_size=0.20) # partition training & testing set

        RandomForest = RandomForestClassifier(n_estimators=1000, max_features=None, n_jobs=-1)
        RandomForest.fit(trainX, trainY)
        testYpredict = RandomForest.predict(testX)

        print str("K: ") + str(metrics.f1_score(str(testY), testYpredict))
        results.append(metrics.f1_score(str(testY), testYpredict))
    avg_results.append(sum(results))
    k.append(K)

for i in len(avg_results):
    print k[i]
    print avg_results[i]
        # split data into testing & training set
        trainX, testX, trainY, testY = train_test_split(corpus, HIVstatus, test_size=0.20) # partition training & testing set

"""



