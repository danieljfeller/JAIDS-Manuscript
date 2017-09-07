import gensim
import numpy
import collections
import pandas
import re
import sys
import csv
from sklearn.feature_extraction.text import CountVectorizer

print "USAGE: sys.argv[1] is K (# of topics)"

print "reading in data"
csv.field_size_limit(sys.maxsize) # unlimited size of text fields
notes = pandas.read_csv("V2refined.csv",engine='python', converters={'note': str, 'mrn':str})
print notes.shape
#names = ["X", "X", "file", "MRN", "note", "status", "sample"]
cohort = numpy.array(pandas.read_csv("../propensity_matched_cohort.csv", converters={'x':str})['x'])
notes = notes[notes['mrn'].isin(cohort)]
print str(notes.shape[0]) + " notes being modeled"

print list(notes.columns.values)
documents = notes['note']

# obtain the frequency of each word
frequency = collections.defaultdict(int)
for note in documents:
    for token in str(note).split():
        frequency[token] += 1

print "obtaining most frequent words"
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
documents_noLowFreq = [[token for token in str(note).split() if frequency[token] > 3 and token not in high_freq_words]
                      for note in documents]

documents_joined = [" ".join(note) for note in documents_noLowFreq]
count_vect = CountVectorizer()
print "creating document-term matrix"
dtm0 = count_vect.fit_transform(documents_joined)
print dtm0.shape
dtm1 = pandas.DataFrame(dtm0.toarray())
print dtm1.shape
note_info = notes.loc[:, ['file', 'mrn', 'status']]
print "concatenating note info with dtm"
dtm = pandas.concat([note_info, dtm1], axis=1)
print dtm.shape
print "writing to csv"
dtm.to_csv("notes_dtm.csv")

dictionary = gensim.corpora.Dictionary(documents_noLowFreq)  # dictionary
document_corpus = [dictionary.doc2bow(doc) for doc in documents_noLowFreq]  # vector representation

print "training model"

# train final model
final_K = int(sys.argv[1])
final_model = gensim.models.LdaModel(document_corpus, id2word=dictionary, num_topics=final_K)

print "generating topic proportions"
posterior = final_model[document_corpus]  # inference on testing corpus

try:
    matrix = numpy.zeros((len(documents), final_K))  # document-topic matrix
except:
    matrix = numpy.zeros((len(posterior), final_K))  # document-topic matrix

note_index = 0
for note in posterior:
    for topic in note:
        matrix[note_index, topic[0]] = topic[1]  # add topic proportions to matrix
    note_index += 1

print "saving posterior to csv"
features = pandas.DataFrame(matrix)
note_info = notes.loc[:, ['file', 'mrn', 'status']]

feature_df = pandas.concat([note_info, features], axis=1)
feature_df.to_csv("vanillaLDA_" + str(final_K) + "_features.csv")

topic_words = []
for topic in final_model.show_topics(num_topics=final_K, num_words=15):
    topic_words.append(topic)

print "create pandas data frame of the top words for each trained model"
topic_df = pandas.DataFrame({'top_words': topic_words})
topic_df.to_csv("vanillaLDA_" + str(final_K) + "topics.csv")
