import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
import pandas
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, f1_score

Baseline = pandas.read_csv('Baseline_pred_probs.csv')
Unigram = pandas.read_csv('Unigram_pred_probs.csv')
LDA = pandas.read_csv('LDA_pred_probs.csv')
RedLDA = pandas.read_csv('RedLDA_pred_probs.csv')

colors = cycle(['cyan', 'indigo', 'blue']) # 'blue', 'darkorange'
names = ['Baseline', 'Unigrams', 'LDA']
plt.clf()

i = 0
for model, color, model_name in zip([Baseline, Unigram, LDA], colors, names):
    print model_name
    F = f1_score(model.iloc[:,1], model.iloc[:, 3], pos_label=True, average='binary')
    precision, recall, thresholds = precision_recall_curve(model.iloc[:,1], model.iloc[:, 3], pos_label = True)
    plt.plot(recall, precision, label='Precision-recall curve of {0} (area = {1:0.2f})'
                   ''.format(model_name, F))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.legend(loc="lower left")
plt.show()