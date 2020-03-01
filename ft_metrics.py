#!/usr/bin/env python
#!/usr/local/bin/python3
# @author cpuhrsch https://github.com/cpuhrsch
# @author Loreto Parisi loreto@musixmatch.com
# On 2018/09/08 modified by Yuen-Hsien Tseng from:
# https://gist.github.com/loretoparisi/41b918add11893d761d0ec12a3a4e1aa

import argparse
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report

def parse_labels(path):
    with open(path, 'r') as f:
#        return np.array(list(map(lambda x: x[9:], f.read().split())))
        return np.array(list(map(lambda x: x[9:], f.read().split())))

def tcfunc(x, n=4): # trancate a number to have n decimal digits
    d = '0' * n
    d = int('1' + d)
# https://stackoverflow.com/questions/4541155/check-if-a-number-is-int-or-float
    if isinstance(x, (int, float)): return int(x * d) / d
    return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display confusion matrix.')
    parser.add_argument('test', help='Path to test labels')
    parser.add_argument('predict', help='Path to predictions')
    args = parser.parse_args()
    test_labels = parse_labels(args.test)
    pred_labels = parse_labels(args.predict)
    eq = test_labels == pred_labels

#    print("Accuracy: " + str(eq.sum() / len(test_labels)))   
#    print(confusion_matrix(test_labels, pred_labels))

    print("\tPrecision\tRecall\tF1\tSupport")
    (Precision, Recall, F1, Support) = list(map(tcfunc, 
        precision_recall_fscore_support(test_labels, pred_labels, average='micro')))
    print("Micro\t{}\t{}\t{}\t{}".format(Precision, Recall, F1, Support))
    (Precision, Recall, F1, Support) = list(map(tcfunc, 
        precision_recall_fscore_support(test_labels, pred_labels, average='macro')))
    print("Macro\t{}\t{}\t{}\t{}".format(Precision, Recall, F1, Support))

    exit()
'''
    try: 
        print(classification_report(test_labels, pred_labels, digits=4))
    except ValueError:
        print('May be some category has no predicted samples')
'''
