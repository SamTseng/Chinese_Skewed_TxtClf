#!/usr/bin/env python
# coding: utf-8

import pandas, numpy, sys

if __name__ == "__main__":
    print('''
Print out the F1-score of the result of BERT classifier for a data set.
  Usage:   python eval_clf.py Data_Name Test_File Predicted_File
  Example: python eval_clf.py CnonC CnonC_test.txt Output_CnonC/test_results.tsv
''')
print("sys.argv:", sys.argv)
if len(sys.argv) == 4 and sys.argv[1] != '-f':
    prog, Data_Name, Test_File, Predicted_File = sys.argv
else:
    Data_Name = 'Reuters'
    Test_File = '/Users/sam/GoogleDrive/AnacondaProjects/Shared/TxtClf_Dataset/Reuters_test_sl.txt'
    Predicted_File = '/Users/sam/Downloads/test_results_20.tsv' # not correct, just to extract Label_List

df0 = pandas.read_csv(Test_File, sep='\t', header=None)

#print("The first 5 rows of '%s':" % Test_File)
df0.head(5)

test_yL = df0[0]
#print("The first 5 categories of '%s'" % Test_File)
#print(test_yL[0:5])


# ## Set variable: Label_List based on get_labels() function
# We need to set the Label_List by ourself, 
#   because in /Users/sam/GoogleDrive/BERT/bert/Output_CnonC/test_results.tsv
#   the column order is defined by the get_labels() function
cat2num = pandas.Series(df0[0]).value_counts()
print(cat2num.sort_values(ascending=False))
Label_List = list(cat2num.keys())
print("From '%s' file: df0\nLabel_List='%s'"%(Test_File, str(Label_List)))
print("Number of Labels: " + str(len(Label_List)))


# ## Label_List should be same as that in get_labels() in run_classifier.py

# Note: 
# 1. The value of Label_List here is obtained by running the code before this line
# 2. The value of Label_List here should also be copy to get_labels() in run_classifier.py 
if Data_Name == 'CnonC':
    Label_List = ["001-營建類", "002-非營建類"]
    #Label_List = ["001-营建类", "002-非营建类"]
elif Data_Name == 'PCNews':
    Label_List = ['004-產業', '003-財經', '001-政治', '002-社會', '009-生活', '006-娛樂', '008-地方', '005-科技', '007-體育', '013-文教', '010-醫藥', '012-休閒']
elif Data_Name == 'PCWeb':
    Label_List = ['050219-網頁設計工作室', '050218-網頁設計教學', '050221-電子賀卡', '05022002-國內網站網頁搜尋', '05022007-主題搜尋', '05020701-駭客', '050209-ISP', '050210-網域註冊', '050208-網路資訊討論', '05022001-國外網站網頁搜尋', '050222-網路調查', '050207-網路安全', '050211-網站評鑑', '05022006-搜尋引擎連結', '050217-網站宣傳', '050220-搜尋引擎', '050213-網路文化', '050216-Proxy', '05021201-Plug-in', '050212-瀏覽器', '05020901-固接專線', '050214-電子商務', '05022005-檔案搜尋', '05022004-BBS文章搜尋', '050215-Intranet', '05022003-電子郵件搜尋']
elif Data_Name == 'Joke':
    Label_List = ['冷笑話', '職場笑話', '愛情笑話', '家庭笑話', '校園笑話', '其他笑話/不分類', '黃色笑話', '名人笑話', '術語笑話']
elif Data_Name == 'CTC':
    Label_List = ['Cu10', 'P52', 'Cu52', 'Cu4', 'P10', 'E53', 'Cu6', 'P1', 'E37', 'P5', 'P31', 'E56', 'Cu5', 'E57', 'E5', 'E6', 'E34', 'P73', 'P6', 'Cu31', 'E8', 'Cu3', 'Cu71', 'Cu8', 'E10', 'P4', 'P8', 'P12', 'E3', 'P13', 'P7', 'P72', 'Cu7', 'P71', 'E36', 'P32', 'E55', 'E1', 'E51', 'Cu91', 'E4', 'E35', 'Cu41', 'P2', 'E7', 'Cu1', 'E38', 'E11', 'Cu32', 'P51', 'E36a', 'E54', 'Cu2', 'Cu72', 'Cu35', 'Cu1-3', 'E86', 'E9', 'P3', 'Cu33', 'Cu51', 'Cu34', 'E52', 'P5Hi', 'P11', 'Cu1-2', 'E1-2', 'E32', 'E39', 'HK', 'P10a', 'Cu73', 'P33', 'P9', 'E79', 'Bi', 'E1a', 'E1-4', 'E83', 'E1-3', 'P53']
elif Data_Name == 'Reuters':
    ['earn', 'acq', 'crude', 'money-fx', 'interest', 'trade', 'ship', 'money-supply', 'sugar', 'coffee', 'gold', 'alum', 'cpi', 'gnp', 'cocoa', 'copper', 'iron-steel', 'reserves', 'nat-gas', 'jobs', 'veg-oil', 'ipi', 'tin', 'grain', 'wpi', 'bop', 'rubber', 'orange', 'cotton', 'gas', 'fuel', 'pet-chem', 'strategic-metal', 'livestock', 'zinc', 'carcass', 'income', 'lead', 'lumber', 'heat', 'potato', 'dlr', 'tea', 'lei', 'platinum', 'housing', 'retail', 'meal-feed', 'nickel', 'jet', 'instal-debt', 'cpu']
else:
    print('Incorrect Data_Name!') 
    exit()
print("From copy of get_labels() in BERT's run_classifier.py\nLabel_List='%s'"%str(Label_List))
print("Number of Labels:" + str(len(Label_List)))


# ## Read the prediction file
df = pandas.read_csv(Predicted_File, sep='\t', names=Label_List)

def maxLabel(List, labels=Label_List):
    maxi = 0;
    for i in range(len(List)):
        if List[i]>List[maxi]: maxi = i
    return labels[maxi]

df['label'] = df.aggregate(func=maxLabel, axis='columns')
#df.head(5)

cat2num = pandas.Series(df['label']).value_counts()
#print(cat2num.sort_values(ascending=False))

# The next function does the similar report, but assume multiple columns
#   and each column (category) has 0 or 1 values
# See: https://towardsdatascience.com/multi-label-text-classification-with-scikit-learn-30714b7819c5
def show_stats(All_DF):
    df_label = All_DF.drop(['text'], axis=1)
    counts = []
    categories = list(df_label.columns.values)
    print(categories)
    for c in categories:
        counts.append((c, df_label[c].sum()))
    df_stats = pandas.DataFrame(counts, columns=['category', 'number_of_texts'])
    print(df_stats)


pred_yL = df['label']
#print(type(pred_yL))

# ## Now use test_yL and pred_yL to calculate the metrics
# Now we have test_yL and pred_yL
from sklearn import preprocessing, metrics

LabEncoder = preprocessing.LabelEncoder() # convert label name to label int
pred_y = LabEncoder.fit_transform(pred_yL)
test_y = LabEncoder.fit_transform(test_yL)
# we do not need test_y and pred_y anymore, we need LabEncoder.classes_
Num_Classes = len(LabEncoder.classes_)


print("Num of Classes (Categories or Labels):", Num_Classes)
print(type(LabEncoder.classes_),"Label Names [:2]:", LabEncoder.classes_[:2])
print("Label Names transformed[:2]:", LabEncoder.transform(LabEncoder.classes_[:2]))
print("Label inverse transform [0, 1]:", LabEncoder.inverse_transform([0, 1]))


# ## Evaluate the prediction performance 
# based on variables: LabEncoder.classes_, test_yL, and pred_yL.


# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
import itertools # replace this line by next line on 2019/01/03, because cannot find itertools for Python 3.6.7
#import more_itertools # On 2019/08/18 with Python 3.6.7, this package did not work, but the above one works!
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm) # print out consufion matrix

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
# remark the use more_itertools on 2019/08/18 because it did not work anymore for Python 3.6.7.
# Replace the above line by the next line on 2019/01/03, because cannot find itertools for Python 3.6.7
#    for i, j in more_itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# use global variables:  LabEncoder.classes_
def show_confusion_matrix_plot(test_yL, predictions):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(test_yL, predictions)
    numpy.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=LabEncoder.classes_ ,
                      title='Confusion matrix, without normalization')
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=LabEncoder.classes_ , normalize=True,
                      title='Normalized confusion matrix')

    plt.show()


def tcfunc(x, n=4): # trancate a number to have n decimal digits
    d = '0' * n
    d = int('1' + d)
# https://stackoverflow.com/questions/4541155/check-if-a-number-is-int-or-float
    if isinstance(x, (int, float)): return int(x * d) / d
    return x


# http://scikit-learn.org/stable/modules/model_evaluation.html
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def show_Result(test_yL, predictions):
    #print(predictions[:5], "\n")

    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    print("\n\nMicroF1 = %0.4f, MacroF1=%0.4f" %
        (metrics.f1_score(test_yL, predictions, average='micro'),
         metrics.f1_score(test_yL, predictions, average='macro')))
    # https://stackoverflow.com/questions/455612/limiting-floats-to-two-decimal-points

# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    print("\tPrecision\tRecall\tF1\tSupport")
    (Precision, Recall, F1, Support) = list(map(tcfunc, 
        precision_recall_fscore_support(test_yL, predictions, average='micro')))
    print("Micro\t{}\t{}\t{}\t{}".format(Precision, Recall, F1, Support))
    (Precision, Recall, F1, Support) = list(map(tcfunc, 
        precision_recall_fscore_support(test_yL, predictions, average='macro')))
    print("Macro\t{}\t{}\t{}\t{}\n\n".format(Precision, Recall, F1, Support))
    
    if True:
#    if False:
        print(confusion_matrix(test_yL, predictions))
        print()
        try: 
            print(classification_report(test_yL, predictions, digits=4))
        except ValueError:
            print('May be some category has no predicted samples')
        #show_confusion_matrix_plot(test_yL, predictions)

show_Result(test_yL, pred_yL)

#get_ipython().system('python -V')
