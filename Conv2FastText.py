# On 2018/08/31 written by Sam Tseng
# To convert classification texts into the format for FastTtext
# Usage Example:
# $ python3 Conv2FastText.py CnonC_train.txt 0 > CnonC_train_ft.txt
# $ python3 Conv2FastText.py CnonC_test.txt 0 > Cnonc_test_ft.txt
# for eprint(), see:　https://stackoverflow.com/questions/5574702/how-to-print-to-stderr-in-python
from __future__ import print_function
import sys, time, re
from nltk.stem import PorterStemmer, WordNetLemmatizer
import jieba

time1 = time.time()

def eprint(*args, **kwargs): # print to stderr 
    print(*args, file=sys.stderr, **kwargs)

# After removing punctuations (adding the next statement),
#   The performance jump from MicroF1=0.85 to 0.9 for CnonC dataset!
#StopWords = '''
#. , ? ! @ # $ % ^ & * ( ) - + = ~ ` \ [ ] { } ' " ; : / | < > 
#。 ， ？ ！  ＠  ＃  ＄  ％  ︿  ＆  ＊  （  ） ─  ＋  ﹍  ＝ 
#； ： /  ～  ┘  ┌   「  」  『  』 〈   〉 ．  、  …  “   ” 
#1 2 3 4 5 6 7 8 9 0
#'''.split()
#StopWords.extend([' ', '\t', '\n'])

StopWords = '''
1 2 3 4 5 6 7 8 9 0 ０ １ ２ ３ ４ ５ ６ ７ ８ ９ 
的 是 了 和 與 及 或 於 也 並 之 以 在 另 又 該 由 但 仍 就
都 讓 要 把 上 來 說 從 等 
我 你 他 妳 她 它 您 我們 你們 妳們 他們 她們 
並有 並可 可以 可供 提供 以及 包括 另有 另外 此外 除了 目前 現在 仍就 就是 
'''.split()
# StopWords.extend(['　', '■']) # these punctuations belong to r'\W'
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/stop_words.py
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
StopWords.extend(list(ENGLISH_STOP_WORDS))
StopWords.extend('''said told '''.split())

# The English punctuations are from: https://keras.io/preprocessing/text/
PunctuationStr = '''
!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
！＃＄％＆\、（）＊＋，。/：；「」『』　■．．・…’’“”〝〞‵′。
''' # there is a Chinese white space before '■'
Punctuations = [x for x in PunctuationStr]
Punctuations.extend([' ', '\t', '\n'])

termFile = ''
if len(sys.argv) >= 4:
    (infile, lowtf, termFile) = sys.argv[1:4]
else:
    (infile, lowtf) = sys.argv[1:3]
lowtf = int(lowtf)

def clean_text(text): 
    '''
    Given a raw text string, return a clean text string.
    Example: 
        input:  "Years  passed. 多少   年过 去 了 。  "
        output: "years passed.多少年过去了。"
    '''
    text = text.lower() # 'years  passed. 多少   年过 去 了 。'
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    # Next line will remove punctuations. \w matches Chinese characters
    #text = re.sub('\W', ' ', text) # 'Years passed  多少 年过 去 了  '
    # Next line will remove redundant white space for jeiba to cut
    text = re.sub('\s+([^a-zA-Z0-9.])', r'\1', text) # years passed.多少年过去了。
# see: https://stackoverflow.com/questions/16720541/python-string-replace-regular-expression
    text = text.strip(' ')
    return text

ps = PorterStemmer()
wnl = WordNetLemmatizer()

def clean_words(words):
#    print("After jieba.lcut():", words)
#    WL = [ w 
#    WL = [ ps.stem(w)
    WL = [ wnl.lemmatize(w)
            for w in words if (not re.match('\s', w)) and 
                (w not in StopWords) and
                (w not in Punctuations) and 
                (not re.match('^\W$', w)) and
                (not re.match('^\d+$', w)) and
                (not re.match('^[a-z_]$', w))
         ]
    return WL

w2f = {}
texts = []
i = 0
for line in open(infile, encoding ='utf8').read().split('\n'):
    if line == "": continue
    category, doc = line.split("\t")
    words = jieba.cut(clean_text(doc), cut_all=False)
    words = clean_words(words)
    texts.append(words)
    i += 1
    for w in words:
        if w in w2f:
            w2f[w] += 1
        else:
            w2f[w] = 1

if termFile != '':
    out = open(termFile, 'w', encoding='utf8')
    #for key, value in sorted(w2f.iteritems(), key=lambda (k,v): (v,k), reverse=True):
    for key, value in sorted(w2f.items(), key=lambda x: x[1], reverse=True):
        out.write("%s\t%s\n" % (key, value))
    out.close()

i = 0
for line in open(infile, encoding ='utf8').read().split('\n'):
    if line == "": continue
    category, doc = line.split("\t")
#    words = jieba.cut(clean_text(doc), cut_all=False)
#    words = clean_words(words)
    words = texts[i]
    i += 1
    docstr = " ".join([ w for w in words if w2f[w]>=lowtf ])
    label = ["__label__"+c for c in category.split(",")]
    label = " ".join(label)
    print(label + "\t" + docstr)
# print to stderr output
eprint("It takes", round(time.time()-time1), "seconds")