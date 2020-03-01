#!/usr/bin/env python
# On 2018/09/08 written by Yuen-Hsien Tseng

import sys, re
prog, file = sys.argv
for line in open(file).read().split("\n"):
    if (re.match('^It take', line) or 
        re.match('^\tPrecision', line) or
        re.match('^Micro', line) or
        re.match('^Macro', line) or
        re.match('^NB', line) or
        re.match('^LR', line) or
        re.match('^SVM', line) or
        re.match('^RF', line) or
        re.match('^Xgb', line) or
        re.match('^NN', line) or
        re.match('.+\.shape', line) or
        re.match('^Trainable', line)):
            print(line)