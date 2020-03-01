#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written on 2019/03/01 by Yuen-Hsien Tseng
# Given a train/test text file, output the number of documents and 
#   average, max, and min characters for each category
# Ex: python3 dataset_stat.py Datasets/joke_train.txt 
#   See the bottom for some running examples.

import sys
f = sys.argv[1]
fh = open(f, 'r', encoding='utf8')
cat2DocNum = {}
for line in fh.readlines():
    (cat, joke) = line.split('\t')
    length = len(joke)
    if cat in cat2DocNum:
        cat2DocNum[cat].append(length)
    else:
        cat2DocNum[cat] = [length]

TotalSum = 0; TotalNum = 0
print("Category\tNum of Doc\tAverage\tMax\tMin")
for cat, DocLen in cat2DocNum.items():
    i, sum, max, min = 0, 0, 0, 100000
    for length in DocLen:
        TotalSum += length # overall length
        sum += length
        max = length if max < length else max
        min = length if min > length else min
        i += 1
    TotalNum += len(DocLen)
    print("%s\t%d\t%d\t%d\t%d"%(cat, i, round(sum/i), max, min))
print("Total Doc=%d, Average Char.=%d" % (TotalNum, TotalSum/TotalNum))

'''
On 2019/10/03 add some codes and re-run:
$ python dataset_stat.py Datasets/joke_train.txt
Category	Num of Doc	Average	Max	Min
冷笑話	553	97	828	3
家庭笑話	239	117	348	31
其他笑話/不分類	215	104	1720	7
校園笑話	228	130	759	14
愛情笑話	318	104	841	15
職場笑話	376	139	646	29
名人笑話	152	111	713	12
術語笑話	146	121	1140	14
黃色笑話	162	166	855	8
Total Doc=2389, Average Char.=117

$ python dataset_stat.py Datasets/joke_test.txt
Category	Num of Doc	Average	Max	Min
冷笑話	237	103	771	10
名人笑話	66	119	518	19
家庭笑話	102	126	764	33
其他笑話/不分類	93	106	610	13
職場笑話	161	152	648	24
校園笑話	98	105	371	16
愛情笑話	136	102	644	21
術語笑話	63	137	1289	20
黃色笑話	69	175	726	4
Total Doc=1025, Average Char.=121

$ python dataset_stat.py Datasets/joke_All.txt
Category	Num of Doc	Average	Max	Min
冷笑話	790	99	828	3
家庭笑話	341	120	764	31
其他笑話/不分類	308	105	1720	7
校園笑話	326	122	759	14
愛情笑話	454	103	841	15
職場笑話	537	143	648	24
名人笑話	218	114	713	12
術語笑話	209	126	1289	14
黃色笑話	231	169	855	4
Total Doc=3414, Average Char.=118

'''