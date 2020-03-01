#!/bin/bash
# On 2018/09/08 modified by Yuen-Hsien Tseng from:
#   https://gist.github.com/loretoparisi/41b918add11893d761d0ec12a3a4e1aa
# $ ./ft_metrics.sh CnonC Datasets/CnonC_test_ft.txt Out/Cnonc_model.bin Out2

NAME=$1
DATA=$2
MODEL=$3
DIR=$4
#echo Normalizing dataset: $DATA ...
#awk 'BEGIN{FS=OFS="\t"}{ $1 = "__label__" tolower($1) }1' $DATA > $DIR/norm
#awk 'BEGIN{FS=OFS="\t"}{ $1 = tolower($1) }1' $DATA > $DIR/norm
#awk 'BEGIN{FS=OFS="\t"}1' $DATA > $DIR/norm
cut -f 1 -d$'\t' $DATA > $DIR/${NAME}_TrueLabels.txt

#echo Calculating predictions...
fastText/fasttext predict $MODEL $DATA > $DIR/${NAME}_PredictLabels.txt
./ft_metrics.py $DIR/${NAME}_TrueLabels.txt $DIR/${NAME}_PredictLabels.txt