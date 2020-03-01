# Chinese_Skewed_TxtClf

Chinese text classification datasets and their machine-learning based classifiers described in a comparative article.

Datasets are (details of the datasets can be found in the article listed below): 
1. WebDes
2. News
3. CTC
4. CnonC

Classifiers: 
1. Naive Bayes (NB)
2. Support Vector Machine (SVM)
3. Random Forest (RF)
4. Single hidden-layer neural network (NN)
5. Convolutional Neural Networks (CNN)
6. Recurrent Convolutional Neural Networks (RCNN)
7. Facebook's fastText
8. Bidirectional Encoder Representations from Transformers (BERT)

## 1. Description of Files:
1. Datasets: datasets mentioned above.
2. BERT_txtclf: a folder for running BERT classifier.
3. BERT_txtclf_HowTo.docx: a document describing how to run the BERT classifier for the datasets.
4. TxtClfer.ipynb: Self-explained Jupyter Notebook for NB, SVM, NN, CNN, RCNN. You can save it into TxtClfer.py for running in command mode.
5. fastText_run_log.txt: a document and log file to describe how to run fastText classifier for the datasets.
6. ft_metrics.sh: batch execution file to run fastText.
7. ft_metrics.py: code required by the above batch execution file.

## 2. To cite this datasets, source codes, or experiment results:
Yuen-Hsien Tseng, "The Feasibility of Automated Topic Analysis: An Empirical Evaluation of Deep Learning Techniques Applied to Skew-Distributed Chinese Text Classification"
under review by Journal of Educational Media & Library Sciences.
