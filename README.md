# Fake_and_Real_News

## Dataset : https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset

## Implementation :

### Case 1 : 
Bag of Words : Set of vectors containing count of word occurences, a simple and flexible approach in Text Classification. 
Sklearn Library : SVM and RandomForest : Optimal Classifiers for Binary Classification.

### Case 2 : 
TF-IDF : It assigns a value to a term according to its importance in the text scaled by its importance across all the texts in the data. A popular approach in NLP. 
XGBoost and LightGBM : Both are based on Gradient Boosted Decision Trees. In XGBoost, trees grow depth-wise and in LightGBM, trees grow leaf-wise. Both models had great success in enterprise applications and data science competitions. XGBoost is extremely powerful, though model training is faster in LightGBM.

### Case 3 : 
Pre-trained GloVe Embedding : GloVe = Global vectors for word representation. It is an unsupervised algorithm developed by Standford for generating word embeddings by aggregating global word-word co-occurence matrix from a corpus, which gives semantic relationships between words. Here, I have user Pretrained Word Vector of Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors, 1.42 GB) from https://nlp.stanford.edu/projects/glove/ 
Tensorflow Framework : Bi-Directional LSTM : LSTM is classic model used for NLP tasks

### Case 4 : 
PyTorch Framework : HuggingFace transformers Library 
BERT : Google's BERT (October-2018) is the transformer based method for NLP, outperforming state-of-the-art on several tasks such as QnA, language inference. It is a pre-trained deep Bi-directional Encoder Representation from transformer with Masked Language Modelling (MLM) and Next Sentence Prediction (NSP). 
RoBERTa : Facebook's RoBERTa (July-2019), robustly optimized BERT approach, advancing the state-of-the-art in self-supervised systems. It is a BERT without Next Sentence Prediction (NSP). To improve training procedure, RoBERTa removes the Next Sentence Prediction (NSP) task from BERT's pre-training and dynamic masking so that the masked token changes during training epochs.

### The Most Preferred Model : From these 4 cases, currently the RoBERTa model is the most preferred one, as it is the optimized BERT approach.
