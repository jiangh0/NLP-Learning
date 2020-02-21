import pandas as pd
import numpy as np
import os

path = "./sentiment-analysis-on-movie-reviews/train.tsv"
train_data = pd.read_csv(path, delimiter="\t")
path = "./sentiment-analysis-on-movie-reviews/test.tsv"
test_data = pd.read_csv(path, delimiter="\t")
# ngram-vectorizer = CountVectorizer()
print(train_data)
print(test_data)