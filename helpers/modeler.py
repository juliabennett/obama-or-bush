from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x[self.column] 

def pos_tokenizer(string_of_tags):
    return string_of_tags.split()

def load_clf(model_file):
    return joblib.load(model_file)

