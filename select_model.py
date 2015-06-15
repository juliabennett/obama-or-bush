import sqlite3
import re
import sys
import pandas as pd
import numpy as np
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.externals import joblib
from moe.easy_interface.experiment import Experiment
from moe.easy_interface.simple_endpoint import gp_next_points
from moe.optimal_learning.python.data_containers import SamplePoint

# Removes first sentence, last sentence, and all numbers.
def process_speech(speech):
    sentences = PunktSentenceTokenizer().tokenize(speech)
    stripped = " ".join(sentences[1:-1])
    return re.sub("[0-9]", "", stripped)

# Removes punctuation, first sentence, last sentence, 
#   and all sentences that have a tagging error.
def process_pos(string_of_tags):
    sentences = PunktSentenceTokenizer().tokenize(string_of_tags)
    stripped = " ".join([sent for sent in sentences[1:-1] 
                         if "NONE" not in sent])
    return re.sub(" [^a-zA-Z ]+", "", stripped) 

def get_and_clean_data(database_name):
    con = sqlite3.connect(database_name)
    from_sql = pd.read_sql("SELECT speech, pos, speaker FROM radio_addresses", con)
    data = pd.DataFrame({
        "speech": [process_speech(speech) 
                   for speech in from_sql["speech"]],
        "pos": [process_pos(string_of_tags) 
                for string_of_tags in from_sql["pos"]],
        "speaker": [1 if speaker == "obama" else 0 
                    for speaker in from_sql["speaker"]]
    })
    return data

def split_data(data):
    train, test = train_test_split(data, test_size=0.3, random_state=512)
    data_train = pd.DataFrame(train, columns=data.columns)
    data_test = pd.DataFrame(test, columns=data.columns)
    return data_train, data_test  

def round_to_int(number):
    return int(round(number, 0))

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x[self.column] 

def pos_tokenizer(string_of_tags):
    return string_of_tags.split()

# Defines a classifier based on a list describing a set of hyperparameters. 
def create_model(param_list):
    # Define two pipelines that will be used as building blocks. 
    speech_pipeline = Pipeline([
        ("prepare", ItemSelector("speech")),           
        ("create", TfidfVectorizer(stop_words="english"))
    ])
    pos_pipeline = Pipeline([
        ("prepare", ItemSelector("pos")),
        ("create", TfidfVectorizer(use_idf=False, lowercase=False,
                                   tokenizer=pos_tokenizer))
    ])

    # Start a dictionary of hyperparameters. 
    param_dict = {
        'union__speech__create__ngram_range': (1, round_to_int(param_list[0])), 
        'union__speech__create__max_df': param_list[1],
        'union__speech__create__min_df': param_list[2],
        'select__k': round_to_int(param_list[3]),
        'model__C': param_list[4],
    }

    # Choose features to include. Update hyperparameter dictionary if needed.
    feature_list = [("speech", speech_pipeline)] 
    if round_to_int(param_list[5]): 
        feature_list.append(("pos", pos_pipeline))
        param_dict['union__pos__create__ngram_range'] = (2, round_to_int(param_list[6]))
        param_dict['union__pos__create__min_df'] = param_list[7]
        
    # Define classifier and set its hyperparameters. 
    clf = Pipeline([
        ("union", FeatureUnion(transformer_list=feature_list)),
        ("select", SelectKBest(chi2)), 
        ("model", LinearSVC(class_weight="auto", random_state=415))
    ])
    clf.set_params(**param_dict)

    # Return hyperparameter dictionary and tuned model. 
    return param_dict, clf

# Computes F1 scores from 10-fold cross validation.
def score_model(clf, data_train, targets_train):
    try: 
        return cross_val_score(clf, data_train, targets_train, 
                               scoring="f1", cv=10, n_jobs=-1)
    except: 
        # Fitting these models can fail when k in SelectKBest is set too big.
        clf.set_params(select__k="all") 
        return cross_val_score(clf, data_train, targets_train, 
                               scoring="f1", cv=10, n_jobs=-1)

def compute_SEM(scores):
    return np.std(scores)/np.sqrt(len(scores))

# Evaluates objective function to be minimimized during hyperparameter search. 
def objective(score, status_quo):
    # Follow MOE's guidelines to define a good objective function.
    return -1*(score/status_quo - 1)

# Implements a Bayesian hyperparameter search and records stats for each resulting model.
def search_models(status_quo, data_train, targets_train):
    # Define the range of each hyperparameter. 
    exp = Experiment([
        [1, 2],         # max of ngram range (gets rounded to nearest integer)
        [.7, 1],        # max_df for ngrams
        [0, .1],        # min_df for ngrams
        [250, 2000],    # maximum number of features (gets rounded to nearest integer)
        [.01, 100],     # regularization parameter
        [0, 1],         # include parts of speech features if and only if value is >= .5
        [2, 3],         # max of ngram range for parts of speech (gets rounded to nearest integer)
        [0, .1]         # min_df for ngrams from parts of speech
    ])

    # Run the search. 
    # Use 20 more iterations than the 80 suggested by MOE to compensate for lack of historical data. 
    search_results = []
    for _ in range(100):
        # Choose the next model based on results from previous models.
        param_list = gp_next_points(exp)[0]
        param_dict, clf  = create_model(param_list)

        # Compute F1 scores using cross validation and save the result. 
        scores = score_model(clf, data_train, targets_train)
        mean_score = np.mean(scores)
        search_results.append(
            (clf, param_dict, mean_score, compute_SEM(scores))
        )

        # Plug F1 scores into objective function and update historical data.
        objectives = [objective(score, status_quo) for score in scores]
        sample_point = SamplePoint(param_list, np.mean(objectives), np.var(objectives))
        exp.historical_data.append_sample_points([sample_point])

        #Print most recent F1 score so that user can see progress. 
        print "mean F1 score:", mean_score

    return search_results

#Selects all models that scored within one standard error of the best score.
def select_models(search_results):
    best_score = max([result[2] for result in search_results])
    SE = min([result[3] for result in search_results
              if result[2] == best_score])
    return [result for result in search_results
            if result[2] >= best_score - SE]

def search_select_evaluate(database_name):
    print "Processing data..."
    data = get_and_clean_data(database_name)
    data_train, data_test = split_data(data)
    targets_train = data_train["speaker"]
    targets_test = data_test["speaker"]

    print "Obtaining a 'status quo' model..."
    # Choose reasonable (but probably not perfect) hyperparameters. 
    param_list = [2, .85, .05, 1000, 1.0, 0, 2, 0] 
    _, clf  = create_model(param_list)
    scores = score_model(clf, data_train, targets_train)
    status_quo = np.mean(scores)
    print "Mean F1 score from cross validation of 'status quo' model:", status_quo

    print "Running hyperparameter search..."
    search_results = search_models(status_quo, data_train, targets_train)
    best_results = select_models(search_results)

    print "\nAll of the following models scored within one standard error of the best score:"
    for i, result in enumerate(best_results):
        print "\n%d." % i
        print "Mean F1 score from cross validation:", result[2]
        print "Hyperparameters:", result[1]
    print "\nInput the number of a model to save and evaluate on testing set." 
    print "Enter 'none' to exit without choosing a model."
    selection = raw_input('> ')
    if selection == "none": 
        return

    print "\nTraining and saving the selected model..."
    selected_index = int(selection)
    clf = best_results[selected_index][0]
    try: 
        clf.fit(data_train, targets_train)
    except: 
        # Fitting the model may fail if k in SelectKBest was set too high. 
        clf.set_params(select__k="all") 
        clf.fit(data_train, targets_train)
    joblib.dump(clf, "final_model.pkl")
    print "Model saved as 'final_model.pkl' in working directory with auxilary files."

    print "Evaluating model on testing set..."
    predicted = clf.predict(data_test)
    print "Accuracy:", accuracy_score(targets_test, predicted)
    print "F1 Score:", f1_score(targets_test, predicted)
    print "Recall:", recall_score(targets_test, predicted)
    print "Precision:", precision_score(targets_test, predicted)

if __name__ == '__main__':
    try: 
        database_name = sys.argv[1]
    except IndexError:
        print "usage: select_model.py database_name"
        sys.exit("\nExecution failed: user must provide location of sql database.")
    search_select_evaluate(database_name)
