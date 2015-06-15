#obama-or-bush

Explores radio addresses given by Obama and Bush through webscraping, natural language processing, and machine learning. Everything is done in Python, with the following packages handling most of the heavy lifting: pandas, beautiful soup, scikit-learn, nltk, and moe.

##Overview 

This repository is describing a project that has four goals:

1. Provide a nice database storing data about radio addresses given by either Obama or Bush, including titles, dates, cleaned-up transcripts, and translations to parts of speech.

2. Create a model that can predict which of these two presidents is the speaker using text from a radio address that it's never seen before. 

3. Try out Bayesian optimization for hyperparameter tuning using Yelp's Metric Optimization Engine (MOE). 

4. Develop an app that allows users to try to beat (or at least tie...) this model. Users will be able to explore the features on which the model bases each prediction, hopefully getting a better understanding of how the model works and having a little fun.


##Details

####The data.

The sqlite database obama\_or\_bush.db contains a table called radio\_addresses storing the data. This table has the following columns:

* id: an integer assigned to uniquely identify each radio address
* date: the date the radio address was released
* title: the title provided for the radio address
* speaker: the name of the president giving the radio address (either "obama" or "bush")
* speech: a transcript of the speech (all unicode symbols were replaced in the standard way so that the text is super clean)
* pos: the result of translating each word and punctuation in the transcripts to its part of speech using nltk's part of speech tagger (which does produce a few errors along the way)

The database was created by running the script make\_data.py, which webscrapes the official Whitehouse website and the former official Whitehouse website. It's worth noting that this compilation is nearly comprehensive, but the webscraping script does fail to catch some small number of radio addresses. 

####The model & hyperparameter tuning.

Using standard techniques from natural language processing for author classification, I created a pretty strong baseline model. The script select\_model.py implements Bayesian optimization to perform an "intelligent" search for the best hyperparameters for this model. The selection process is not completely automated, as the user is ultimately asked to choose from a list of models that each realize a mean F1 score from 10-fold cross validation that's within one standard error of the best found. This search relies on Yelp's MOE, which you can read more about [here](http://yelp.github.io/MOE/). 

After running this script and choosing a model that nicely compromised between the number of features and the regularization parameter, I tested the result on unseen data and observed the following stats: 

* accuracy: 0.990697674419
* F1 Score: 0.987804878049
* recall: 0.975903614458
* precision: 1.0

Here's a few key components of this model: 

* Filters noise by removing numbers, punctuation, and tagging errors from the data. 
* Creates two type of features: 1) tf-idf scores from regular ol' 1-grams, and 2) tf scores from 2-grams describing parts of speech (e.g. "NOUN VERB"). 
* Selects about 1500 of these features based on their chi-squared scores. 
* Uses a support vector machine with a linear kernel and a regularization parameter about equal to 10. 

Since Bush and Obama both use very particular opening and closing greetings, I removed the first and last sentence of each speech before building a model. This ensured that prediction is based only on true content. 

If you would like access to the tuned and trained model, download the folder "model_files" and run the following two lines from a directory containing its contents: 

```
from modeler import * 
clf = load_clf("final_model.pkl")
```


### The app.

The app is hopefully coming soon! 