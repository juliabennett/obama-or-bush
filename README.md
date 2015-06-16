#obama-or-bush

Explores radio addresses given by Obama and Bush through webscraping, natural language processing, and machine learning. Everything is done in Python, with the following packages handling most of the heavy lifting: pandas, beautiful soup, scikit-learn, nltk, and moe.

##Overview 

This repository contains data and code from a ongoing project that has four goals:

1. Create a nice database storing data about radio addresses given by either Obama or Bush, including titles, dates, cleaned-up transcripts, and translations to parts of speech.

2. Define a model that can predict which of these two presidents is the speaker using text from a radio address that it's never seen before. 

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

The database was created by running the script make\_data.py, which webscrapes the official Whitehouse website and the former official Whitehouse website. It's worth noting that this compilation is nearly comprehensive, but the webscraping script does fail to catch some small number of radio addresses (and obviously doesn't include any that will happen in the future). As is always the case, this script will stop working if either of these websites changes too much. 

####The model & hyperparameter tuning.

Using standard techniques from natural language processing for author classification, I created a pretty strong baseline model without too much trouble. The script select\_model.py implements Bayesian optimization to perform an "intelligent" search to improve the choice of hyperparameters for this model. This script does not completely automate the selection process for these hyperparameters. Intead, it allows the user to choose from a list of tuned models that each realize a mean F1 score from 10-fold cross validation that's within one standard error of the best found. After this selection is made, the script saves the final model and evaluates it on unseen data.  The searching process relies on Yelp's MOE, which you can read more about [here](http://yelp.github.io/MOE/).

I ran this script and chose a model that nicely compromised between the number of features and the regularization parameter, realizing the following stats on unseen data:

* accuracy: 0.990697674419
* F1 Score: 0.987804878049
* recall: 0.975903614458
* precision: 1.0

Here's a few key components of this model: 

* Filters noise by removing numbers, punctuation, and tagging errors from the data. 
* Creates two types of features: 1) tf-idf scores from regular ol' 1-grams, and 2) tf scores from 2-grams describing parts of speech (e.g. "NOUN VERB"). 
* Selects about 1500 of these features based on their chi-squared scores. 
* Uses a support vector machine with a linear kernel. The regularization parameter is about equal to 10. 

Since Bush and Obama both use very particular opening and closing greetings, I removed the first and last sentence of each speech before creating any models. This ensured that prediction is based only on true content. 

If you would like access to the tuned and trained model, download the folder "model_files" and run the following two lines from a directory containing its contents: 

```
from modeler import * 
clf = load_clf("final_model.pkl")
```

For example, you can explore the model's coefficients and take a closer look at the final choice of hyperparameters. (This type of information will also be easily available in the finished app.)

### The app.

The app is hopefully coming soon! 