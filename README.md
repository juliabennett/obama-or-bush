
#obama-or-bush

Explores radio addresses given by Obama and Bush through webscraping, natural language processing, and machine learning. Everything is done in Python, with the following packages handling most of the heavy lifting: pandas, Beautiful Soup, scikit-learn, NLTK, and MOE.

Check out the Flask app: [juliaben.net/t/who-said-it](http://juliaben.net/t/who-said-it/). This includes interactive plots made with D3.js. 

##Overview 

This repository contains data and code from a project that had four goals:

1. Create a nice database storing data about radio addresses given by either Obama or Bush, including titles, dates, cleaned-up transcripts, and translations to parts of speech.

2. Define a model that can identify which of these two presidents is the speaker using text from a radio address that it's never seen before. 

3. Try out Bayesian optimization for hyperparameter tuning using Yelp's Metric Optimization Engine (MOE). 

4. Develop an app that lets users to try to beat (or at least tie...) this model and visually explore how it makes decisions. 


##Details

####The data.

The sqlite database obama\_or\_bush.db contains a table called radio\_addresses storing the primary data for this project. This table has observations for 715 radio addresses and includes the following columns:

* id: an integer assigned to uniquely identify each radio address
* date: the date the radio address was released
* title: the title provided for the radio address
* speaker: the name of the president giving the radio address (either "obama" or "bush")
* speech: a transcript of the speech (all unicode symbols were replaced in the standard way so that the text is super clean)
* pos: the result of translating each word and punctuation in the transcript to its part of speech using nltk's part of speech tagger (which does produce a few errors along the way)

The script make\_data.py created this table by webscraping the official Whitehouse website and the former official Whitehouse website. 

It's worth noting that the compilation is nearly comprehensive, but the webscraping script does fail to catch some small number of radio addresses (and obviously doesn't include any that will happen in the future). As is always the case, the script will stop working if either of these websites changes too much. Also, there appear to be a few entries in this table that are mistakes due to imprecise scraping techniques and poor website designs, but these are very rare (likely less than 5 entries out of over 700) and should not significantly effect any analysis. 

####The model & hyperparameter tuning.

Using standard techniques from natural language processing for author classification, I created a pretty strong baseline model without too much trouble. The script select\_model.py implements Bayesian optimization to perform an "intelligent" search to improve the choice of hyperparameters for this model. This script does not completely automate the selection process for these hyperparameters. Instead, it allows the user to choose from a list of tuned models that each realize a mean F1 score from 10-fold cross validation that's within one standard error of the best found. After this selection is made, the script saves the final model and evaluates it on unseen data.  The searching process relies on Yelp's MOE, which you can read more about [here](http://yelp.github.io/MOE/).

I ran this script and chose a model that nicely compromised between the number of features and the regularization parameter, realizing the following stats on unseen data:

* accuracy: 0.990697674419
* F1 score: 0.987804878049
* recall: 0.975903614458
* precision: 1.0

Here's a few key components of this model: 

* Avoids noise by predicting on data that was preprocessed by removing all numbers, punctuation, and tagging errors. 
* Creates two types of features: 1) tf-idf scores from regular ol' 1-grams, and 2) tf scores from 2-grams describing parts of speech (e.g. "NOUN VERB"). These are filtered by setting maximum and minimum document frequencies. 
* Selects about 1300 of these features based on their chi-squared scores. 
* Uses a support vector classifer with a linear kernel. The regularization parameter is about equal to 10. 

Since Bush and Obama both use very particular opening and closing greetings, I also removed the first and last sentence of each speech during preprocessing. This ensured that prediction is based only on true content. 

Note that a standard 70/30 split was used to break the data into training and testing sets after preprocessing. The preprocessed and split data can be found in the tables data\_train and data\_test in the database obama\_or\_bush.db.

If you would like more details about the model, check out [the app](http://juliaben.net/t/who-said-it/) or follow the instructions at the bottom of this page to load it from the included pickle files.


####The app.

As mentioned above, the app can be found here: [juliaben.net/t/who-said-it](http://juliaben.net/t/who-said-it/). It was built with Flask and D3.js. All data and code used by the app is available in the folder /game.  The data was created by running the script populate\_game\_database.py located in the the folder /helpers. 


####Build it yourself. 

If you're interested in building parts of this project yourself, you have a few options after downloading this repository: 

1. Use the database obama\_and\_bush.db to do your own analysis. It could be interesting to add radio addresses from other presidents and train a new model that does multiclass classification. 

2. You can see the hyperparameter search in action and choose your own model by running the following code in the main directory: 


    ```
    python select_model.py obama_or_bush.db 
    ```

    In addition to pickling the model you select, this will recreate tables data_train and data_test in obama_or_bush.db. A model produced by this script is designed to make predictions from the processed data in these tables. Since the contents of the folder /model\_files will be replaced with new pickle files, this model can be loaded into memory by following the next set of instructions. 

3. If you'd like to use my model or just explore it more thoroughly, you can run the following two lines from the directory /helpers: 


    ```
    from modeler import * 
    clf = load_clf("../model_files/final_model.pkl")
    ```

    The model will then be accessible through the variable `clf` using standard sci-kit learn tools. It's built to make predictions on the processed data from the tables data\_train and data\_test in obama\_or\_bush.db. 

4. You might want to recreate and/or update the radio_addresses table in the database obama_or_bush.db (possibly after improving my webscraping script). This can be completed by running the following code in the main directory: 

    ```
    python make_data.py obama_or_bush.db 
    ```

5. Finally, the data used in the app can be reproduced by running the following code from the main directory: 


    ```
    python helpers/populate_game_database.py obama_or_bush.db game/data.db 
    ```

Please let me know if there's something else I can include to help you with whatever you're working on. I'm very open to questions and suggestions! 