import re
import sys
import sqlite3
import pandas as pd
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk import pos_tag, word_tokenize
from scipy.sparse import hstack, csr_matrix, diags

from modeler import *  

# Joins processed testing data with the original raw data.
def get_data(con_in):
	command = " ".join([
		"SELECT",
		"radio_addresses.id, speech, processed_speech, processed_pos, speaker_num",
		"FROM radio_addresses",
		"JOIN data_test",
		"ON radio_addresses.id=data_test.id",
		"WHERE radio_addresses.id != 242 AND radio_addresses.id != 681", # These two speeches are data errors. 
		"ORDER BY radio_addresses.id"
	])
	return pd.read_sql(command, con_in)

# Translates part of speech tags to nicely formatted text. 
def create_pretty_pos_pairs(pos_pair):
	pos_dict = {
		"CC": "conjunction",
		"CD": "number",
		"DT": "determiner",
		"EX": "existential",
		"FW": "foreign",
		"IN": "preposition",
		"JJ": "adjective",
		"JJR": "comparative adjective",
		"JJS": "superlative adjective",
		"MD": "modal",
		"NN": "singular noun",
		"NNS": "plural noun",
		"NNP": "singular proper noun",
		"NNPS": "plural proper noun",
		"PDT": "predeterminer",
		"POS": "possessive ending",
		"PRP": "personal pronoun",
		"PRP\$": "possessive pronoun",
		"RB": "adverb",
		"RBR": "compariative adverb",
		"RBS": "superlative adverb",
		"RP": "particle",
		"TO": "to",
		"UH": "interjection",
		"VB": "base verb",
		"VBD": "past verb",
		"VBG": "present particple verb",
		"VBN": "past participle verb",
		"VBP": "non-3rd person sing present verb",
		"VBZ": "3rd person singular present verb",
		"WDT": "wh-determiner",
		"WP": "wh-pronoun",
		"WP\$": "possessive wh-pronoun",
		"WRB": "Wh-adverb",
	}  
	new_pos_pair = "(" + pos_pair.replace(" ", ", ") + ")"
	for pos_key, pos_val in pos_dict.items():
		exp = "|".join(["(?<= )" + pos_key + "(?=\))",
						"(?<=\()" + pos_key + "(?=,)"])
		new_pos_pair = re.sub(exp, pos_val, new_pos_pair)
	return "POS Pair: " + new_pos_pair

def create_pretty_word(word):
	return 'Word: "' + word + '"'

# Extracts a list of feature indices and a list of feature names from imported model. 
def get_features(clf): 
	speech_create = clf.named_steps["union"].get_params()["speech"].named_steps["create"]
	pos_create = clf.named_steps["union"].get_params()["pos"].named_steps["create"]

	speech_vocab = dict([
		(item[1], [item[0], create_pretty_word(item[0])]) 
		for item in speech_create.vocabulary_.items()
	])
	pos_vocab = dict([
		(item[1] + len(speech_vocab), [item[0], create_pretty_pos_pairs(item[0])])
		for item in pos_create.vocabulary_.items()
	])
	extended_vocab = dict(speech_vocab.items() + pos_vocab.items())

	support = clf.named_steps["select"].get_support()
	support_indices = [i for i, logical in enumerate(support) if logical == True]
	feature_names = [extended_vocab[key] for key in support_indices]
	
	return support_indices, feature_names

def determine_feature_type(long_name):
	if "Word:" in long_name: 
		return "word"
	if "POS Pair:" in long_name:
		return "pos"
	return "intercept"

# Creates and stores a table describing model coefficients.
def populate_coef_data(clf, con_out):	
	_, feature_names = get_features(clf)
	coef_list = clf.named_steps["model"].coef_[0]
	feature_type = [determine_feature_type(name[1]) for name in feature_names]

	coef_df = pd.DataFrame({
		"short_name": [name[0] for name in feature_names],
		"long_name": [name[1] for name in feature_names],
		"value": coef_list,
		"feature_type": feature_type
	})
	coef_df.to_sql("coefs", con_out, index=False, if_exists="replace")

# Creates and stores a table describing feature contributions for individual radio addresses. 
def populate_contrib_data(clf, data, con_out):
	support_indices, feature_names = get_features(clf)

	speech_pipeline = clf.named_steps["union"].get_params()["speech"]
	pos_pipeline = clf.named_steps["union"].get_params()["pos"]
	feature_matrix = csr_matrix(hstack([
		speech_pipeline.transform(data), 
		pos_pipeline.transform(data)
	]))
	selected_matrix = feature_matrix[:,support_indices]

	model = clf.named_steps["model"]
	coef_list = model.coef_[0]
	coef_matrix = diags([coef for coef in coef_list], offsets=0)
	contrib_matrix = selected_matrix * coef_matrix

	tups = []
	intercept = model.intercept_[0]
	for i, speech_id in enumerate(data["id"]):
		row = contrib_matrix[i, :].toarray()[0]
		new_tups = [(speech_id, feature_names[j][0], feature_names[j][1], cont) 
					 for j, cont in enumerate(row) if cont != 0]
		new_tups.append((speech_id, "FIXED INTERCEPT", "FIXED INTERCEPT", intercept))
		tups += new_tups

	cols = ["speech_id", "short_name", "long_name", "value"]
	contrib_df = pd.DataFrame(tups, columns=cols)

	feature_type = [determine_feature_type(long_name)
					for long_name in contrib_df["long_name"]]
	contrib_df.insert(4, "feature_type", feature_type)

	contrib_df.to_sql("contribs", con_out, index=False, if_exists="replace")
	return contrib_df

# Removes first and last sentence of a radio address.
def strip_greetings(speech):
	speech = speech.replace("Ft.", "Ft").replace("U.S.", "US") # Avoids errors in sentence tokenizing.
	sentences = PunktSentenceTokenizer().tokenize(speech)
	exp = "|".join([sentences[0], sentences[-1]])
	stripped = re.sub(exp, "", speech)
	return stripped.strip()

# Selects the feature that had the largest contribution towards prediction. 
def choose_top_feature(speech_id, predicted, contrib_df):
	cond_one = contrib_df["speech_id"] == speech_id
	cond_two = contrib_df["feature_type"] != "intercept"
	current_values = contrib_df[cond_one & cond_two]				

	if predicted: 
		current_values = current_values.sort("value", ascending=False).reset_index()
	else: 
		current_values = current_values.sort("value").reset_index()

	return (current_values["short_name"][0], current_values["feature_type"][0])


# Adds html span tags for displaying highlighting.
# Years between 2000 and 2019 are replaced with **** after tags are added.
def add_spans(speech_info, contrib_df):
	stripped, speech_id, predicted = speech_info
	print "Adding spans to speech " + str(speech_id) + "..."
	
	feature = choose_top_feature(speech_id, predicted, contrib_df)

	if feature[1] == "pos":
		pos_pair = feature[0]
		tagged_words = [tagged for tagged in pos_tag(word_tokenize(stripped))
						if re.findall("[a-zA-Z]", tagged[1])]
		tagged_bigrams = zip(tagged_words[:-1], tagged_words[1:])
		word_tuples = [(bigram[0][0], bigram[1][0]) for bigram in tagged_bigrams
				       if " ".join((bigram[0][1], bigram[1][1]))==pos_pair]

		ranges = []
		for word_tup in word_tuples:
			exp = "".join(["(^|[^a-zA-Z])(?P<word_pair>",
			 			   word_tup[0],
			 			   "[^a-zA-Z\n]*",
			 			   word_tup[1],
			 			   ")([^a-zA-Z]|$)"])
			matches = re.finditer(exp, stripped)
			ranges += [m.span("word_pair") for m in matches]
		
		spans = []
		ranges.sort()
		for new_range in ranges:
			if spans and new_range[0] < spans[-1][1]:
				spans[-1] = (spans[-1][0], new_range[1])
			else: 
				spans.append(new_range)

	else: 
		word = feature[0]
		exp = "(^|[^a-zA-Z])(?P<w>" + word + ")([^a-zA-Z]|$)"
		matches = re.finditer(exp, stripped, flags = re.IGNORECASE)
		spans = [m.span("w") for m in matches]

	spans.sort(reverse=True)
	for span in spans:
		stripped = "".join([
			stripped[0:span[0]], 
		 	'<span class="highlight">', 
		 	stripped[span[0]:span[1]], 
		 	"</span>",
		 	stripped[span[1]:]
		])

	exp = "(?<=[^0-9])20[0-1][0-9](?=[^0-9])"
	return re.sub(exp, "****", stripped)

# Creates and stores a database that saves radio addresses split into paragraphs.
#  These are modified to be ready for use in flask app. 
def populate_paragraphs(clf, data, contrib_df, con_out):
	stripped_speeches = [strip_greetings(speech) for speech in data["speech"]]
	predicted = clf.predict(data)
	classy_speeches = [add_spans(triple, contrib_df) for triple in 
					   zip(stripped_speeches, data["id"], predicted)]

	par_lists = [speech.split("\n\n") for speech in classy_speeches]
	tups = [(par, speech_tup[1], i)
			for speech_tup in zip(par_lists, data["id"])
			for i, par in enumerate(speech_tup[0])]
	paragraphs = pd.DataFrame(tups, columns = ["par", "speech_id", "par_id"])

	speaker_info = pd.DataFrame({
		"speech_id": data["id"], 
	   	"predicted": ["Obama" if prediction else "Bush" 
	   				 for prediction in predicted],
	   	"observed": ["Obama" if observed else "Bush" 
	   				 for observed in data["speaker_num"]]
	})

	paragraphs = pd.merge(paragraphs, speaker_info, on="speech_id")
	paragraphs.to_sql("paragraphs", con_out, index=False, if_exists="replace")
	
if __name__ == '__main__':
	try: 
		input_database_name = sys.argv[1]
		output_database_name = sys.argv[2]
	except IndexError:
		print "usage: select_model.py input_database_name output_database_name"
		sys.exit("\nExecution failed: incorrect usage.")

	con_in = sqlite3.connect(input_database_name)
	con_out = sqlite3.connect(output_database_name)

	clf = load_clf("../model_files/final_model.pkl")
	data = get_data(con_in)
	populate_coef_data(clf, con_out)
	contrib_df = populate_contrib_data(clf, data, con_out)
	populate_paragraphs(clf, data, contrib_df, con_out)

