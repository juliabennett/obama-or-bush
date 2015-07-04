import re
import json
import sqlite3
import numpy as np
from flask import g
from flask import Flask
from flask import render_template
from flask import request

app = Flask(__name__)

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect('data.db')
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def query_db(query, args=(), tups=True):
    cur = get_db().execute(query, args)
    fetched = cur.fetchall()
    cur.close()
    if tups: 
        return fetched
    else: 
        return [tup[0] for tup in fetched]

with app.app_context():
    speech_ids = query_db("SELECT DISTINCT speech_id FROM paragraphs", tups=False)

num_total = 0
num_correct = 0

@app.route('/')
def reset():
    speech_id = np.random.choice(speech_ids)

    command = "SELECT par FROM paragraphs WHERE speech_id=? ORDER BY par_id"
    pars = query_db(command, (speech_id,), tups=False)

    command = "SELECT predicted, observed FROM paragraphs WHERE speech_id=? LIMIT 1"
    pred, obs = query_db(command, (speech_id,))[0]

    global num_total 
    global num_correct 
    num_total += 1
    if pred == obs:
        num_correct += 1

    command = "SELECT COUNT(*) FROM contribs WHERE speech_id=?"
    num_contributors = query_db(command, (speech_id,), tups=False)[0]

    return render_template(
        'game.html', 
        speech_id=speech_id,
        pars=pars, 
        predicted=pred, 
        observed=obs, 
        num_correct = num_correct, 
        num_total = num_total,
        num_contributors = num_contributors
    )

@app.route('/details/')
def details():
    return render_template("details.html")

@app.route('/values/')
def get_values():
    table = request.args.get('table', '')
    speech_id = request.args.get('speech_id', 0)
    feature_type = request.args.get('type', 'all')
    search = request.args.get('search', '')
    order = request.args.get('order', 'descMag')
    lower = request.args.get('from', 1)
    upper = request.args.get('to', 50)
    
    type_dict = {
        'all': ['word', 'pos', 'intercept'],
        'word': ['word', '', ''],
        'pos': ['pos', '', '']
    }
    if table == "contribs":
        args = tuple([speech_id] + type_dict[feature_type])
        command = " ".join(["SELECT long_name, value FROM contribs", 
                            "WHERE speech_id=? AND feature_type IN (?, ?, ?)"])
        results = query_db(command, args)
    else:
        args = tuple(type_dict[feature_type])
        command = "SELECT long_name, value FROM coefs WHERE feature_type IN (?, ?, ?)"
        results = query_db(command, args)

    value_pairs = [value_pair for value_pair in results
                   if re.search(search, value_pair[0].split(":")[-1], flags = re.IGNORECASE)]

    orderDict = {
        "descMag": {"key": lambda pair: abs(pair[1]), 
                    "reverse":  True },
        "ascMag": {"key": lambda pair: abs(pair[1]), 
                    "reverse":  False },
        "descVal": {"key": lambda pair: pair[1], 
                    "reverse":  True},
        "ascVal": {"key": lambda pair: pair[1], 
                    "reverse":  False}
    }
    value_pairs.sort(**orderDict[order])
    final_value_pairs = value_pairs[int(lower)-1:int(upper)]

    for_json = {
        "feature_names":[value_pair[0] for value_pair in final_value_pairs],
        "values": [round(value_pair[1], 7) for value_pair in final_value_pairs]
    }
    
    try: 
        for_json["mag"] = max([abs(value) for value in for_json["values"]])
    except:
        for_json["mag"] = 1

    return json.dumps(for_json)

if __name__ == '__main__':
    app.run()

