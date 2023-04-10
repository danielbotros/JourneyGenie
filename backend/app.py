import json
import math
import os
import numpy as np
import re
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
MYSQL_USER = "root"
MYSQL_USER_PASSWORD = "perfectpup_4300!"
MYSQL_PORT = 3306
MYSQL_DATABASE = "dogdb"

mysql_engine = MySQLDatabaseHandler(
    MYSQL_USER, MYSQL_USER_PASSWORD, MYSQL_PORT, MYSQL_DATABASE)

# Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)

# # Sample search, the LIKE operator in this case is hard-coded,
# # but if you decide to use SQLAlchemy ORM framework,
# # there's a much better and cleaner way to do this
# def sql_search(episode):
#     query_sql = f"""SELECT * FROM episodes WHERE LOWER( title ) LIKE '%%{episode.lower()}%%' limit 10"""
#     keys = ["id","title","descr"]
#     data = mysql_engine.query_selector(query_sql)
#     return json.dumps([dict(zip(keys,i)) for i in data])


@app.route("/")
def home():
    return render_template('base.html', title="sample html")


@app.route("/perfectpupper")
def dog_search():
    print("request: ", request)
    # print(preprocess()[0])
    cleaned_data = preprocess()
    inv_indx = inv_idx(cleaned_data)
    #print("inv_inde:", inv_idx(cleaned_data))
    print("idf: ", compute_idf(inv_indx, len(
        cleaned_data), min_df=0, max_df_ratio=.95))
    hours = request.args.get("hours")
    space = request.args.get("space")
    trait1 = request.args.get("trait1")
    trait2 = request.args.get("trait2")
    trait3 = request.args.get("trait3")
    return time_commitment(hours, space, trait1, trait2, trait3)


def time_commitment(hours, space, trait1, trait2, trait3):
    # the scale of this needs to be changed because it doesn't work for inputs < .3
    print("hours: ",  hours)
    size = space_commitment(space)  # change this later, KILOGRAMS
    print("size: ", size)
    print("trait1: ",  trait1)
    print("trait2: ",  trait2)
    print("trait3: ",  trait3)
    query_sql = f"""SELECT breed_name, trainability_value, descript, temperament, max_weight
    FROM breeds
    WHERE trainability_value >= {hours}
    AND min_weight >= {size*10 - 20} AND min_weight <= {size*10}
    AND (temperament LIKE '%%{trait1}%%'
    OR temperament LIKE '%%{trait2}%%'
    OR temperament LIKE '%%{trait3}%%')
    limit 10"""
    data = mysql_engine.query_selector(query_sql)
    keys = ["breed_name", "trainability_value",
            "descript", "temperament", "max_weight"]
    # keys = ["breed_name", "descript", "temperament", "popularity", "min_height", "max_height",
    #         "min_weight",
    #         "max_weight",
    #         "min_expectancy",
    #         "max_expectancy",
    #         "dog_group",
    #         "grooming_frequency_value",
    #         "grooming_frequency_category",
    #         "shedding_value",
    #         "shedding_category",
    #         "energy_level_value",
    #         "energy_level_category",
    #         "trainability_value",
    #         "trainability_category",
    #         "demeanor_value",
    #         "demeanor_category"]
    # keys = ["breed_name", "trainability_value",
    #         "energy_level_value", "grooming_frequency_value"]
    print(data)
    return json.dumps([dict(zip(keys, i)) for i in data])
# WHERE 2*(trainability_value) <= '%%{hours}%% AND '%%{hours}%%' < 2*(trainability_value);"""


def space_commitment(size):
    size = size.lower()
    if size == "small":
        size = 2
    elif size == "medium":
        size = 4
    else:
        size = 6
    return size


def tokenize(text):
    if text != None:
        return [x for x in re.findall(r"[a-z]+", text.lower())]
    else:
        return []


def preprocess():
    query_sql = f"""SELECT descript, temperament FROM breeds"""
    data = mysql_engine.query_selector(query_sql)
    cleaned_data = []
    for descript, temperament in list(data):
        breed_data = []
        breed_data.append(tokenize(descript))
        breed_data.append(tokenize(temperament))
        breed_data = [item for sublist in breed_data for item in sublist]
        cleaned_data.append(breed_data)
    return cleaned_data

# build inverted index


def inv_idx(cleaned_data):
    inv_index = {}
    for i, data in enumerate(cleaned_data):
        tf = {}
        for tok in data:
            if tok in tf:
                tf[tok] += 1
            else:
                tf[tok] = 1
        for t, count in tf.items():
            if t not in inv_index:
                inv_index[t] = [(i, count)]
            else:
                inv_index[t].append((i, count))
    return inv_index

# compute idf
# TODO: MESS AROUND WITH VALUES


def compute_idf(inv_idx, n_docs, min_df=0, max_df_ratio=.95):
    idf_dict = {}
    for d, l in inv_idx.items():
        if len(l) >= min_df and len(l) <= max_df_ratio*n_docs:
            val = n_docs/(1 + len(l))
            idf = math.log(val, 2)
            idf_dict[d] = idf
    return idf_dict

# compute norms


def compute_doc_norms(index, idf, n_docs):
    norms = np.zeros(n_docs)
    for term, lis in index.items():
        if term in idf:
            idf_value = idf[term]
            for (d, tf) in lis:
                norms[d] += (tf * idf_value) ** 2
    norms = np.sqrt(norms)
    return norms


app.run(debug=True)
