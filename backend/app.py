import json
import math
import os
import numpy as np
import re
from nltk.tokenize import TreebankWordTokenizer
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler

os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

MYSQL_USER = "root"
MYSQL_USER_PASSWORD = ""
MYSQL_PORT = 3306
MYSQL_DATABASE = "dogdb"
INDEX_TO_BREED = {}
treebank_tokenizer = TreebankWordTokenizer()


mysql_engine = MySQLDatabaseHandler(
    MYSQL_USER, MYSQL_USER_PASSWORD, MYSQL_PORT, MYSQL_DATABASE)

mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return render_template('base.html', title="sample html")


@app.route("/perfectpupper")
def dog_search():
    print("request: ", request)
    index_to_breed()
    cleaned_data = preprocess()
    inv_indx = inv_idx(cleaned_data)
    n_docs = len(cleaned_data)
    idf = compute_idf(inv_indx, n_docs, min_df=0, max_df_ratio=.95)
    doc_norms = compute_doc_norms(inv_indx, idf, n_docs)

    # TODO: how to weigh traits more/less

    time = request.args.get("time")
    space = request.args.get("space")
    trait1 = request.args.get("trait1")
    trait2 = request.args.get("trait2")
    trait3 = request.args.get("trait3")
    # TODO: make sure same trait not inputted twice
    query = trait1 + " " + trait2 + " " + trait3
    print("qUERY: ", query)
    direct_search_results = direct_search(time, space)
    index_search_results = format_breeds(index_search(query, inv_indx, idf, doc_norms,
                                                      score_func=accumulate_dot_scores, tokenizer=treebank_tokenizer))
    # print("index search results: ", index_search_results)
    # print("sql search results: ", direct_search_results)
    combined_breeds = merge_results(
        direct_search_results, index_search_results)
    # results = format_output(combined_breeds)
    # print("breeds: ", combined_breeds)
    # print("results: ", results)
    results = ()
    for breed_name in combined_breeds:
        results = results + tuple(breed_name)
    print(results)

    query_sql = f"""SELECT breed_name, descript, temperament, 
    energy_level_value, trainability_value, grooming_frequency_value,
    max_weight, max_height FROM breeds WHERE breed_name IN {results}"""
    data = mysql_engine.query_selector(query_sql)
    keys = ["breed_name", "descript", "temperament", "energy_level_value", "trainability_value",
            "grooming_frequency_value", "max_weight", "max_height"]
    return json.dumps([dict(zip(keys, i)) for i in data])


def merge_results(direct_results, index_results):
    matches = []

    dir = set(direct_results)
    ind = set(index_results)
    matches = dir.intersection(ind)  # TODO: why no intersection

    if len(matches) < 10:
        for res in index_results:
            matches.add(res)

    return list(matches)[:10]


def direct_search(time, space):
    time_values = compute_time(time)
    space_values = compute_space(space)
    query_sql = f"""SELECT breed_name
    FROM breeds
    WHERE max_height <= {space_values[0]} 
    AND max_weight <= {space_values[1]}

    AND energy_level_value <= {time_values[0]} 
    AND grooming_frequency_value <= {time_values[1]}
    AND trainability_value >= {time_values[2]} 
    
    """
    data = mysql_engine.query_selector(query_sql)
    return list(data)


def compute_space(space):
    space = space.lower()
    if space == "small":
        return [40, 11]
    elif space == "medium":
        return [68, 27]
    else:
        return [999, 999]


def compute_time(time):
    time = time.lower()
    if time == "1":
        return [0.6, 0.6, 0.6]
    elif time == "2":
        return [0.8, 0.8, 0.4]
    else:
        return [999, 999, 0]


def index_to_breed():
    query_sql = f"""SELECT breed_name FROM breeds"""
    breeds = mysql_engine.query_selector(query_sql)
    for i, breed in enumerate(breeds):
        INDEX_TO_BREED[i] = breed


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

# implement term at a time score accumulation


def accumulate_dot_scores(query_word_counts, index, idf):
    doc_scores = {}
    for term, lis in index.items():
        if term in query_word_counts and term in idf:
            for (d, tf) in lis:
                if d in doc_scores:
                    doc_scores[d] += (tf * (idf[term]**2) *
                                      query_word_counts[term])
                else:
                    doc_scores[d] = (tf * (idf[term]**2) *
                                     query_word_counts[term])

    return doc_scores


def index_search(query, index, idf, doc_norms, score_func=accumulate_dot_scores, tokenizer=treebank_tokenizer):
    results = []
    query = query.lower()
    tokens = tokenizer.tokenize(query)
    query_word_counts = {}
    for t in tokens:
        if t in query_word_counts:
            query_word_counts[t] += 1
        else:
            query_word_counts[t] = 1
    q = 0
    for term in query_word_counts:
        if term in idf:
            q += (idf[term] * query_word_counts[term]) ** 2

    scores = score_func(query_word_counts, index, idf)
    # print(scores)
    for i, score in scores.items():
        curr = score/((q ** (1/2)) * doc_norms[i])
        results.append((curr, i))

    results = sorted(results, key=lambda x: x[0], reverse=True)
    return results

# just for testing purposes


def format_breeds(raw_results):
    results = []
    for score, id in raw_results[:50]:
        results.append(INDEX_TO_BREED[id])
    return results


# def format_output(raw_results):
#     for score, id in raw_results[:10]:
#         print("breed: ", INDEX_TO_BREED[id], " score: ", score)


# app.run(debug=True)
