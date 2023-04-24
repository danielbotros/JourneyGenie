import matplotlib.pyplot as plt
import matplotlib
import json
import math
import os
import numpy as np
import re
from nltk.tokenize import TreebankWordTokenizer
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize


os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

MYSQL_USER = "root"
MYSQL_USER_PASSWORD = "perfectpup_4300!"
MYSQL_PORT = 3306
MYSQL_DATABASE = "dogdb"
INDEX_TO_BREED = {}
treebank_tokenizer = TreebankWordTokenizer()
QUERY_VECTOR = []
DOCS_COMPRESSED_NORMED = []
V_TRANS = []


mysql_engine = MySQLDatabaseHandler(
    MYSQL_USER, MYSQL_USER_PASSWORD, MYSQL_PORT, MYSQL_DATABASE)

mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)


def index_to_breed():
    query_sql = f"""SELECT breed_name FROM breeds"""
    breeds = mysql_engine.query_selector(query_sql)
    for i, breed in enumerate(breeds):
        INDEX_TO_BREED[i] = breed


def svd(query):
    text = get_data()
    index_to_breed()
    print("text: ", text)
    # TODO: make vectorizer global variable
    vectorizer = TfidfVectorizer(
        stop_words='english', smooth_idf=True)  # TODO: add df?
    input_matrix = vectorizer.fit_transform(text)
    print(input_matrix.shape)
    query_vector = vectorizer.transform(
        [query]).toarray().T
    print("query vector: ", query_vector)
    QUERY_VECTOR = query_vector

    u, s, v_trans = svds(input_matrix, k=40)
    print("u: ", u)
    print("s: ", s)
    print("v_trans: ", v_trans.shape)

    docs_compressed = u
    # query_vector = docs_compressed.dot(query_vector)

    docs_compressed_normed = normalize(docs_compressed)

    V_TRANS = v_trans
    DOCS_COMPRESSED_NORMED = docs_compressed_normed
    index_search_results = []

    for i, score in cossim_with_svd(query_vector, docs_compressed_normed, v_trans, k=30): 
        print("breed: ", INDEX_TO_BREED[i], " score: ", score)
        index_search_results.append(INDEX_TO_BREED[i])
    print()
    return index_search_results


@ app.route("/")
def home():
    return render_template('base.html', title="sample html")


@ app.route("/perfectpupper")
def dog_search():
    # print("data ", get_data())
    # get_topics(components)
    index_to_breed()

    print("request: ", request)
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

    print("Using SVD:")
    index_search_results = svd(query)

    direct_search_results = direct_search(time, space)

    combined_breeds = merge_results(
        direct_search_results, index_search_results)

    results = ()
    for breed_name in combined_breeds:
        results = results + tuple(breed_name)
    print("results: ", results)

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

    return list(matches)[: 10]


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


def get_data():
    # TODO: does this get new descriptions and temperamnents?
    query_sql = f"""SELECT descript, temperament FROM breeds"""
    data = mysql_engine.query_selector(query_sql)
    descript_temp_list = []
    # print(list(data))
    for descript, temperament in list(data):
        breed_data = ""
        if descript != None:
            descript = descript.lower()
            breed_data += descript

        if temperament != None:
            temperament = temperament.lower()
            breed_data += temperament
        descript_temp_list.append(breed_data)
        print("temperament: ", temperament)

    return descript_temp_list


# list of all descriptions, all temperaments
# TODO: SQL query to get all relevant dat?


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


def cossim_with_svd(query_vector, docs, v_trans, k=5):
    # denom = np.linalg.norm(query) * np.linalg.norm(docs)
    print("query vector: ", query_vector.shape)
    print("docs: ", docs.shape)
    query_vector = v_trans.dot(query_vector)

    sims = docs.dot(query_vector)
    sims_with_index = []
    for i, product in enumerate(sims):
        sims_with_index.append((i, -1*(product[0])))
    print("sims: ", sims)
    asort = sorted(sims_with_index, key=lambda t: t[1])
    print("asort: ", asort)
    # results = [(INDEX_TO_BREED[i], sims[i]) for i in asort[1:]]

    results = asort[:k+1]
    print("results: ", results)
    return results


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


app.run(debug=True)
