import json
import math
import os
import numpy as np
import re
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize


os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

MYSQL_USER = "root"
MYSQL_USER_PASSWORD = ""
MYSQL_PORT = 3306
MYSQL_DATABASE = "dogdb"
INDEX_TO_BREED = {}
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
    # print("text: ", text)
    # TODO: make vectorizer global variable to speed up app
    vectorizer = TfidfVectorizer(
        stop_words='english', smooth_idf=True)  # TODO: add df?
    input_matrix = vectorizer.fit_transform(text)
    query_vector = vectorizer.transform(
        [query]).toarray().T

    u, s, v_trans = svds(input_matrix, k=40)

    docs_compressed = u
    docs_compressed_normed = normalize(docs_compressed)
    index_search_results = []

    for i, score in cossim_with_svd(query_vector, docs_compressed_normed, v_trans, k=30):
        # print("breed: ", INDEX_TO_BREED[i], " score: ", score)
        index_search_results.append((INDEX_TO_BREED[i], score))
    # print()
    return index_search_results


@ app.route("/")
def home():
    return render_template('base.html', title="sample html")


@ app.route("/perfectpupper")
def dog_search():
    index_to_breed()
    print("request: ", request)
    # TODO: how to weigh traits more/less

    time = request.args.get("time")
    space = request.args.get("space")
    traits = request.args.get("traits")

    traits = traits.split(",")

    print("traits: ",traits)

    # TODO: make sure same trait not inputted twice

    query = ""
    empty_query = True
    for i in range(0,len(traits)):
        if(i != len(traits) - 1):
            query += traits[i] + " "
        else:
            query += traits[i]

    print("query",query)

    # query = trait1 + " " + trait2 + " " + trait3
    if(len(traits) != 0):
        empty_query = False

    # query = trait1 + " " + trait2 + " " + trait3
    # empty_query = trait1 == "" and trait2 == "" and trait3 == ""

    # print("Using SVD:")
    index_search_results = svd(query)  # [(breed, score), ...]
    direct_search_results = direct_search(time, space)  # [breed1, breed2, ...]
    # merged direct + index w/h scores [((breed,), score), (...)]
    index_search_breeds = []

    combined_breeds = []
    print("trait: ", query, " time: ", time, " space: ", space)
    if (not empty_query):
        index_search_breeds = merge_results(
            direct_search_results, index_search_results)

        combined_breeds = [x[0] for x in index_search_breeds]

    else:
        print("no trait input query: ")
        for res in direct_search_results:
            combined_breeds.append(res)  # no score
        combined_breeds = [x[0] for x in combined_breeds]

    # print("index_search_breeds", index_search_breeds)
    results = tuple(combined_breeds)
    # print("combined breeds: ", combined_breeds)

    # print("results: ", results)

    all_data = []
    for i in range(len(results)):
        query_sql = f"""SELECT breed_name, img, trainability_value, max_weight, max_height, descript1, temperament2,
    energy_level_value, grooming_frequency_value, hypoallergenic FROM breeds WHERE breed_name = '{str(results[i])}'"""
        all_data.append(list(mysql_engine.query_selector(query_sql))[0])

    keys = ["breed_name", "img", "trainability_value", "max_weight", "max_height", "descript1", "temperament2", "energy_level_value",
            "grooming_frequency_value",  "hypoallergenic"]

    res = json.dumps([dict(zip(keys, i)) for i in all_data])
    res = json.loads(res)
    # print("result: ", res)
    # print(type(res))
    direct_search_results = [x[0] for x in direct_search_results]
    #print("direct results: ", direct_search_results)
    max_score = 0.2
    if (not empty_query):
        dict_res = dict(index_search_breeds)
        for (breed, breed_score) in index_search_breeds:

            score = (abs(breed_score)/max_score)*50

            if breed in direct_search_results:
                #print("breed in both: ", breed, " score: ", score)

                score += 50

            dict_res[breed] = min(score, 100)

        # print("dict res: ", dict_res)
        for i, breed_result in enumerate(res):
            # print("res: ", breed_result)
            res[i]["score"] = round(dict_res[breed_result["breed_name"]], 0)
            print("breed: ", breed_result['breed_name'],
                  " score: ", dict_res[breed_result["breed_name"]])
    else:
        for i, breed_result in enumerate(res):
            res[i]["score"] = 100
            print("breed: ", breed_result['breed_name'],
                  " score: ", res[i]["score"])
    # print("final results: ", res)
    return res


def merge_results(direct_results, index_results):
    matches = []
    index_search_breeds = [x[0][0] for x in index_results]
    dict_res = {}
    for (breed, score) in index_results:
        dict_res[breed[0]] = score

    # print("dictionary: ", dict_res)

    dir = [x[0] for x in direct_results]
    # print("direct results: ", dir)
    ind = index_search_breeds
    # print("index search: ", ind)
    matches = [x for x in ind if x in dir]

    # print("matches ", (matches))

    for res in index_search_breeds:
        if (res not in dir):
            matches.append(res)
    result = []
    for breed in matches:
        # print("breed: ", breed)
        result.append((breed, dict_res[breed]))

    # print("matches: ", result[:20])

    return result[: 20]  # 20?


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
    data = list(data)
    # print("direct search results: ", data)
    return data


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


# def preprocess():

#     query_sql = f"""SELECT descript1, temperament1, descript2, temperament2 FROM breeds"""
#     data = mysql_engine.query_selector(query_sql)
#     cleaned_data = []
#     for descript1, temperament1, descript2, temperament2 in list(data):
#         breed_data = []
#         breed_data.append(tokenize(descript1))
#         breed_data.append(tokenize(temperament1))
#         breed_data.append(tokenize(descript2))
#         breed_data.append(tokenize(temperament2))
#         breed_data = [item for sublist in breed_data for item in sublist]
#         cleaned_data.append(breed_data)
#     return cleaned_data


def get_data():
    query_sql = f"""SELECT descript1, temperament1, descript2, temperament2 FROM breeds"""
    data = mysql_engine.query_selector(query_sql)
    descript_temp_list = []
    for descript1, temperament1, descript2, temperament2 in list(data):
        breed_data = ""
        if descript1 != None:
            descript1 = descript1.lower()
            breed_data += descript1

        if descript2 != None:
            descript2 = descript2.lower()
            breed_data += descript2

        if temperament1 != None:
            temperament1 = temperament1.lower()
            breed_data += temperament1

        if temperament2 != None:
            temperament2 = temperament2.lower()
            breed_data += temperament2

        descript_temp_list.append(breed_data)

    return descript_temp_list


# cosine similarity between vectorized query + U matrix
def cossim_with_svd(query_vector, docs, v_trans, k=5):
    query_vector = v_trans.dot(query_vector)
    sims = docs.dot(query_vector)
    sims_with_index = []

    for i, product in enumerate(sims):
        sims_with_index.append((i, -1*(product[0])))

    asort = sorted(sims_with_index, key=lambda t: t[1])
    results = asort[:k+1]
    # print("vossim results: ", results)
    return results


def format_breeds(raw_results):
    results = []
    for score, id in raw_results[:50]:
        results.append(INDEX_TO_BREED[id])
    return results


# app.run(debug=True)
