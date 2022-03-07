from math import*
from decimal import Decimal
from flask import Flask, render_template, request
import pickle
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import difflib

app = Flask(__name__)

rfc = pickle.load(open('rfc.pkl', 'rb'))

dataset1 = pd.read_csv('Nutrients.csv')

dataset = pd.read_csv('input.csv')
dataset = dataset.drop(['VegNovVeg'], axis=1)
dataset = dataset.dropna()
dataset = dataset.reset_index(drop=True)
fooditemlist = dataset['Food_items']
nutrientData = dataset.iloc[:, 4:].values
cosine_sim = linear_kernel(nutrientData, nutrientData)
smd = dataset.iloc[:, [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
titles = smd['Food_items']
indices = pd.Series(smd.index, index=smd['Food_items'])


def getfooditems(initialItem, smd):
    predictItems = []
    epoch = 0
    remainbf = initialItem

    while(True):
        if(epoch == 3):
            break
        if(checkRequirement(remainbf, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 3, initialItem)):
            break
        bfpred = rfc.predict([remainbf])
        predictItems.append(bfpred)
        yo = smd['Food_items'].values == bfpred
        n = smd[yo]
        n = n.values.tolist()
        n = n[0][1:]
        remainbf = Diff(remainbf, n)
        epoch = epoch+1
    return(predictItems)


def nth_root(value, n_root):
    root_value = 1/float(n_root)
    return round(Decimal(value) ** Decimal(root_value), 3)


def checkRequirement(x, y, p_value, init):
    remain = nth_root(sum(pow(abs(a-b), p_value)
                      for a, b in zip(x, y)), p_value)
    initial = nth_root(sum(pow(abs(a-b), p_value)
                       for a, b in zip(init, y)), p_value)
    if(remain <= float(initial)*0.2):
        return True
    else:
        return False


def Diff(list1, list2):
    sublist = []
    for i in range(len(list1)):
        sublist.append(list1[i]-list2[i])
    return sublist


def get_recommendations(title, indices, cosine_sim, titles):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    food_items = [i[0] for i in sim_scores]
    return titles.iloc[food_items]


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/conatct')
def contact():
    return render_template('Contact.html')


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/mainform', methods=['POST'])
def mainform():
    return render_template('main.html')


@app.route('/nutrients', methods=['POST'])
def nutrients():
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    height = int(request.form['height'])
    weight = int(request.form['weight'])
    if gender == 0:
        bmr = 10 * weight + 6.25 * height - 5 * age+5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    cal = bmr*float(request.form['active'])
    fat = (cal*0.3)/9
    carbs = (cal*0.45)/4
    arr = dataset1.loc[(dataset1['Age'] == age)
                       & (dataset1['Gender'] == gender)].values
    required = [cal, fat, arr[0][2], arr[0][3], arr[0][4],
                arr[0][5], arr[0][6],	carbs, arr[0][7],	arr[0][8],	arr[0][9]]
    breakfastCalRequirment = [x * 0.2 for x in required]
    lunchCalRequirment = [x * 0.30 for x in required]
    dinnerCalRequirment = [x * 0.50 for x in required]

    breakfastrecom = getfooditems(breakfastCalRequirment, smd)
    lunchrecom = getfooditems(lunchCalRequirment, smd)
    dinnerrecom = getfooditems(dinnerCalRequirment, smd)
    breakfastalt = []
    lunchalt = []
    dinneralt = []
    for i in breakfastrecom:
        breakfastalt.append((get_recommendations(
            i[0], indices, cosine_sim, titles)[0:2]).tolist())
    for i in lunchrecom:

        lunchalt.append((get_recommendations(
            i[0], indices, cosine_sim, titles)[0:2]).tolist())

    for i in dinnerrecom:
        dinneralt.append((get_recommendations(
            i[0], indices, cosine_sim, titles)[0:2]).tolist())

    return render_template('nutrients.html', breakfastrecom=breakfastrecom, lunchrecom=lunchrecom, dinnerrecom=dinnerrecom, dinneralt=dinneralt, lunchalt=lunchalt, breakfastalt=breakfastalt, lenbreakfastrecom=len(breakfastrecom), lendinnerrecom=len(dinnerrecom), lenlunchrecom=len(lunchrecom))


if __name__ == "__main__":
    app.run(debug=true)
