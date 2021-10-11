import flask
import pandas as pd
import pickle

#load the dataset
df=pd.read_csv("sample30.csv")

# Load the models from the file
recom_df = pd.read_pickle('models/item-item_recommender.pkl')
vectorizer = pickle.load(open("./models/feature_model.pkl", "rb"))
with open('models/xgboost_model.pkl' , 'rb') as pickle_file:
    classif_model = pickle.load(pickle_file)

#get recommendations for top 20 products per user
def getRecommendedItems(user):
    items = recom_df.loc[user].sort_values(ascending=False)[0:20]
    recommendations = pd.DataFrame({'product':items.index, 'score':items.values})
    return recommendations

#Function to get improved item based recommendations by using sentiment scores
def getImprovedRecommendations(user):
    ranking={}
    items = recom_df.loc[user].sort_values(ascending=False)[0:20]
    for idx in items.index:
        itm=df.loc[df['name'] == str(idx),:]
        features=vectorizer.transform(itm["reviews_text"])
        sentiment=classif_model.predict(features).tolist()
        score=(len([ele for ele in sentiment if ele > 0]) / len(sentiment)) * 100
        ranking[idx]=score    
    ranking_sorted = sorted(ranking, key=ranking.get, reverse=True)
    return ranking_sorted[0:5]

app = flask.Flask(__name__)



# Set up the main route
@app.route('/')
def home():
    return flask.render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if (flask.request.method == 'POST'):
        name=flask.request.form['user_name']
        res = getImprovedRecommendations(name)
        items=[]
        for i in range(len(res)):
            items.append(res[i])
        return flask.render_template('results.html',product_name=items)
    


if __name__ == '__main__':
    app.run()