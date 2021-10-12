import pandas as pd
import pickle

#load the dataset
df=pd.read_csv("sample30.csv")

# Load the models from the file
recom_df = pd.read_pickle('models/item-item_recommender.pkl')
vectorizer = pickle.load(open("./models/feature_model.pkl", "rb"))
with open('models/randomforest_model.pkl' , 'rb') as pickle_file:
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