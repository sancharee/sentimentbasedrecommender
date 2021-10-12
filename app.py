import flask
import model


app = flask.Flask(__name__)



# Set up the main route
@app.route('/')
def home():
    return flask.render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if (flask.request.method == 'POST'):
        name=flask.request.form['user_name']
        res = model.getImprovedRecommendations(name)
        items=[]
        for i in range(len(res)):
            items.append(res[i])
        return flask.render_template('results.html',product_name=items)
    
@app.route("/predict_api", methods=['POST', 'GET'])
def predict_api():
    print(" request.method :",request.method)
    if (request.method == 'POST'):
        data = request.get_json()
        return jsonify(recommendation.getImprovedRecommendations([np.array(list(data.values()))]).tolist())
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run()