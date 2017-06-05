from flask import Flask, request
import json
from sentiment_generator import SentimentGenerator
import timeit
app = Flask(__name__)
st = SentimentGenerator()

@app.route("/")
def default():
    return "<html><body><div>Welcome</div></body></html>"

@app.route('/sentiment-cnn/update/<rec_id>', methods=['PUT'])
def updateSentiment(rec_id):
    print("received update request for record = "+str(request.get_json(force=True)))
    resp = st.updateRecord(rec_id, request.get_json(force=True))

    return json.dumps({"success": resp})


@app.route('/sentiment-cnn', methods=['POST'])
def getSentiment():
    print("received form = "+request.form.get('reference_id', ""))
    result_json = st.runSentiment(request.form['text'],
                                  request.form.get('user', ""),
                                  request.form.get('reference_id', ""))
    # result_json = {
    #     'status': "SUCCESS",
    #     'score': str(s),
    #     'sentiment': t,
    #     'text': request.form['text'],
    #     'time_taken': str(round(elapsed, 0))+' ms'
    # }
    print(result_json)
    result_json['updated_ts'] = str(result_json['updated_ts'])
    return json.dumps(result_json)

@app.route('/sentiment-cnn/all', methods=['GET'])
def getAllRecords():
    page = request.args.get('start') if request.args.get('start') is not None else '0'
    limit = request.args.get('limit') if request.args.get('limit') is not None else '100'
    print("accessing records page = "+page+" limit = "+limit)
    resp = st.getAllRecords(int(page), int(limit))

    return json.dumps(resp)


if __name__ == "__main__":
    app.run()