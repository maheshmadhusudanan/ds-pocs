from flask import Flask,request
import json
from sentiment_generator import SentimentGenerator
import timeit
app = Flask(__name__)
st = SentimentGenerator()

@app.route("/")
def default():
    return "<html><body><div>Welcome</div></body></html>"

@app.route('/sentiment-cnn', methods=['POST'])
def getSentiment():
    start_time = timeit.default_timer()
    s, t = st.runSentiment(request.form['text'])
    elapsed = (timeit.default_timer() - start_time) * 1000
    result_json = {
        'status': "SUCCESS",
        'score': str(s),
        'sentiment': t,
        'text': request.form['text'],
        'time_taken': str(round(elapsed, 0))+' ms'
    }
    print(result_json)
    return json.dumps(result_json)

if __name__ == "__main__":
    app.run()