from flask import Flask, request
from sentiment_service import sentiment_service_api
from sentiment_file_service import sentiment_file_api
app = Flask(__name__)
app.register_blueprint(sentiment_service_api)
app.register_blueprint(sentiment_file_api)
app.config["APPLICATION_ROOT"] = "/sentiment-cnn"
@app.route("/")
def default():
    return "<html><body><div>Welcome To Sentiment services. Use APIs to access the features.</div></body></html>"

if __name__ == "__main__":
    app.run("0.0.0.0", port=5000)