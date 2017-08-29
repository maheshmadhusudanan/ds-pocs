from flask import Flask, request, Blueprint
import json
from sentiment_generator_contextual import SentimentGeneratorContextual
import timeit
sentiment_service_api = Blueprint("sentiment_service_api", __name__)
st = SentimentGeneratorContextual()


@sentiment_service_api.route('/sentiment-cnn/update/<rec_id>', methods=['PUT'])
def updateSentiment(rec_id):
    print("received update request for record = "+str(request.get_json(force=True)))
    resp = st.updateRecord(rec_id, request.get_json(force=True))

    return json.dumps({"success": resp})


@sentiment_service_api.route('/sentiment-cnn', methods=['POST'])
def getSentiment():
    print("....recieved request ...")
    print("received form = "+request.form.get('reference_id', ""))
    model_type = request.form.get('model_type', "")

    if model_type.lower() == "lexicon":
        result_json = st.runSentimentNltk(request.form['text'],
                                      request.form.get('user', ""),
                                      request.form.get('process_terms', ""),
                                      request.form.get('reference_id', ""))
    else:
        result_json = st.runSentiment(request.form['text'],
                                  request.form.get('user', ""),
                                  request.form.get('process_terms', ""),
                                  request.form.get('reference_id', ""))
    # result_json = {
    #     'status': "SUCCESS",
    #     'score': str(s),
    #     'sentiment': t,
    #     'text': request.form['text'],
    #     'time_taken': str(round(elapsed, 0))+' ms'
    # }
    print(result_json)
    if 'updated_ts' in result_json:
       result_json['updated_ts'] = str(result_json['updated_ts'])
    return json.dumps(result_json)

@sentiment_service_api.route('/sentiment-cnn/all', methods=['GET'])
def getAllRecords():
    page = request.args.get('start') if request.args.get('start') is not None else '0'
    limit = request.args.get('limit') if request.args.get('limit') is not None else '100'
    print("accessing records page = "+page+" limit = "+limit)
    resp = st.getAllRecords(int(page), int(limit))

    return json.dumps(resp)
