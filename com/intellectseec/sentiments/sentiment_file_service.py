from flask import Flask, request, redirect, url_for, Blueprint, send_from_directory
from time import gmtime, strftime
import os
from werkzeug.utils import secure_filename
import json
from sentiment_generator import SentimentGenerator
from os import listdir
from os.path import isfile, join
import timeit
sentiment_file_api = Blueprint("sentiment_file_api", __name__)
st = SentimentGenerator()
current_dir = os.getcwd()
UPLOAD_DIR = current_dir + "/files/"
FILE_DOWN_LOAD_SERVICE = "/sentiment-cnn/file/download/"

@sentiment_file_api.route('/sentiment-cnn/file/upload', methods=['POST'])
def upload_file():
    print("uploading excel file ")
    file = request.files['test-file']
    print("uploading excel file "+file.filename)
    user = request.form.get('user', "")
    if file.filename == '':
        return "<p>No File selected</p>"
    if file:
        filename = secure_filename(file.filename)
        file_namepart, file_extension = os.path.splitext(file.filename)
        date_time = strftime("%Y_%m_%d___%H-%M", gmtime())
        new_file_name = file_namepart+"-"+date_time+"-IN-PROGRESS"+file_extension
        os.chdir(UPLOAD_DIR)
        # arcpy.env.workspace = UPLOAD_DIR
        file.save(os.path.join(UPLOAD_DIR) + new_file_name)
        oper_resp_errors = st.process_bulk_files(new_file_name)
        result = {
            "success": True,
            "errors": oper_resp_errors
        }

    return json.dumps(result)

@sentiment_file_api.route('/sentiment-cnn/file/all', methods=['GET'])
def get_uploaded_files():
    print("reading files from dir = "+UPLOAD_DIR)

    os.chdir(UPLOAD_DIR)
    files = filter(os.path.isfile, os.listdir(UPLOAD_DIR))
    files = [f for f in files]  # add path to each file
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    count = len(os.listdir(UPLOAD_DIR))
    #files = [f for f in os.listdir(UPLOAD_DIR) if isfile(join(UPLOAD_DIR, f))]
    resp = {
        'totalCount': count,
        'results':[]
    }

    for f in files:
        fimeta = {
            'filename':f,
            "file_url": FILE_DOWN_LOAD_SERVICE + f,
        }
        resp['results'].append(fimeta)

    return json.dumps(resp)

@sentiment_file_api.route('/sentiment-cnn/file/download/<filename>', methods=['GET'])
def download(filename):
    print("reading files from dir = "+UPLOAD_DIR+filename)
    return send_from_directory(directory=UPLOAD_DIR, filename=filename)