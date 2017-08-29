'''
Created on Jun 2, 2017

@author: Mahesh.M
'''
import xml.etree.ElementTree as ET
import utils
import re
from utils import *
from keras.datasets import imdb
import numpy as np
import os
import requests
from pymongo import MongoClient
import json
from langdetect import detect
from random import randrange
import time
from numbers import Number
from collections import Counter
from operator import itemgetter
import csv
import timeit
import flask
from sentiments_db import SentimentsDB
from sentiment_generator import SentimentGenerator

class SentimentGeneratorContextual:
    #
    # dir paths
    #
    current_dir = os.getcwd()
    DATA_DIR = current_dir + "/data"
    LANGUAGE_WORDS_FILE = DATA_DIR + "/" + "{ln}_global_list.p"
    MODEL_PATH = current_dir + "/models/"
    #
    # Global variables
    #
    LN_WORDS_TO_IDX = {}
    LN_IDX_TO_WORDS = {}
    SUPPORTED_LANGUAGES = ['da']
    CATEGORY_KEYS = ["Negative", "Neutral", "Positive"]
    MODELS = {'en': 'model-s50000-v05-06-17.h5',
              'da': 'model-ln-da-s40000-v2017-07-26.h5'}
    VOCAB_SIZE = {'en': 50000,
                  'da': 40000}

    CNN_MODEL = {}
    stdb = SentimentsDB()
    enSt = SentimentGenerator()

    # maximum num of words the sentiment can process
    MAX_WORD_COUNT = 500

    #print(DATA_DIR, ORIGINAL_DATA, TRAIN_DATA)
    #vocabsize cannot be changed , the model would have to be regenerated
    #if it needs to be increased

    def __init__(self):
        self.initialize_model()

    def initialize_model(self):
        #
        # load the words list from the file and sort it in order of its usage
        #
        self.initialize_ln_words_from_file()
        #
        # Load all the model weights for different languages
        #
        for ln in self.SUPPORTED_LANGUAGES:
            #
            #  Build the keras model
            #
            vocabsize = self.VOCAB_SIZE[ln]
            cat_model = Sequential()
            cat_model.add(Embedding(input_dim=vocabsize + 2, output_dim=32, input_length=self.MAX_WORD_COUNT, dropout=0.2))
            cat_model.add(Dropout(0.3))
            cat_model.add(Convolution1D(64, 3, activation='relu', border_mode='same'))
            cat_model.add(Dropout(0.2))
            cat_model.add(MaxPooling1D())
            cat_model.add(Flatten())
            cat_model.add(Dense(100, activation='relu'))
            # cat_model.add(Dropout(0.1))
            cat_model.add(Dense(3, activation='softmax'))
            cat_model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=['accuracy'])
            # cat_model.summary()
            #
            # load the weights
            #
            print ("****loading model weights for ln = "+ln+" from path = "+self.MODEL_PATH + self.MODELS[ln])
            cat_model.load_weights(self.MODEL_PATH + self.MODELS[ln])
            self.CNN_MODEL[ln] = cat_model

        print("######### Completed loading all the models")

    #
    # The primary function to load the dictionary words of all supported
    # languages.
    # The words are mapped to an index id based on the vocabsize of the model.
    #
    def initialize_ln_words_from_file(self):

        for ln in self.SUPPORTED_LANGUAGES:
            ##
            # Build the word to idx and idx to word and dump it for later use
            #
            ln_data_file_path = self.LANGUAGE_WORDS_FILE.replace('{ln}',ln)
            print("***** about to read language words = "+ln+" from file = "+ln_data_file_path)
            ln_words_list = pickle.load(open(ln_data_file_path, "rb"), encoding='latin1')

            # print(GLOBAL_WORDS_LIST[:20])
            vocab_size = self.VOCAB_SIZE[ln]
            print("vocabsize for ln = "+ln+" = "+str(vocab_size)+" word list size = "+str(len(ln_words_list)))
            ln_words_list = ln_words_list[:vocab_size]

            print("***** total words in dictionary = " + str(len(ln_words_list)))
            word_to_idx_list = {w: idx for idx, w in enumerate(ln_words_list)}
            idx_to_word_list = {idx: w for idx, w in enumerate(ln_words_list)}
            assert len(word_to_idx_list) == len(idx_to_word_list)
            print("word to idx size = " + str(len(word_to_idx_list)))
            print("id to word size = " + str(len(idx_to_word_list)))

            self.LN_WORDS_TO_IDX[ln] = word_to_idx_list
            self.LN_IDX_TO_WORDS[ln] = idx_to_word_list

    #
    # For predicting the sentiment the target text needs to be converted to
    # word indexes and the process terms needs to be converted to the unique
    # index key
    #
    def convert_words_to_idx(self, text, process_terms,text_language):

        ln = text_language.lower()

        if ln not in self.SUPPORTED_LANGUAGES:
            raise ValueError('Language '+text_language+' is not supported yet')

        word_to_idx_list = self.LN_WORDS_TO_IDX[ln]
        #     global missing_most_common_words
        word_idx = []
        p_terms = [x.lower().replace('*', '') for x in process_terms.split(",")]
        text = text.lower().strip()
        text = text.replace(".", " . ")
        text = text.replace(",", " , ")
        vocab_size = self.VOCAB_SIZE[ln]
        #
        # Every process term is given an unique id, here the id
        # is = vocabsize + 1
        #
        for pt in p_terms:
            try:
                print(">>> processing pterm = "+pt)
                text = text.replace(pt, str(vocab_size + 1))
            except UnicodeDecodeError:
                tx = text.encode('utf-8')
                text = tx.replace(pt, str(vocab_size + 1))

        for word in text.split():
            word = word.strip()
            #         if word in p_terms:
            #             word_idx.append(vocabsize + 1)
            if word == str(vocab_size + 1):
                word_idx.append(vocab_size + 1)
            else:
                if (word not in word_to_idx_list):
                    if not self.has_numbers(word):
                        word_idx.append(vocab_size)
                        # if word not in missing_most_common_words:
                        # missing_most_common_words.append(word)
                        # print ("word not found in vocab = "+word)
                else:
                    word_idx.append(word_to_idx_list[word])

        return word_idx

    def has_numbers(self,txt):
        return bool(re.search(r'\d', txt))


    def getAllRecords(self, start, limit):

        return self.stdb.get_records(start, limit)

    def updateRecord(self, rec_id, record):

        return self.stdb.update_record(rec_id, record)

    def process_bulk_files(self, filename):

        errors = []
        try:
            print("about to open file = "+self.UPLOAD_DIR+filename)
            generated_sentiments = []
            with open(self.UPLOAD_DIR+filename, 'r') as f:
                csvfile = csv.reader(f)
                firstrow = next(csvfile, None)  # skip header
                # firstrow = csvfile[0]
                print("first row text = "+firstrow[2])
                if firstrow[0] != "GENERATED" or firstrow[1] != "ACTUAL" or firstrow[2] != "TEXT":
                    errors.append("Invalid excel template! Please use a valid template")

                if len(errors) > 0:
                    f.close()
                    os.remove(self.UPLOAD_DIR+filename)
                    return errors

                deviations = 0
                total_count = 0
                for row in csvfile:
                    if row[2] == 'TEXT':
                        continue
                    sentiment_resp = self.runSentiment(row[2], reference_id=filename)
                    # print(sentiment_resp['sentiment'])
                    row[0] = sentiment_resp['sentiment']
                    #
                    # keep a count of deviations if the sentiment does not matches.
                    #
                    deviations += 0 if (row[0] == row[1]) else 1
                    total_count += 1
                    generated_sentiments.append(row)
                    # csvfile.writerow(row)
                # finally close file
                f.close()

            #
            # calculate percent error
            #
            accuracy_percentage = ((total_count - deviations) * 100.0) / total_count
            print("accuracy percentage = "+str(accuracy_percentage))

            print("writing csv file")

            if sys.version_info[0] == 2:  # Not named on 2.6
                access = 'wb'
                kwargs = {}
            else:
                access = 'wt'
                kwargs = {'newline': ''}

            with open(self.UPLOAD_DIR+filename, access, **kwargs) as f:
                writer = csv.writer(f)
                #
                # write down the stats
                #
                stats_row = ["TOTAL = ", str(total_count)]
                writer.writerow(stats_row)
                stats_row = ["DEVIATIONS = ", str(deviations)]
                writer.writerow(stats_row)
                stats_row = [" ACCURACY = ", str(accuracy_percentage) + "%"]
                writer.writerow(stats_row)
                header_row = ["GENERATED", "ACTUAL", "TEXT"]
                writer.writerow(header_row)
                for row in generated_sentiments:
                    writer.writerow(row)
                f.close()

            new_filename = filename.replace("IN-PROGRESS", "COMPLETE")
            os.rename(self.UPLOAD_DIR+filename, self.UPLOAD_DIR+new_filename)
            print("renamed the file to "+new_filename)
            return None
        except Exception as e:
            print(str(e))
            new_filename = filename.replace("IN-PROGRESS", "ERROR")
            os.rename(self.UPLOAD_DIR + filename, self.UPLOAD_DIR + new_filename)
            errors.append("some error occurred, please check logs")
            return errors

    def get_padded_data(self,word_idx):
        return sequence.pad_sequences(word_idx, maxlen=self.MAX_WORD_COUNT, value=0)


    def runSentiment(self, text, user="", process_terms = "", reference_id=""):

        start_time = timeit.default_timer()

        error_msg = None
        status = "SUCCESS"
        detected_ln = 'en'
        sentiment = ''
        sentiment_score = 0.0
        try:
            detected_ln = detect(text)
        except Exception as ex:
            print(ex)
            status = "ERROR"
            error_msg = "Error occured while trying to detect the language! Please check the text"

        if detected_ln.lower() == "en":
            return self.enSt.runSentiment(text, user, process_terms, reference_id)

        preds = {}
        try:
            text_words_idxs = self.convert_words_to_idx(text, process_terms, detected_ln)
            text_idx_array_padded = self.get_padded_data([np.array(text_words_idxs)])
            prediction = self.CNN_MODEL[detected_ln].predict(text_idx_array_padded, batch_size=1, verbose=1)
            # print(prediction)
            preds = dict(zip(self.CATEGORY_KEYS, prediction[0]))
            # print(preds)
            ordered_preds_keys = sorted(preds,key=preds.get, reverse=True)
            sentiment = ordered_preds_keys[0]
            sentiment_score = preds[ordered_preds_keys[0]]
            #     num = float(((ordered_preds_keys[0])[:5]).strip())
            #     return num
            print ("\n\n------------------- model prediction (probabilities) -------------------")
            print("\n".join([x.ljust(20)+" \t\t "+str(preds[x]) for x in ordered_preds_keys]))
            # print("\n".join(ordered_preds_keys))
            print("\n\n====================R E S U L T=====================================")
            print("sentiment = "+ordered_preds_keys[0]+" = "+str(preds[ordered_preds_keys[0]]))

        except Exception as ex:
            print(ex)
            status = "ERROR"
            error_msg = str(ex)

        elapsed = (timeit.default_timer() - start_time) * 1000
        result_json = {
                        'status': status,
                        'ln': detected_ln,
                        'user': user,
                        'score': str(preds),
                        'sentiment': sentiment,
                        'text': text,
                        'time_taken': str(round(elapsed, 0)) + ' ms',
                        'untrained_words': '',
                        'process_terms': process_terms,
                        'sentiment_manual': '',
                        'reference_id': reference_id,
                        'model_ver': self.MODELS[detected_ln],
                        'error': error_msg
                     }

        if status == "SUCCESS":
            result_json['_id'] = self.stdb.save_record(result_json)
        result_json['text'] = ''

        return result_json

    def runSentimentNltk(self, text, user="", process_terms="", reference_id=""):

        start_time = timeit.default_timer()

        error_msg = None
        status = "SUCCESS"
        detected_ln = 'en'
        sentiment = ''
        sentiment_score = 0.0
        try:
            detected_ln = detect(text)
        except Exception as ex:
            print(ex)
            status = "ERROR"
            error_msg = "Error occured while trying to detect the language! Please check the text"

        if detected_ln.lower() != 'da':
            status = "ERROR"
            error_msg = "Language "+detected_ln.upper()+" not supported yet!"
        else:
            try:
                request_json = {
                    'body_text': text,
                    'processed_terms': process_terms
                }
                url = 'http://34.234.24.236:5010/api/sentiment'
                headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
                resp = requests.post(url, data=json.dumps(request_json), headers=headers)
                print("@@@@@@@@@@@"+resp.text)
                sentiment = resp.text
                sentiment_score = resp.text


            except Exception as ex:
                print(ex)
                status = "ERROR"
                error_msg = str(ex)

        elapsed = (timeit.default_timer() - start_time) * 1000
        result_json = {
            'status': status,
            'ln': detected_ln,
            'user': user,
            'score': str(sentiment_score),
            'sentiment': str(sentiment),
            'text': text,
            'time_taken': str(round(elapsed, 0)) + ' ms',
            'untrained_words': '',
            'process_terms': process_terms,
            'sentiment_manual': '',
            'reference_id': reference_id,
            'model_ver': '',
            'error': error_msg
        }

        if status == "SUCCESS":
            result_json['_id'] = self.stdb.save_record(result_json)
        result_json['text'] = ''

        return result_json


    # def runSentimentOld(self, text, user="", process_terms = "", reference_id=""):
    #     start_time = timeit.default_timer()
    #     # text_clean = re.sub('\W+', ' ', text)
    #     text_clean = text
    #     #
    #     # split the sentence into words and prepare an index
    #     #
    #     # text = "dont know what could have saved limp dispiriting yam but it definitely wasnt a lukewarm mushroom as murky and appealing as bong water"
    #     # text = "Oh and to top it all off the staff was taking photo in front of the food so we couldn't get in and the manager was five feet away not even paying attention "
    #     # text = "When I made the reservation they made sure to ask when and where our show was to ensure we had enough time to dine and still make it to our show"
    #     # text = "Very nice little place. Prices are reasonable and food was good. Staff is courteous and patient with a non Spanish speaking guy...lol."
    #     #text = "food was very good , but the service was amazing"
    #     #text = "The parents of two of the seven cousins killed in that crash have sued the truck's driver and the trucking company"
    #     textWordsArray = np.array(text_clean.lower().split())
    #     #textWordsArray
    #     textWordsIdx = []
    #
    #     status = "SUCCESS"
    #     ignored_words = ""
    #     for index, w in enumerate(textWordsArray):
    #         if index > 499:
    #             status = "EXCEEDED_WORD_LIMIT"
    #             break;
    #
    #         word = ''.join(c for c in w if (c.isalnum() or c == "'"))
    #         if word not in self.widx:
    #             textWordsIdx.append(self.vocabsize - 1)
    #             print(" not found word = "+word)
    #             ignored_words += word + ", "
    #         elif self.widx[word] > self.vocabsize - 1:
    #             textWordsIdx.append(self.vocabsize - 1)
    #             print("rare word = " + word)
    #         else:
    #             textWordsIdx.append(self.widx[word])
    #
    #     textWordsIdxArray = [np.array(textWordsIdx)]
    #     #print textWordsIdx
    #
    #     #print textWordsIdxArray
    #     textIdxArrayPadded = sequence.pad_sequences(textWordsIdxArray, maxlen=self.STR_MAX_LEN, value=0)
    #     #print textIdxArrayPadded
    #     prediction = self.model.predict(textIdxArrayPadded, batch_size=1,verbose=1)
    #     sentiment_score = prediction[0][0]
    #     if sentiment_score > 0.60:
    #         sentiment = "POSITIVE"
    #     elif sentiment_score < 0.50:
    #         sentiment = "NEGATIVE"
    #     else:
    #         sentiment = "NEUTRAL"
    #
    #     elapsed = (timeit.default_timer() - start_time) * 1000
    #     result_json = {
    #         'status': status,
    #         'user': user,
    #         'score': str(sentiment_score),
    #         'sentiment': sentiment,
    #         'text': text,
    #         'time_taken': str(round(elapsed, 0)) + ' ms',
    #         'untrained_words': ignored_words,
    #         'sentiment_manual': '',
    #         'reference_id': reference_id,
    #         'model_ver': self.MODEL_VERSION
    #     }
    #
    #     result_json['_id'] = self.stdb.save_record(result_json)
    #
    #     return result_json

