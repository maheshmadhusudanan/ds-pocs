'''
Created on Jun 2, 2017

@author: Mahesh.M
'''
import xml.etree.ElementTree as ET
import utils
import csv
from utils import *
import re
import numpy as np
import os
import timeit
from keras.datasets import imdb
from sentiments_db import SentimentsDB

class SentimentGenerator:
    current_dir = os.getcwd()
    UPLOAD_DIR = current_dir + "/files/"
    DATA_DIR = current_dir + "/data"
    ORIGINAL_DATA = DATA_DIR + "/original"
    TRAIN_DATA = DATA_DIR + "/train"
    NEG_REVIEWS_TRAIN_FILE = TRAIN_DATA + "/negetivereviews.txt"
    POS_REVIEWS_TRAIN_FILE = TRAIN_DATA + "/positivereviews.txt"
    MODEL_PATH = current_dir + "/models/"
    WORDS_TO_IDX_DATA_FILE = DATA_DIR + "/words_to_idx.p"
    IDX_TO_WORD_DATA_FILE = DATA_DIR + "/idx_to_words.p"
    MODEL_VERSION = "v05-06-17"
    STR_MAX_LEN = 500

    MODEL_NAME = "model-s" + str(vocabsize) + "-"+MODEL_VERSION + ".h5"
    model = Sequential()
    widx = None
    stdb = SentimentsDB()
    #print(DATA_DIR, ORIGINAL_DATA, TRAIN_DATA)
    #vocabsize cannot be changed , the model would have to be regenerated
    #if it needs to be increased


    def __init__(self):
        self.initializeModel()
    #
    # this method should be used to generate the words_to_idx.p
    # pickle dump file
    #
    def initializeWordsIndxIfNotExits(self):
        imdbwords = imdb.get_word_index();
        pickle.dump(imdbwords, open(self.WORDS_TO_IDX_DATA_FILE, "wb"))

    def initializeModel(self):
        #
        # load the words list from the file and sort it in order of its usage
        #
        #self.initializeWordsIndxIfNotExits()
        print("about to read file ="+self.WORDS_TO_IDX_DATA_FILE)
        self.widx = pickle.load(open(self.WORDS_TO_IDX_DATA_FILE,"rb"))
        print(len(self.widx))
        #idx2word = {v:k for k,v in widx.iteritems()}
        idx2word = {v:k for k,v in self.widx.items()}
        widx_sorted_list = sorted(self.widx, key=self.widx.get)
        #
        #  Build the keras model
        #
        self.model = Sequential()
        self.model.add(Embedding(input_dim=self.vocabsize, output_dim=32, input_length=self.STR_MAX_LEN, dropout=0.2))
        self.model.add(Dropout(0.2))
        self.model.add(Convolution1D(64, 5, activation='relu', border_mode='same'))
        self.model.add(Dropout(0.5))
        self.model.add(MaxPooling1D())
        self.model.add(Flatten())
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dropout(0.9))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss="binary_crossentropy", optimizer=Adam(),  metrics=['accuracy'])
        self.model.summary()
        #
        # load the weights
        #
        self.model.load_weights(self.MODEL_PATH + self.MODEL_NAME)

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




    def runSentiment(self, text, user="", reference_id=""):
        start_time = timeit.default_timer()
        # text_clean = re.sub('\W+', ' ', text)
        text_clean = text
        #
        # split the sentence into words and prepare an index
        #
        # text = "dont know what could have saved limp dispiriting yam but it definitely wasnt a lukewarm mushroom as murky and appealing as bong water"
        # text = "Oh and to top it all off the staff was taking photo in front of the food so we couldn't get in and the manager was five feet away not even paying attention "
        # text = "When I made the reservation they made sure to ask when and where our show was to ensure we had enough time to dine and still make it to our show"
        # text = "Very nice little place. Prices are reasonable and food was good. Staff is courteous and patient with a non Spanish speaking guy...lol."
        #text = "food was very good , but the service was amazing"
        #text = "The parents of two of the seven cousins killed in that crash have sued the truck's driver and the trucking company"
        textWordsArray = np.array(text_clean.lower().split())
        #textWordsArray
        textWordsIdx = []

        status = "SUCCESS"
        ignored_words = ""
        for index, w in enumerate(textWordsArray):
            if index > 499:
                status = "EXCEEDED_WORD_LIMIT"
                break;

            word = ''.join(c for c in w if (c.isalnum() or c == "'"))
            if word not in self.widx:
                textWordsIdx.append(self.vocabsize - 1)
                print(" not found word = "+word)
                ignored_words += word + ", "
            elif self.widx[word] > self.vocabsize - 1:
                textWordsIdx.append(self.vocabsize - 1)
                print("rare word = " + word)
            else:
                textWordsIdx.append(self.widx[word])

        textWordsIdxArray = [np.array(textWordsIdx)]
        #print textWordsIdx

        #print textWordsIdxArray
        textIdxArrayPadded = sequence.pad_sequences(textWordsIdxArray, maxlen=self.STR_MAX_LEN, value=0)
        #print textIdxArrayPadded
        prediction = self.model.predict(textIdxArrayPadded, batch_size=1,verbose=1)
        sentiment_score = prediction[0][0]
        if sentiment_score > 0.60:
            sentiment = "POSITIVE"
        elif sentiment_score < 0.50:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"

        elapsed = (timeit.default_timer() - start_time) * 1000
        result_json = {
            'status': status,
            'user': user,
            'score': str(sentiment_score),
            'sentiment': sentiment,
            'text': text,
            'time_taken': str(round(elapsed, 0)) + ' ms',
            'untrained_words': ignored_words,
            'sentiment_manual': '',
            'reference_id': reference_id,
            'model_ver': self.MODEL_VERSION
        }

        result_json['_id'] = self.stdb.save_record(result_json)

        return result_json

