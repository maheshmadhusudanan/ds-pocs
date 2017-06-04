'''
Created on Jun 2, 2017

@author: Mahesh.M
'''
import xml.etree.ElementTree as ET
import utils
from utils import *
import numpy as np
import os
import timeit
from keras.datasets import imdb

class SentimentGenerator:
    current_dir = os.getcwd()
    DATA_DIR = current_dir + "/data"
    ORIGINAL_DATA = DATA_DIR + "/original"
    TRAIN_DATA = DATA_DIR + "/train"
    NEG_REVIEWS_TRAIN_FILE = TRAIN_DATA + "/negetivereviews.txt"
    POS_REVIEWS_TRAIN_FILE = TRAIN_DATA + "/positivereviews.txt"
    MODEL_PATH = current_dir + "/models/"
    WORDS_TO_IDX_DATA_FILE = DATA_DIR + "/words_to_idx.p"
    IDX_TO_WORD_DATA_FILE = DATA_DIR + "/idx_to_words.p"
    STR_MAX_LEN = 500
    vocabsize = 60000
    model = Sequential()
    widx = None
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
        self.model.add(Dropout(0.7))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss="binary_crossentropy", optimizer=Adam(),  metrics=['accuracy'])
        self.model.summary()
        #
        # load the weights
        #
        self.model.load_weights(self.MODEL_PATH + 'sentiment_model_v1_0.h5')


    def runSentiment(self,text):
        #
        # split the sentence into words and prepare an index
        #
        # text = "dont know what could have saved limp dispiriting yam but it definitely wasnt a lukewarm mushroom as murky and appealing as bong water"
        # text = "Oh and to top it all off the staff was taking photo in front of the food so we couldn't get in and the manager was five feet away not even paying attention "
        # text = "When I made the reservation they made sure to ask when and where our show was to ensure we had enough time to dine and still make it to our show"
        # text = "Very nice little place. Prices are reasonable and food was good. Staff is courteous and patient with a non Spanish speaking guy...lol."
        #text = "food was very good , but the service was amazing"
        #text = "The parents of two of the seven cousins killed in that crash have sued the truck's driver and the trucking company"
        textWordsArray = np.array(text.lower().split())
        #textWordsArray
        textWordsIdx = []

        for w in textWordsArray:
            word = ''.join(c for c in w if c.isalnum())
            if word not in self.widx:
                textWordsIdx.append(self.vocabsize - 1)
                print(" not found word = "+word)
            elif self.widx[word] > self.vocabsize -1:
                textWordsIdx.append(self.vocabsize - 1)
                print("rare word = "+ word)
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
        elif sentiment_score < 0.40:
            sentiment = "NEGETIVE"
        else:
            sentiment = "NEUTRAL"

        # result_json = {
        #     'status': "SUCCESS",
        #     'score': str(s),
        #     'sentiment': t,
        #     'text': request.form['text'],
        #     'time_taken': str(round(elapsed, 0)) + ' ms'
        # }

        return sentiment_score, sentiment

