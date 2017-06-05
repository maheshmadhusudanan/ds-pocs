import json
from pymongo import MongoClient
from datetime import datetime
from bson.objectid import ObjectId

class SentimentsDB:

    host = "mongodb://localhost:27017"
    uid = ""
    pwd = ""
    client = None
    db = None
    sentiment_collection = None

    def __init__(self):
        print("############# initializing Mongo DB Client")
        self.client = MongoClient(self.host)
        self.db = self.client.sentiment_data_db
        self.sentiment_collection = self.db.sentiment_data_tags

    def save_record(self, entry):
        entry['updated_ts'] = datetime.now()
        result = self.sentiment_collection.insert_one(entry)
        print(" >> record inserted successfully id = "+str(result.inserted_id))
        return str(result.inserted_id)

    def update_record(self, rec_id, record):
        try:
            result = self.sentiment_collection.update_one({"_id": ObjectId(rec_id)},
                                                  {"$set": {"sentiment_manual": record['sentiment_manual']}})
            # print(" >> record updated successfully id = " + rec_id + " updated results  " + json.dumps(result))
            return True
        except Exception as e:
            print(str(e))
            return False

    def get_records(self, start, limit_size):
        results = []
        count = self.sentiment_collection.find({}).count()
        if start > 0:
            cursor = self.sentiment_collection.find({}).sort("updated_ts", -1).skip(start).limit(limit_size)
        else:
            cursor = self.sentiment_collection.find({}).sort("updated_ts", -1).limit(limit_size)
        for doc in cursor:
            # print(doc)
            r = {'_id': str(doc.get('_id')),
                 'status': doc.get('status'),
                 'score': doc.get('score'),
                 'sentiment': doc.get('sentiment'),
                 'user': doc.get('user'),
                 'text': doc.get('text'),
                 'updated_ts': str(doc.get('updated_ts')),
                 'time_taken': doc.get('time_taken'),
                 'untrained_words': doc.get('untrained_words'),
                 'reference_id': doc.get('reference_id'),
                 'model_ver': doc.get('model_ver'),
                 'sentiment_manual': doc.get('sentiment_manual')}

            results.append(r)

        resp = {'totalCount':count, 'results': results}

        return resp
