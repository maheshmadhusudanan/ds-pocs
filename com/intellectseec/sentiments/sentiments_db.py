import json
from pymongo import MongoClient
from datetime import datetime

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

    def get_records(self, start, limit_size):
        results = []
        count = self.sentiment_collection.find({}).count()
        if start > 0:
            cursor = self.sentiment_collection.find({}).skip(start).limit(limit_size)
        else:
            cursor = self.sentiment_collection.find({}).limit(limit_size)
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
                 'sentiment_manual': doc.get('sentiment_manual')}
            results.append(r)

        resp = {'totalCount':count,'results': results}

        return resp
