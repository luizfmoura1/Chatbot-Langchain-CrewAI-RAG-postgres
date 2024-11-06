import redis
from redis.commands.search.field import *
from redis.commands.search.indexDefinition import *

if __name__ == "__main__":
    client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    if client.ping():
        print('CONECTADO')
    idx = client.ft('idx:rapadura')
    try:
        print(idx.info())
    except Exception as e:
        idx.create_index(
            fields=[
                VectorField(
                    name="embedding",
                    algorithm="FLAT",
                    attributes={
                        "TYPE": "FLOAT32",
                        "DIM": 128,
                        "DISTANCE_METRIC": "COSINE"
                    }
                ),
                TextField('content')
            ],
            
            definition=IndexDefinition(prefix=["emb:"], index_type=IndexType.HASH)
        )
        print(idx.info())