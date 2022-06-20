import fasttext
import gensim
import pathlib, shutil
from elasticsearch import Elasticsearch
from rich.progress import track

# global variables
ES_HOST = "http://127.0.0.1:9200"
INDEX_NAME = "ted-talk-index"


es = Elasticsearch(ES_HOST)

# getting saved index
index_size = es.count(index=INDEX_NAME)["count"]
results = es.search(index=INDEX_NAME, query={"match_all": {}}, size=index_size)["hits"]["hits"]

# creating temp dir
temp_path = pathlib.Path("temp/")
temp_path.mkdir(parents=True, exist_ok=True)

# saving all of the transcripts into files at temp dir
# because fasttext only accepts file_path as input
file_path_per_doc = {}
for result in track(results, description="Saving transcript files ..."):
    doc_id = result["_id"]
    transcript = result["_source"]["transcript"]
    file_name = f"{temp_path}/transcript_{doc_id}.txt"

    with open(file_name, "w") as file:
        file.write(transcript)
    file_path_per_doc[doc_id] = file_name

# creating vector models for each transcript
# getting top 10 frequent words and saving them as comma separated labels field for each document in index
for doc_id, file_name in track(file_path_per_doc.items(), description="Labelling documents"):
    model = fasttext.train_unsupervised(file_name, minCount=0)
    sorted_freq_words = [
        word
        for word in model.words
        if word not in gensim.parsing.preprocessing.STOPWORDS and len(word) >= 3
    ]
    labels = ",".join(sorted_freq_words[:10])
    es.update(index=INDEX_NAME, id=doc_id, doc={"labels": labels})


# Removing temp transcript files
shutil.rmtree(temp_path)
