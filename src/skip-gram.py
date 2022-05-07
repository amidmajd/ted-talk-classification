import pathlib, shutil
import fasttext
from elasticsearch import Elasticsearch
from rich.progress import track
from concurrent.futures import ProcessPoolExecutor, wait

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
list_of_file_paths = []
for result in track(results, description="Saving transcript files ..."):
    transcript_id = result["_id"]
    transcript = result["_source"]["transcript"]
    file_name = f"{temp_path}/transcript_{transcript_id}.txt"

    with open(file_name, "w") as file:
        file.write(transcript)

    list_of_file_paths.append(file_name)

# creating vector models for each transcript
transcript_vector_models = []
for file_name in list_of_file_paths[:4]:
    model = fasttext.train_unsupervised(file_name)
    transcript_vector_models.append(model)


# an example vector
print(transcript_vector_models[0].get_word_vector("the"))

# Removing temp transcript files
shutil.rmtree(temp_path)
