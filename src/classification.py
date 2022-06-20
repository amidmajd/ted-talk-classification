import fasttext
from elasticsearch import Elasticsearch
from sklearn.model_selection import train_test_split


ES_HOST = "http://127.0.0.1:9200"
INDEX_NAME = "ted-talk-index"


def save_transcript_with_labels(elastic_data_output, file_name):
    with open(file_name, "w") as file:
        text_to_write = []
        for data in elastic_data_output:
            transcript = data["_source"]["transcript"]
            labels = data["_source"]["labels"].split(",")
            text_to_write.append(
                f"{' '.join([f'__label__{label}' for label in labels])} {transcript}\n"
            )
        file.writelines(text_to_write)


es = Elasticsearch(ES_HOST)

# getting new labeled index and saving it after splitting to train and test
index_size = es.count(index=INDEX_NAME)["count"]
results = es.search(index=INDEX_NAME, query={"match_all": {}}, size=index_size)["hits"]["hits"]

train, test = train_test_split(results, test_size=0.9, random_state=444)
save_transcript_with_labels(train, "train_data.txt")
save_transcript_with_labels(test, "test_data.txt")

# Modeling
classifier = fasttext.train_supervised(input="train_data.txt", lr=1.25, wordNgrams=3, epoch=5000)
classifier.save_model("classifier_model.bin")
# classifier = fasttext.load_model("classifier_model.bin")

example_text = """
We currently have enough fossil fuels to progressively transition off of them,
says climate campaigner Tzeporah Berman, but the industry continues to expand oil,
gas and coal production and exploration. With searing passion and unflinching nerve,
Berman reveals the delusions keeping true progress from being made 
and offers a realistic path forward: the Fossil Fuel Non-Proliferation Treaty.
"""

print("Example:\n", example_text)
print(classifier.predict(example_text, k=10))

# Model Evaluation
_, precision, recall = classifier.test("test_data.txt")
print(f"\n\nTest Precision: {precision}, Recall: {recall}")
