import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from bs4 import BeautifulSoup as bfs
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import InvalidArgumentException, WebDriverException
from elasticsearch import Elasticsearch
from elasticsearch.client import IndicesClient


def get_transcript(index, url):
    """
    Get a single transcript from the given url
    """

    url += '/transcript'
    browser = webdriver.Chrome(
        executable_path=CHROME_PATH, options=browser_options, service=browser_service
    )

    try:
        browser.get(url)
    except InvalidArgumentException:
        browser.close()
        return index, None  # return None if request error

    html = browser.page_source
    browser.close()

    # getting transcripts using their specific class names
    bfs_html = bfs(html, features="html.parser")
    transcript_bfs_list = bfs_html.find_all(
        'span', attrs={'class': 'cursor-pointer inline hover:bg-red-300 css-82uonn'}
    )

    # creating an array of a cleaned talk transcript
    if transcript_bfs_list:
        transcript_list = [transcript.text.strip() for transcript in transcript_bfs_list]
    else:
        return index, None  # return None if transcript was not found

    return index, ' '.join(transcript_list)


##############################################################################
#################################### Main ####################################
##############################################################################

# need to install google-chrome & an download chromedriver for selenium to work
# path to google-chrome & chromedriver or command to run them (if added to windows PATH)
# in windows:
# CHROME_PATH = "C:\Program Files\Google\Chrome\Application\chrome.exe"
# CHROME_DRIVER_PATH = "./chromedriver.exe"

DATA_PATH = "data.csv"
ES_HOST = "http://127.0.0.1:9200"
INDEX_NAME = "ted-talk-index"
CHROME_PATH = "google-chrome"
CHROME_DRIVER_PATH = "chromedriver"


# reading data
df = pd.read_csv(DATA_PATH)

# Initializing a headless browser for selenium
# Selenium is used instead of requests lib, because
# Ted webpage loads transcripts after loading main website (JS Async)
browser_options = webdriver.ChromeOptions()
browser_options.add_argument('--headless')
browser_options.add_argument('--no-sandbox')
browser_options.add_argument('--disable-dev-shm-usage')
browser_options.add_argument("--lang=en-US")  # making sure we get english transcript
browser_service = Service(CHROME_DRIVER_PATH)


# running multi-threaded to increase speed
with ProcessPoolExecutor(max_workers=5) as executor:
    # initiating threads
    results = [executor.submit(get_transcript, index, row['link']) for index, row in df.iterrows()]
    for result in as_completed(results):
        try:
            index, transcript = result.result()
            df.loc[index, 'transcript'] = transcript
            if transcript is not None:
                print(
                    f'Completed: {index},\t Remaining: {len(df) - len(df.loc[~df["transcript"].isnull()])}'
                )
        except WebDriverException as e:
            print(e)  # ignoring chromedriver errors to continue the runs

# saving new data
df.to_csv("data_with_transcript.csv", index=False)

# initializing elasticsearch client
es = Elasticsearch(ES_HOST)
es_indices_client = IndicesClient(es)  # for using elasticsearch analyzers

# indexing talks+transcripts if transcript is not Null
# some talks in data don't have a transcript avaliable in ted website
for idx, row in df.loc[~df['transcript'].isnull()].iterrows():
    # normalizing transcripts: remove stop-words and punctuations, lowercase, ...
    normal_transcript_tokens = es_indices_client.analyze(analyzer="stop", text=row['transcript'])
    row['transcript'] = ' '.join([token['token'] for token in normal_transcript_tokens['tokens']])
    es.index(index=INDEX_NAME, id=idx, document=row.to_dict())  # indexing document

# printing a test indexed item
response = es.get(index=INDEX_NAME, id=0)
print(response['_source'])
