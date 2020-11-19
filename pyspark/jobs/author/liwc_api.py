'''
liwc

- iterate over author-texts (9709)
- for each author merge all posts (sorted by link_id, created_utc)
- request API
- results as {author: name, liwc: {json response} }
'''

import pickle
import json
import time
import random
import jsonlines
from requests import *

pp = pprint.PrettyPrinter(width=100)

apikey = ''
apisecret = ''
ping_url = 'https://api-v3.receptiviti.com/v3/api/ping'
content_api_url = 'https://api-v3.receptiviti.com/v3/api/content'

author_texts = pickle.load(open('/home/username/data/output/_jobs/liwc_posts_merged.pickle','rb'))

with jsonlines.open('/home/username/liwc_scores.jsonl', mode='w') as writer:
    for i, (author, text) in author_texts[:1]:
        response_json = test_ping()
        writer.write(response_json)
        print('#' + i + ' - ' + author + ' done.')
        time.sleep(random.randint(2, 6))


def merge(posts):
    _sorted = map(lambda x: x['text'], sorted(posts, key=lambda x: (x['link_id'], int(x['created_utc']))))
    return ' '.join(_sorted)


def test_ping():
    headers = auth_headers(apikey, apisecret)
    print("PING URL:---------> " + ping_url)
    response = get(ping_url, headers=auth_headers)
    response_json = json.loads(response.content.decode('utf-8'))

    return response_json


def get_score_content(content_data):
    headers = auth_headers(apikey, apisecret)
    response = post(content_api_url, json=content_data, headers=headers)

    response_json = json.loads(response.content.decode('utf-8'))
    if response.status_code != 200:
        print('Error: ' + response_json)

    return response_json


def base_headers(apikey, apisecret):
    header = auth_headers(apikey, apisecret)
    header['Content-type'] = 'application/json'
    return header


def auth_headers(apikey, apisecret):
    header = {}
    if apikey:
        header['X-API-KEY'] = apikey
    if apisecret:
        header['X-API-SECRET-KEY'] = apisecret
    return header
