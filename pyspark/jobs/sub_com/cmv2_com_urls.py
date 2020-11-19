from pyspark import SparkContext, SparkConf
import json
import re
from urllib.parse import urlparse

conf = SparkConf().setAppName("cmv")
sc = SparkContext(conf=conf)

def jsonloads(l):
    try:
        return json.loads(l)
    except:
        return ""

def url_parse(x):
    return ".".join(x.split('.')[-2:]) if x.find("co.uk") == -1 else ".".join(x.split('.')[-3:])

def urls(s, prop):
    ### urls
    ANY_URL_REGEX = r"""(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"""
    DOMAIN_REGEX = r"""\/\/(?:[^@\/\n]+@)?(?:www\.)?([^:\/\n]+)"""
    anyurl = re.compile(ANY_URL_REGEX)
    domainregex = re.compile(DOMAIN_REGEX)

    text = s[prop]
    
    urls = re.findall(anyurl, text)
    s['urls'] = list( map( lambda x : url_parse(x), filter(lambda x : x != None and not any(u in x for u in \
        ["reddit", "delta", "CMVModBot"]), map(lambda x : x[1], urls)) ) )
    
    return s

com_file = sc.textFile('/corpora/corpora-thirdparty/corpora-reddit/corpus-comments/')
com_json = com_file.map(lambda l : jsonloads(l) ).filter(lambda l : l is not "")
print("\nReddit sub total count: "+str(com_json.count()))
com_json = com_json.filter(lambda l : "changemyview" in l.get('subreddit', ""))
print("\nReddit CMV sub count: "+str(com_json.count()))

### duplicates - take latest by retrieval time ('retrieved_on': 1504556955,)
reduced = com_json.map(lambda x: (x['id'], x)).reduceByKey(lambda a,b: a if a['retrieved_on'] > b['retrieved_on'] else b)
mapped = reduced.map(lambda x: x[1])

com_processed = mapped.map(lambda s : urls(s, 'body'))
com_processed = com_processed.map(lambda l : json.dumps(l, ensure_ascii=False).encode('utf8'))
com_processed.saveAsTextFile("hdfs://hadoop:8020/user/username/data/cmv_com_urls")