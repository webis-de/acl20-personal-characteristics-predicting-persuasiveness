import json
import pickle
import pprint
pp = pprint.PrettyPrinter(width=100)

# pairs = pickle.load(open('/home/username/data/output/_jobs/pairs_winargs.pickle','rb'))

authors_from_pairs = set(pickle.load(open('/home/username/data/output/_jobs/authors_from_pairs.pickle','rb')))
len(authors_from_pairs)

'''
Rerun  corpus_clean -> cleaned_cmv_authors2
because some authors where missing - don't know why...
'''

subs_pairauthors2 = sc.pickleFile('/user/username/data/output/_jobs/cleaned_cmv_authors2/subs')\
    .filter(lambda x: x['author'] in authors_from_pairs)
coms_pairauthors2 = sc.pickleFile('/user/username/data/output/_jobs/cleaned_cmv_authors2/coms')\
    .filter(lambda x: x['author'] in authors_from_pairs)
print(subs_pairauthors2.count(), coms_pairauthors2.count())


cmv_subs2 = subs_pairauthors2.filter(lambda l : "changemyview" in l.get('subreddit', ""))
cmv_coms2 = coms_pairauthors2.filter(lambda l : "changemyview" in l.get('subreddit', ""))
reduced = cmv_coms2.map(lambda x: (x['id'], x)).reduceByKey(lambda a,b: a if a['retrieved_on'] > b['retrieved_on'] else b)
cmv_coms2 = reduced.map(lambda x: x[1])
print(cmv_subs2.count(), cmv_coms2.count())


cmv_subs2_mapped = cmv_subs2.map(lambda x: {'author': x['author'], 'text': x['selftext'], 
                                            'type': 'sub', 'created_utc': x['created_utc'],
                                            'subreddit': x['subreddit'], 'link_id': x['id']})
cmv_coms2_mapped = cmv_coms2.map(lambda x: {'author': x['author'], 'text': x['body'], 
                                            'type': 'com', 'created_utc': x['created_utc'],
                                            'subreddit': x['subreddit'], 'link_id': x['link_id']})
united = cmv_subs2_mapped.union(cmv_coms2_mapped)

groupedbyauthor = united.groupBy(lambda x: x['author']).mapValues(list)


def count_type(posts, type):
    return len([p for p in posts if p['type'] == type])
def count_words(posts):
    return sum([len(p['text'].split(' ')) for p in posts])
author_stats = groupedbyauthor.map(lambda x: {'author': x[0], 'subs': count_type(x[1], 'sub'), 'coms': count_type(x[1], 'com'),
                                       'total': len(x[1]), 'total_words': count_words(x[1])}).collect()
authors_more_1000 = [w['author'] for w in author_stats if w['total_words'] >= 1000]


liwc_posts = groupedbyauthor.filter(lambda x: x[0] in set(authors_more_1000))
liwc_posts_collected = liwc_posts.collect()

pickle.dump(liwc_posts_collected, open('/home/username/data/output/_jobs/liwc_posts_collected.pickle','wb'))