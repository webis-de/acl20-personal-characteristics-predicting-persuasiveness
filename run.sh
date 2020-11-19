#!/bin/bash

cd pyspark
make -s build
# =======================================================================================
spark-submit --master yarn --deploy-mode client --py-files dist/jobs.zip main.py --job other.corpus_clean --job-args subs-path=/corpora/corpora-thirdparty/corpora-reddit/corpus-submissions/ coms-path=/corpora/corpora-thirdparty/corpora-reddit/corpus-comments/ botlist-csv=/home/username/data/botlist.csv

# =======================================================================================
# pairs jobs
spark-submit --master yarn --deploy-mode client --executor-memory 25g --driver-memory 20g --py-files dist/jobs.zip main.py --job sub_com.threads --job-args subs-path=/user/username/data/output/sub_urls coms-path=/user/username/data/output/com_urls_duplicates
spark-submit --master yarn --deploy-mode client --py-files dist/jobs.zip main.py --job sub_com.pairs_wa --job-args input-job=sub_com_threads  botlist-csv=/home/username/data/botlist.csv
spark-submit --master yarn --deploy-mode client --py-files dist/jobs.zip main.py --job sub_com.pairs_wa_split --job-args input-job=sub_com_pairs_wa

# malleability
spark-submit --master yarn --deploy-mode client --executor-memory 25g --driver-memory 20g --py-files dist/jobs.zip main.py --job sub_com.pairs_wa_malleability --job-args input-job=sub_com_threads  botlist-csv=/home/username/data/botlist.csv

# =======================================================================================
# subreddit features

spark-submit --master yarn --deploy-mode client --py-files dist/jobs.zip main.py --job author.subreddit --job-args subs-path=/corpora/corpora-thirdparty/corpora-reddit/corpus-submissions coms-path=/corpora/corpora-thirdparty/corpora-reddit/corpus-comments botlist-csv=/home/username/data/botlist.csv
spark-submit --master yarn --deploy-mode client --py-files dist/jobs.zip main.py --job author.subreddit_df --job-args input-job=author_subreddit
spark-submit --master yarn --deploy-mode client --py-files dist/jobs.zip main.py --job author.subreddit_vec --job-args input-job=author_subreddit subreddit-df-pickle=/home/username/data/output/_jobs/subreddit_df.pickle

spark-submit --master yarn --deploy-mode client --py-files dist/jobs.zip main.py --job author.subreddit --job-args subs-path=/user/username/data/output/_jobs/cleaned_cmv_authors/subs/ coms-path=/user/username/data/output/_jobs/cleaned_cmv_authors/coms/ botlist-csv=/home/username/data/botlist.csv
spark-submit --master yarn --deploy-mode client --py-files dist/jobs.zip main.py --job author.subreddit_df --job-args input-job=author_subreddit
spark-submit --master yarn --deploy-mode client --py-files dist/jobs.zip main.py --job author.subreddit_vec --job-args input-job=author_subreddit subreddit-df-pickle=/home/username/data/output/_jobs/subreddit_df.pickle


# =======================================================================================
# subreddit-category features

spark-submit --master yarn --deploy-mode client --py-files dist/jobs.zip main.py --job author.category --job-args input-job=author_subreddit_vec subreddit-category-pickle=/home/username/data/output/_jobs/subreddit_category.pickle
spark-submit --master yarn --deploy-mode client --py-files dist/jobs.zip main.py --job author.category_df --job-args input-job=author_category
spark-submit --master yarn --deploy-mode client --py-files dist/jobs.zip main.py --job author.category_vec --job-args input-job=author_category category-df-pickle=/home/username/data/output/_jobs/category_df.pickle subreddit-category-pickle=/home/username/data/output/_jobs/subreddit_category.pickle

spark-submit --master yarn --deploy-mode client --py-files dist/jobs.zip main.py --job author.category_subreddit_vec --job-args input-job=author_category_vec subreddit-df-pickle=/home/username/data/output/_jobs/subreddit_df.pickle subreddit-category-pickle=/home/username/data/output/_jobs/subreddit_category.pickle

spark-submit --master yarn --deploy-mode client --py-files dist/jobs.zip main.py --job author.topcat --job-args input-job=author_subreddit_vec subreddit-category-pickle=/home/username/data/output/_jobs/subreddit_category.pickle
spark-submit --master yarn --deploy-mode client --py-files dist/jobs.zip main.py --job author.subreddit_topcategories --job-args input-job=author_topcat
spark-submit --master yarn --deploy-mode client --executor-memory 45g --driver-memory 10g --conf spark.yarn.executor.memoryOverhead=10000 --py-files dist/jobs.zip main.py --job author.subreddit_categories_pca


# =======================================================================================
# author_entity features

spark-submit --master yarn --deploy-mode client --executor-memory 15g --driver-memory 10g --conf spark.yarn.executor.memoryOverhead=5000 --py-files dist/jobs.zip,dist/libs.zip main.py --job author.entity --job-args subs-path=/user/username/data/output/_jobs/cleaned_cmv_authors/subs/ coms-path=/user/username/data/output/_jobs/cleaned_cmv_authors/coms/ stopwords-csv=/home/username/data/ranksnl_stopwords.csv botlist-csv=/home/username/data/botlist.csv sem-path=/home/username/src/tools/semanticizest/semanticizest3.zip sem-model-path=hdfs://hadoop:8020/user/username/data/enwiki_pages_n3_1.model
spark-submit --master yarn --deploy-mode client --executor-memory 15g --driver-memory 10g --conf spark.yarn.executor.memoryOverhead=5000 --py-files dist/jobs.zip main.py --job author.entity_df --job-args input-job=author_entity
spark-submit --master yarn --deploy-mode client --py-files dist/jobs.zip main.py --job author.entity_vec --job-args input-job=author_entity entity-df-pickle=/home/username/data/output/_jobs/entity_df.pickle

# =======================================================================================
# entity-category features
# sentity-category (pca) features

spark-submit --master yarn --deploy-mode client --py-files dist/jobs.zip main.py --job author.entity_category_vec --job-args input-job=author_entity
spark-submit --master yarn --deploy-mode client --executor-memory 45g --driver-memory 10g --conf spark.yarn.executor.memoryOverhead=10000 --py-files dist/jobs.zip main.py --job author.entity_category_pca

spark-submit --master yarn --deploy-mode client --executor-memory 15g --driver-memory 10g --conf spark.yarn.executor.memoryOverhead=10000 --py-files dist/jobs.zip main.py --job author.sentity_category_vec --job-args input-job=author_entity
spark-submit --master yarn --deploy-mode client --executor-memory 45g --driver-memory 10g --conf spark.yarn.executor.memoryOverhead=10000 --py-files dist/jobs.zip main.py --job author.sentity_category_pca

# =======================================================================================
# entity-topics features

spark-submit --master yarn --deploy-mode client --executor-memory 10g --driver-memory 10g --conf spark.yarn.executor.memoryOverhead=10000 --py-files dist/jobs.zip main.py --job author.topic --job-args input-job=author_entity page-topic-pickle=/home/username/data/output/_jobs/page_topics.pickle

spark-submit --master yarn --deploy-mode client --executor-memory 20g --driver-memory 20g --conf spark.yarn.executor.memoryOverhead=10000 --py-files dist/jobs.zip main.py --job author.topic_entity_vec --job-args input-job=author_topic

# =======================================================================================
# post jobs
spark-submit --master yarn --deploy-mode client --executor-memory 20g --driver-memory 30g --py-files dist/jobs.zip main.py --job sub_com.post_features

# =======================================================================================
# csv jobs

spark-submit --master yarn --deploy-mode client --executor-memory 15g --driver-memory 30g --conf spark.yarn.executor.memoryOverhead=10000 --conf spark.yarn.executor.memoryOverhead=10000 --py-files dist/jobs.zip main.py --job sub_com.category_subreddit_csv --job-args sub-com=/user/username/data/output/_jobs/sub_com_pairs/latest subreddit-df-pickle=/home/username/data/output/_jobs/subreddit_df.pickle subreddit-category-pickle=/home/username/data/output/_jobs/subreddit_category.pickle

# =======================================================================================
spark-submit --master yarn --deploy-mode client --executor-memory 15g --driver-memory 30g --conf spark.yarn.executor.memoryOverhead=5000 --py-files dist/jobs.zip main.py --job other.export_author_features


# =======================================================================================
# temp jobs

### history-words
spark-submit --master yarn --deploy-mode client --executor-memory 10g --py-files dist/jobs.zip,dist/libs.zip main.py --job author.history --job-args stopwords-csv=/home/username/data/ranksnl_stopwords.csv

spark-submit --master yarn --deploy-mode client --executor-memory 20g --conf spark.yarn.executor.memoryOverhead=10000 --py-files dist/jobs.zip main.py --job sub_com.pairs_joined

spark-submit --master yarn --deploy-mode client --py-files dist/jobs.zip main.py --job other.corpus_clean --job-args subs-path=/corpora/corpora-thirdparty/corpora-reddit/corpus-submissions/ coms-path=/corpora/corpora-thirdparty/corpora-reddit/corpus-comments/ botlist-csv=/home/username/data/botlist.csv

spark-submit --master yarn --deploy-mode client --executor-memory 25g --conf spark.yarn.executor.memoryOverhead=20000 --num-executors 5 --executor-cores 5 --py-files dist/jobs.zip,dist/libs.zip main.py --job other.entity_tmp --job-args stopwords-csv=/home/username/data/ranksnl_stopwords.csv botlist-csv=/home/username/data/botlist.csv sem-path=/home/username/src/tools/semanticizest/semanticizest3.zip sem-model-path=hdfs://hadoop:8020/user/username/data/enwiki_pages_n3_1.model
