from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType
from pyspark.ml.feature import Tokenizer, Word2Vec, StopWordsRemover, StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.regression import LinearRegression, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from nltk.probability import FreqDist

import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import sys
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder.appName("Youtube")\
    .config("spark.driver.memory", "10g").master('local[*]').getOrCreate()

data = spark.read.csv('youtube_channels_1M_clean.csv', header=True, inferSchema=True,
                      multiLine=True, sep=',', escape="\"")
data.show()

# Создадим столбец со счетчиком для стран
w = Window.partitionBy('country')
data = data.withColumn('count_country', F.count('country').over(w))
data.groupBy('country').count().orderBy('count').show(n=400)


#############################################
##   1  Кластеризация по ключевым словам   ##
#############################################

def text_prep(text):
    # Переводим текст в нижний регистр
    text = str(text).lower()
    # Убираем все, кроме букв
    text = re.sub(r'[^a-z]', ' ', text)
    # Убираем единичные буквы
    text = re.sub(r'\b\w\b', ' ', text)
    # Если пробелов больше одного заменяем на один
    text = re.sub(r'\s+', ' ', text)
    # Удаляем повторяющиеся слова
    text = ' '.join(set(word for word in text.split()))
    # Убираем пробелы слева и справа
    text = text.strip()
    return text


# создание пользовательской функции
prep_text_udf = udf(text_prep, StringType())

data_keywords = data.select('channel_id', 'keywords')
data_keywords = data_keywords.withColumn('prep_keywords', prep_text_udf('keywords'))
data_for_clustering = data_keywords.where(data_keywords['prep_keywords'] != 'none').limit(1000000)
data_for_clustering.select('prep_keywords').show(n=5, truncate=False)

# Построим пайплайн преобразования текста в векторы
tokenizer = Tokenizer(inputCol='prep_keywords', outputCol='tokens')

stopwords = StopWordsRemover.loadDefaultStopWords('english')

stopwords_remover = StopWordsRemover(inputCol='tokens', outputCol='clear_tokens', stopWords=stopwords)

word2Vec = Word2Vec(inputCol='clear_tokens', outputCol='w2v_features', vectorSize=100, minCount=100)

pipeline = Pipeline(stages=[tokenizer, stopwords_remover, word2Vec])
fit_pipelane = pipeline.fit(data_for_clustering)

data_with_vectors = fit_pipelane.transform(data_for_clustering)

# Посмотрим на получившийся DataFrame
data_with_vectors.show(n=2)

# Количество уникальных документов
print(data_with_vectors.groupBy('clear_tokens').count().count())

# Как выглядят word2vec векторы
data_with_vectors.select('w2v_features').show(n=2, truncate=False)

# Посмотрим, что модель word2vec считает самыми близкими словами к "food"
fit_pipelane.stages[-1].findSynonyms(word='food', num=7).show()

# Применим KMeans к полученным векторам
kmeans = KMeans(k=15, initMode="k-means||", featuresCol='w2v_features', predictionCol='cluster', maxBlockSizeInMB=64)
km_model = kmeans.fit(data_with_vectors)
clustering_data = km_model.transform(data_with_vectors)

# Далее посмотрим топ часто встречающихся слова в каждом кластере
d = {}
for i in range(15):
    ls = []
    tmp = clustering_data.select('clear_tokens', 'cluster').where(clustering_data['cluster'] == i).collect()

    tmp = [tmp[j][0] for j in range(len(tmp))]

    for el in tmp:
        ls.extend(el)

    fdist = list(FreqDist(ls))[:5]

    d[i] = fdist

d_frame = pd.DataFrame(list(d.items()), columns=['cluster', 'top_words'])
print(d_frame.to_string())


########################################################
##   2  Зависимость предпочтения контента от страны   ##
########################################################

def translate(dictionary):
    return udf(lambda col: dictionary.get(col),
               ArrayType(StringType()))


# Соединим данные о кластерах с данными о станах
data_for_preference = clustering_data.select('cluster', 'channel_id')\
                                     .join(data.where(data['count_country'] > 3000)
                                     .select('channel_id', 'country', 'total_videos'), how='inner', on='channel_id')

# Сгруппируем данные по кластерам и странам, и найдем кластеры, у которых выкладывалось больше всего для данной страны
windowDept = Window.partitionBy("country").orderBy(F.col("sum_videos").desc())
data_for_preference = data_for_preference.groupBy('cluster', 'country').agg(F.sum('total_videos').alias("sum_videos"))\
                                         .withColumn("row", F.row_number().over(windowDept))\
                                         .filter(F.col("row") <= 5)

data_for_preference = data_for_preference.withColumn('cluster_topics', translate(d)('cluster'))
data_for_preference.show()


def pie_clusters_for_country(df, country):
    df_country = df[df['country'] == country]
    print(df_country['sum_videos'])
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.title(f'Top 5 cluster in {country}')
    ax.pie(df_country['sum_videos'], labels=df_country['cluster_topics'],
           autopct=lambda x: '{:.0f}'.format(x*df_country['sum_videos'].sum()/100))
    fig.show()


pie_clusters_for_country(data_for_preference.toPandas(), 'India')


#############################################
##   3  Предсказывание числа подписчиков   ##
#############################################

features_from_data = ['channel_id', 'country', 'total_videos', 'videos_per_week', 'total_views', 'subscriber_count']
data_for_regression = clustering_data.select('cluster', 'channel_id')\
                                     .join(data.where(data['count_country'] > 3000)
                                     .select(features_from_data),
                                     how='inner', on='channel_id')

# Заменим все null на нули
data_for_regression = data_for_regression.na.fill(0)

# Преобразуем кластеры и страны в OneHot вектора
country_indexer = StringIndexer(inputCol='country', outputCol='index_country')
data_country_index = country_indexer.fit(data_for_regression).transform(data_for_regression)
onehot_country = OneHotEncoder(inputCol='index_country', outputCol='one_hot_country')
data_one_hot_countries = onehot_country.fit(data_country_index).transform(data_country_index)

onehot_cluster = OneHotEncoder(inputCol='cluster', outputCol='one_hot_cluster')
data_one_hot_clusters = onehot_cluster.fit(data_one_hot_countries).transform(data_one_hot_countries)

# zzСоздадим столбец features с вектором фичей
features = ['one_hot_country', 'total_videos', 'videos_per_week', 'total_views', 'one_hot_cluster']
assembler = VectorAssembler(inputCols=features, outputCol='features')
data_with_vec_features = assembler.transform(data_one_hot_clusters)

# Разделим выборку на train и test
train, test = data_with_vec_features.randomSplit([0.8, 0.2])

train.show()
# Обучим модель и сделаем предсказания
linear_regression = GBTRegressor(featuresCol='features', predictionCol='predictions', labelCol='subscriber_count',
                                     ).fit(train)

predictions = linear_regression.transform(test)
predictions.select('predictions', 'subscriber_count').show()

# Посмотрим на точность модели
print(RegressionEvaluator(labelCol='subscriber_count', predictionCol='predictions', metricName='r2').evaluate(predictions))

