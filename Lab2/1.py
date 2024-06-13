from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, VectorIndexer, StringIndexer, IndexToString
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
import matplotlib.pyplot as plt


spark = SparkSession.builder.appName("Iris").master('local[*]').getOrCreate()

# Считаем датасет, выведем информацию о нем
dataset = spark.read.csv('Iris.csv', header=True, inferSchema=True)
real_iris_pd = dataset.select(dataset.columns[1:]).toPandas()
dataset.show(5)
dataset.printSchema()

# Небольшая предобработка данных
vec_assembler = VectorAssembler(inputCols=dataset.columns[1:-1], outputCol='features')

data = vec_assembler.transform(dataset).select('features')

# Посмотрим на график коэффициента силуэта, на основе которого часто выбирают количество кластеров
silhouette_score = []

evaluator = ClusteringEvaluator(featuresCol='features',
                                predictionCol='prediction',
                                metricName='silhouette',
                                distanceMeasure='squaredEuclidean')

for i in range(2, 10):
    kmeans = KMeans(featuresCol='features', k=i)
    model = kmeans.fit(data)
    predictions = model.transform(data)
    score = evaluator.evaluate(predictions)
    silhouette_score.append(score)
    print('Silhouette Score for k =', i, 'is', score)

plt.plot(range(2, 10), silhouette_score)
plt.xlabel('k')
plt.ylabel('silhouette score')
plt.title('Silhouette Score')
plt.show()

# Мы знаем, что кластеров 3, поэтому запустим алгоритм именно с таким значением k
kmeans = KMeans(featuresCol='features', k=3)
model = kmeans.fit(data)
predictions = model.transform(data)

centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

# Визуализируем кластеры по двумерному подпространству, для этого переведем спарк-Df в Pandas
df_pandas = predictions.toPandas()
df_pandas['PW'] = df_pandas.features.str[3]
df_pandas['PL'] = df_pandas.features.str[2]

df_zero_P = df_pandas[df_pandas.prediction == 0][['PW', "PL"]]
df_one_P = df_pandas[df_pandas.prediction == 1][['PW', "PL"]]
df_two_P = df_pandas[df_pandas.prediction == 2][['PW', "PL"]]


colors = ['red', 'blue', 'green']
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

names_irises = real_iris_pd.Species.unique().tolist()
for i, name in enumerate(names_irises):
    print(real_iris_pd)
    df = real_iris_pd[real_iris_pd.Species == name]
    print(df)
    ax.scatter(df['PetalWidthCm'], df['PetalLengthCm'], alpha=0.9, s=100, marker='+',
               c=colors[(i+5) % 3], linewidths=2.5, label=name)


ax.set_xlabel('Petal width, cm')
ax.set_ylabel('Petal length, cm')
ax.scatter(df_zero_P['PW'], df_zero_P['PL'], alpha=0.65, c=colors[0], label='Zero')
ax.scatter(df_one_P['PW'], df_one_P['PL'], alpha=0.65, c=colors[1], label='One')
ax.scatter(df_two_P['PW'], df_two_P['PL'], alpha=0.65, c=colors[2], label="Two")


for i, center in enumerate(centers):
    ax.scatter(center[3], center[2], alpha=0.5, c=colors[i], s=200, linewidths=2.5)

ax.legend()
fig.show()


# Далее классифицируем данные с помощью RF.
# Создадим пайплайн нашей модели, включающий предобработку
labelIndexer = StringIndexer(inputCol="Species", outputCol="indexedLabel").fit(dataset)

labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

(trainingData, testData) = dataset.randomSplit([0.7, 0.3])

rfc = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features")

pipeline = Pipeline(stages=[labelIndexer, vec_assembler, rfc, labelConverter])

model = pipeline.fit(trainingData)

# Делаем предсказания
predictions = model.transform(testData)

# Небольшая визуализация
predictions.groupBy('predictedLabel', 'Species').count().show()
predictions.show()
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))


gbtModel = model.stages[2]
print(gbtModel)  # Параметры модели

