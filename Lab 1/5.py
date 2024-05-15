import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


spark = SparkSession.\
        builder\
        .master('local[*]')\
        .appName("lab1").\
        config("spark.executor.memory", "1g")\
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.0").\
        getOrCreate()

df_orders = spark.read.format('parquet')\
        .option("header", "true")\
        .option("inferSchema", "true")\
        .load("good_orders_parquet")

w = Window.partitionBy('id_order')
df_orders = df_orders.withColumn('sum_price', F.sum(df_orders['price'] * df_orders['count_meal']).over(w))

df_meals = df_orders.select('id_order', 'menu_item', "price", 'count_meal')
df_orders = df_orders.select('id_order', 'date_and_time', 'sum_price')
df_combined = df_orders.join(df_meals
                             .groupBy('id_order')
                             .agg(F.collect_list(F.struct('menu_item', 'price', 'count_meal')).alias('meals')),
                             on='id_order', how='left').dropDuplicates()

df_combined.write.format("com.mongodb.spark.sql")\
        .option('uri', "mongodb://dan:2008@localhost:27017")\
        .option('database', "My_db")\
        .option('collection', "orders").save()

