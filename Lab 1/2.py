import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


spark = SparkSession.builder.config("spark.driver.memory", "10g").master('local[*]').appName('lab1')\
    .config('spark.jars.packages', 'org.apache.spark:spark-avro_2.12:3.4.3')\
    .getOrCreate()
df_orders = spark.read.csv('./orders', header=True, inferSchema=True)
df_orders = df_orders.withColumn('date', F.to_date(F.col('date_and_time')))

condition = F.col('city').isNull() | F.col('price').isNull()
df_bad_rows = df_orders.where(condition)
df_bad_rows.write.mode('overwrite').parquet('bad_orders_parquet')

df_orders = df_orders.where(~condition).cache()

df_orders.write\
    .partitionBy("date", 'city')\
    .mode('overwrite')\
    .parquet('good_orders_parquet')

df_orders.write\
    .partitionBy("date", 'city')\
    .mode('overwrite')\
    .format("avro")\
    .save("good_orders_avro")
