import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


spark = SparkSession.builder.config("spark.driver.memory", "10g").master('local[*]').appName('lab1')\
    .config('spark.jars.packages', 'org.apache.spark:spark-avro_2.12:3.4.3')\
    .getOrCreate()
df_orders = spark.read.csv('./orders', header=True, inferSchema=True)
df_orders = df_orders.withColumnRenamed('Date & Time', 'Date_and_Time')
df_orders = df_orders.withColumnRenamed('Café Name', 'Cafe_name')
df_orders = df_orders.withColumnRenamed('Menu item', 'Menu_item')
df_orders = df_orders.withColumnRenamed('Count meal', 'Count_Meal')
df_orders = df_orders.withColumn('Date', F.to_date(F.col('Date_and_Time')))
df_orders.show()
condition = F.col('City').isNull() | F.col('Price').isNull()
df_bad_rows = df_orders.where(condition)
df_bad_rows.write.mode('overwrite').parquet('bad_orders_parquet')
# df_bad_rows.show()
# print((df_bad_rows.count(), len(df_orders.columns)))
# print((df_orders.count(), len(df_orders.columns)))

df_orders = df_orders.where(~condition)

# print((df_orders1.count(), len(df_orders1.columns)) == (df_orders.dropna().count(), len(df_orders.dropna().columns)))

df_orders.write\
    .partitionBy("Date", 'City')\
    .mode('overwrite')\
    .parquet('good_orders_parquet')

df_orders.write\
    .partitionBy("Date", 'City')\
    .mode('overwrite')\
    .format("avro")\
    .save("good_orders_avro")
