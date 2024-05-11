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
df_orders = df_orders.withColumn('Sum_price', F.sum(df_orders['price'] * df_orders['Count_meal']).over(w))
# df_orders.show()
df_meals = df_orders.select('id_order', 'Menu_item', "Price", 'Count_Meal')
df_orders = df_orders.select('id_order', 'Date_and_Time', 'Sum_price')
df_combined = df_orders.join(df_meals
                             .groupBy('id_order')
                             .agg(F.collect_list(F.struct('Menu_item', 'Price', 'Count_Meal')).alias('meals')),
                             on='id_order', how='left').dropDuplicates()
# df_combined.show()
# df_combined.printSchema()

df_combined.write.format("com.mongodb.spark.sql")\
        .option('uri', "mongodb://dan:2008@localhost:27017")\
        .option('database', "My_db")\
        .option('collection', "orders").save()

# df = spark.read.format("com.mongodb.spark.sql")\
#         .option('uri', "mongodb://dan:2008@localhost:27017")\
#         .option('database', "dabse")\
#         .option('collection', "cole").load()
# df.show()
