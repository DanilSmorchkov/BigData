import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


spark = SparkSession.builder.config("spark.driver.memory", "10g").master('local[*]').appName('lab1')\
    .config('spark.jars.packages', 'org.apache.spark:spark-avro_2.12:3.4.3')\
    .config("spark.jars", r"C:\Users\User\.ivy2\jars\postgresql-42.7.3.jar") \
    .getOrCreate()
spark.sparkContext.setLogLevel('WARN')

df_orders = spark.read.format('avro').load('good_orders_avro').drop('Date')
w = Window.orderBy(F.lit("City"))
df_cafe_dim = df_orders.select('Cafe_name', 'lat', 'lng', 'City').dropDuplicates().orderBy('City')\
    .select('*', F.row_number().over(w).alias('id_cafe'))
# df_cafe.show()
df_orders_facts = df_orders.join(df_cafe_dim, ['Cafe_name', 'lat', 'lng', 'City'], how='left')\
    .drop('Cafe_name', 'lat', 'lng', 'City')
df_meal_dim = df_orders_facts.select('Menu_item').dropDuplicates().orderBy('Menu_item')\
    .select('*', F.row_number().over(w).alias('id_meal'))
# df_meal.show()
df_orders_facts = df_orders_facts.join(df_meal_dim, ['Menu_item'], how='left') .drop('Menu_item')\
    .orderBy('id_order')

df_meal_dim.write.format('jdbc')\
    .mode('append')\
    .option('url', 'jdbc:postgresql://localhost:5432/star')\
    .option("driver", "org.postgresql.Driver").option("dbtable", "meal_dim") \
    .option("user", "postgresDB").option("password", "2008").save()

df_cafe_dim.write.format('jdbc')\
    .mode('append')\
    .option('url', 'jdbc:postgresql://localhost:5432/star')\
    .option("driver", "org.postgresql.Driver").option("dbtable", "cafe_dim") \
    .option("user", "postgresDB").option("password", "2008").save()

df_orders_facts.write.format('jdbc')\
    .mode('append')\
    .option('url', 'jdbc:postgresql://localhost:5432/star')\
    .option("driver", "org.postgresql.Driver").option("dbtable", "facts")\
    .option("user", "postgresDB").option("password", "2008").save()
