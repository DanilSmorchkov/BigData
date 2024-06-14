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

df_orders = spark.read.format('avro').load('good_orders_avro').drop('date')
w = Window.orderBy(F.lit("city"))
df_cafe_dim = df_orders.select('cafe_name', 'lat', 'lng', 'city').dropDuplicates().orderBy('city')\
    .select('*', F.row_number().over(w).alias('id_cafe'))

df_orders_facts = df_orders.join(df_cafe_dim, ['cafe_name', 'lat', 'lng', 'city'], how='left')\
    .drop('cafe_name', 'lat', 'lng', 'city')
df_meal_dim = df_orders_facts.select('menu_item').dropDuplicates().orderBy('menu_item')\
    .select('*', F.row_number().over(w).alias('id_meal'))

df_orders_facts = df_orders_facts.join(df_meal_dim, ['menu_item'], how='left') .drop('menu_item')\
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
