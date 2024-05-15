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

# Считаем данные
df_orders = spark.read.format('avro').load('good_orders_avro').drop('date')
# выделим отдельный фрейм для информации о кафе (добавим в него id_cafe)
w = Window.orderBy(F.lit("city"))
df_cafe = df_orders.select('cafe_name', 'lat', 'lng', 'city').dropDuplicates().orderBy('city')\
    .select('*', F.row_number().over(w).alias('id_cafe'))

# Из начального датафрейма удалим все колонки, связанные с кафе, заменив их на id_cafe)
df_orders = df_orders.join(df_cafe, ['cafe_name', 'lat', 'lng', 'city'], how='left')\
    .drop('cafe_name', 'lat', 'lng', 'city')

# Аналогично проделаем все то же самое с информацией о меню
df_meal = df_orders.select('menu_item', 'price').dropDuplicates().orderBy('menu_item')\
    .select('*', F.row_number().over(w).alias('id_meal'))

df_orders = df_orders.join(df_meal, ['menu_item', 'price'], how='left') .drop('menu_item', 'price')\
    .orderBy('id_order')

# Создадим отдельный датафрейм, который будет связан с блюдами, а остальные вещи, связанные с
# заказом поместим в orders
df_meals_in_order = df_orders.select('id_order', 'id_meal', 'count_meal').orderBy('id_order')
df_orders = df_orders.drop('id_meal', 'count_meal').dropDuplicates().orderBy('id_order')


df_meal.write.format('jdbc')\
    .mode('overwrite')\
    .option('url', 'jdbc:postgresql://localhost:5432/postgres')\
    .option("driver", "org.postgresql.Driver").option("dbtable", "meals") \
    .option("user", "postgresDB").option("password", "2008").save()

df_cafe.write.format('jdbc')\
    .mode('overwrite')\
    .option('url', 'jdbc:postgresql://localhost:5432/postgres')\
    .option("driver", "org.postgresql.Driver").option("dbtable", "cafes") \
    .option("user", "postgresDB").option("password", "2008").save()

df_orders.write.format('jdbc')\
    .mode('overwrite')\
    .option('url', 'jdbc:postgresql://localhost:5432/postgres')\
    .option("driver", "org.postgresql.Driver").option("dbtable", "orders")\
    .option("user", "postgresDB").option("password", "2008").save()

df_meals_in_order.write.format('jdbc')\
    .mode('overwrite')\
    .option('url', 'jdbc:postgresql://localhost:5432/postgres')\
    .option("driver", "org.postgresql.Driver").option("dbtable", "meals_in_order")\
    .option("user", "postgresDB").option("password", "2008").save()

