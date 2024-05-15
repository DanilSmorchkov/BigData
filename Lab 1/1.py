from datetime import datetime, timedelta
import random
import findspark
round_p = round
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import rand, round, udf, expr, row_number, lit, shuffle, count
from pyspark.sql.types import TimestampType, NullType
from pyspark.sql.window import Window
from pyspark.sql import functions as F


# Кастомная функция, которая возвращает случайные время работы кафе с 10:00 по 21:59
def random_date():
    hour = random.randint(10, 21)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_diff = (end_date - start_date).days
    random_days = random.randint(0, date_diff)
    return (start_date + timedelta(days=random_days)).replace(hour=hour, minute=minute, second=second)


# Создаем сессию Спарк
spark = SparkSession.builder.config("spark.driver.memory", "10g").master('local[*]').appName('lab1').getOrCreate()

# Задаем udf для нашей ранее определенной функции
random_date_udf = udf(random_date, TimestampType())

# Создаем каталог заказов. Сначала он содержит только номер заказа и время
df_orders = spark.range(0, 700000)\
    .withColumnRenamed('id', "id_order")\
    .withColumn('date_and_time', random_date_udf())\
    .orderBy('date_and_time')

# Далее добавим кафе в котором были совершены заказы с их местоположением
# Сначала создадим Датафрейм с информацией о кафе
df_cafe = spark.read.csv('./cafe.csv', header=True, inferSchema=True)
# Теперь размножим этот датафрейм
df_replicate_cafe = df_cafe.withColumn('city', expr('explode(array_repeat(City, 15000))'))
# Зашафлим датафрейм
df_replicate_cafe = df_replicate_cafe.withColumn("Random", rand()).orderBy("Random").drop('Random')
# Создадим дополнительный столбец id_order, по которому будет join кафе и заказы
w = Window().orderBy(lit('Moscow'))
df_replicate_cafe = df_replicate_cafe.withColumn('id_order', row_number().over(w)-1)

# Непосредственно join
df_orders = df_orders.join(other=df_replicate_cafe, how='left', on='id_order')
# Создаем отдельный столбец, в котором будет храниться количество блюд для каждого заказа
df_orders_with_num_meals = df_orders.withColumn('num_meals', round(rand()*4, 0) + 1)
# Множим наши строки в соответствии с колонкой num_meals
df_orders_with_num_meals = df_orders_with_num_meals.withColumn(
  "num_meals", expr("explode(array_repeat(num_meals, int(num_meals)))"))
# Сохраним эту колонку в отдельный датафрейм, он нам пригодится дальше (колонка, в которой подряд идут номера заказов,
# повторенные в количестве блюд для каждого заказа)
df_num_meals = df_orders_with_num_meals.select('id_order')
# Удалим этот столбец из основного датафрейма
df_orders = df_orders_with_num_meals.drop('num_meals')

# Далее работаем с меню
df_menu = spark.read.csv('./menu.csv', header=True, inferSchema=True)
# Точно так же множим строки и шафлим
df_replicate_menu = df_menu.withColumn('price', expr('explode(array_repeat(price, 70000))'))
df_replicate_menu = df_replicate_menu.withColumn("Random", rand()).orderBy("Random").drop('Random')

# Далее добавляем колонку с количеством блюд для каждого заказа, с этой целью создаем дополнительные столбцы в обоих
# датафреймах, а потом join их
w = Window().orderBy(lit('A'))
df_num_meals = df_num_meals.withColumn('id', row_number().over(w))
df_replicate_menu = df_replicate_menu.withColumn('id', row_number().over(w))\
    .join(df_num_meals, how='left', on=['id']).drop('id')

# Далее join заказов с меню, правда из-за такой реализации не может получиться так, что в один заказ попали два
# одинаковых блюда, хотя с какой-то стороны это логично - можно выделить отдельное столбец, в котором указывать
# количество блюд в одном заказе
df_orders = df_orders.join(other=df_replicate_menu, how='left', on='id_order').orderBy('id_order').dropDuplicates()
df_orders = df_orders.withColumn('rand', rand())
df_orders = df_orders.withColumn('count_meal', F.when(df_orders['rand'] > 0.97, 2)
                                 .otherwise(F.when(df_orders['rand'] > 0.99, 3)
                                 .otherwise(1)))
df_orders = df_orders.withColumn('rand', rand())
df_orders = df_orders.withColumn('city', F.when(df_orders['rand'] > 0.95, None).otherwise(F.col('city'))).drop('rand')
df_orders = df_orders.withColumn('rand', rand())
df_orders = df_orders.withColumn('price', F.when(df_orders['rand'] > 0.9, None).otherwise(F.col('price'))).drop('rand')


df_orders.write.option("charset", "UTF8").csv('orders', sep=",", header=True)
