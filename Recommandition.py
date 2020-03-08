
from pyspark import SparkContext
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import *

import os

from collections import OrderedDict

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StandardScaler

# Required to allow the file to be submitted and run using spark-submit instead
# of using pyspark interactively

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()



# load metadata
metadata = (
	spark.read.format("com.databricks.spark.csv")
	.option('header', 'True')
	.option("inferSchema", "True")
	.option('codec', 'gzip')
#	.schema(audio_dataset_schemas[audio_dataset_names[2]])
	.load('hdfs:///data/msd/main/summary/metadata.csv.gz')
	)

metadata.count()

metadata.dropDuplicates(['song_id']).count()	#998963


# load user_song play dataset, after cleaned.
triplets = spark.read.parquet('hdfs:///user/ych192/outputs/mds/triplets_not_mismatchet.parquet')
triplets.cache()

# triplets repartition
triplets.rdd.getNumPartitions()
triplets =  triplets.repartition(32)
triplets.cache()

# count lines, unique songs, unique users
triplets.count()	#45795111
triplets.dropDuplicates(['song_id']).count()	#378310

triplets.dropDuplicates(['user_id']).count()	#1019318

# dataframe of plays by user
triplets_by_user = (
    triplets
    .groupBy('user_id')
    .agg(F.sum('plays')
        )
    .orderBy('sum(plays)', ascending=False)
    )

triplets_by_user.show(5, truncate = False)

# +----------------------------------------+----------+
# |user_id                                 |sum(plays)|
# +----------------------------------------+----------+
# |093cb74eb3c517c5179ae24caf0ebec51b24d2a2|13074     |
# |119b7c88d58d0c6eb051365c103da5caf817bea6|9104      |
# |3fa44653315697f42410a30cb766a4eb102080bb|8025      |
# |a2679496cd0af9779a92a13ff7c6af5c81ea8c7b|6506      |
# |d7d2d888ae04d16e994d6964214a1de81392ee04|6190      |
# +----------------------------------------+----------+

# the user who has highest number of plays, listened to 195 songs
triplets.where(triplets.user_id == '093cb74eb3c517c5179ae24caf0ebec51b24d2a2').dropDuplicates(['song_id']).count()	#195 songs played

triplets_by_user_songs = (
	triplets
	.groupBy('user_id')
	.agg(F.count('song_id').alias('songs_per_user'))
	.orderBy('songs_per_user', ascending = False)
	)
triplets_by_user_songs.show(5)

# +--------------------+--------------+
# |             user_id|songs_per_user|
# +--------------------+--------------+
# |ec6dfcf19485cb011...|          4316|
# |8cb51abc6bf8ea293...|          1562|
# |5a3417a1955d91364...|          1557|
# |fef771ab021c20018...|          1545|
# |c1255748c06ee3f64...|          1498|
# +--------------------+--------------+

triplets_by_user_songs = (
	triplets
	.groupBy('user_id')
	.agg(F.countDistinct('song_id').alias('songs_per_user'))
	.orderBy('songs_per_user', ascending = False)
	)
triplets_by_user_songs.show(5)



triplets_by_user_songs_df = triplets_by_user_songs.toPandas()


triplets_by_song_plays = (
	triplets
	.groupBy('song_id')
	.agg(F.sum('plays').alias('song_total_plays'))
	.orderBy('song_total_plays', ascending = False)
	)
triplets_by_song_plays.show(5)

# +------------------+----------------+
# |           song_id|song_total_plays|
# +------------------+----------------+
# |SOBONKR12A58A7A7E0|          726885|
# |SOSXLTC12AF72A7F54|          527893|
# |SOEGIYH12A6D4FC0E3|          389880|
# |SOAXGDH12A8C13F8A1|          356533|
# |SONYKOW12AB01849C9|          292642|
# +------------------+----------------+



# Plot user plays distribution
triplets_by_user_df = triplets_by_user.toPandas()
plot, ax = plt.subplots(dpi=300, figsize= (20, 3 ))

ax.hist(triplets_by_user_df['sum(plays)'], bins = 1000)

plt.title("Distribution of Users by total Plays")
plt.ylabel("Number of Users")
plt.tight_layout()  # reduce whitespace
plot.savefig('plots/triplets_by_user_df.png', bbox_inches="tight")  # save as png and view in windows
plt.close(plot)


# Plot user plays subset (< 600) distribution
triplets_by_user_df_subset = triplets_by_user_df[triplets_by_user_df['sum(plays)'] < 600]

plot, ax = plt.subplots(dpi=300, figsize= (10, 3 ))
ax.hist(triplets_by_user_df_subset['sum(plays)'], bins = 300)
plt.title("Distribution of Users by total Plays")
plt.ylabel("Number of Users")
plt.tight_layout()  # reduce whitespace
plot.savefig('plots/triplets_by_user_df_subset.png', bbox_inches="tight")  # save as png and view in windows
plt.close(plot)


# Plot song plays  distribution
triplets_by_song_plays_df = triplets_by_song_plays.toPandas()

plot, ax = plt.subplots(dpi=300, figsize= (10, 3 ))
ax.hist(triplets_by_song_plays_df['song_total_plays'], bins = 1000)
plt.title("Distribution of Songs by total Played times")
plt.ylabel("Number of Songs")
plt.xlabel("Song played times")
plt.tight_layout()  # reduce whitespace
plot.savefig('plots/triplets_by_song_plays_df.png', bbox_inches="tight")  # save as png and view in windows
plt.close(plot)


# Plot song plays subset (< 200) distribution
triplets_by_song_plays_df_subset = triplets_by_song_plays_df[triplets_by_song_plays_df.song_total_plays < 400]
plot, ax = plt.subplots(dpi=300, figsize= (10, 3 ))
ax.hist(triplets_by_song_plays_df_subset['song_total_plays'], bins = 400)
plt.title("Distribution of Songs by total Played times")
plt.ylabel("Number of Songs")
plt.xlabel("Song played times")
plt.tight_layout()  # reduce whitespace
plot.savefig('plots/triplets_by_song_plays_df_subset1.png', bbox_inches="tight")  # save as png and view in windows
plt.close(plot)




# user play numbers of songs
plot, ax = plt.subplots(dpi=300, figsize= (10, 3 ))
ax.hist(triplets_by_user_songs_df['songs_per_user'], bins = 1000)
plt.title("Distribution of User by Unique Songs played")
plt.ylabel("Number of Users")
plt.tight_layout()  # reduce whitespace
plot.savefig('plots/triplets_by_user_songs_df.png', bbox_inches="tight")  # save as png and view in windows
plt.close(plot)
# subset
triplets_by_user_songs_df_subset = triplets_by_user_songs_df[triplets_by_user_songs_df.songs_per_user < 200]
plot, ax = plt.subplots(dpi=300, figsize= (10, 3 ))
ax.hist(triplets_by_user_songs_df_subset['songs_per_user'], bins = 200)
plt.title("Distribution of User by number of Unique Songs played")
plt.ylabel("Number of Users")
plt.xlabel("Unique songs played")
plt.tight_layout()  # reduce whitespace
plot.savefig('plots/triplets_by_user_songs_df_subset.png', bbox_inches="tight")  # save as png and view in windows
plt.close(plot)


# user sum plays
triplets_by_user_df.quantile(0.01)		# sum(plays) = 10
triplets_by_user_df.quantile(0.05)		# sum(plays) = 14
triplets_by_user_df.quantile(0.4)		# sum(plays) = 51

#song_id total plays
triplets_by_song_plays_df.quantile(0.01)		#sum(plays) = 1
triplets_by_song_plays_df.quantile(0.05)		#sum(plays) = 2
triplets_by_song_plays_df.quantile(0.5)			#sum(plays) = 31
triplets_by_song_plays_df.quantile(0.7)			#93
# user_id, songs per user
triplets_by_user_songs_df.quantile(0.01)		# songs_per_user = 9
triplets_by_user_songs_df.quantile(0.2)		# songs_per_user = 13, quantile = 0.1, song = 11
triplets_by_user_songs_df.quantile(0.4)			# songs_per_user = 20
triplets_by_user_songs_df.quantile(0.5)			# 26


# 1% of users played 9 songs, 5% users played 10 songs, and 10% of users played 11 songs, take M = 11 to 
# filter out < 10% users listen to less than 11 songs. Among 378310 songs, about 5% songs have been played 
# 1 or less times. More than 20% songs played less than 7 times, imagine if it is a popular song, it would be 
# listened hundreds of times. I would say if a song is played less than dozens of times, the song should be 
# considerred for very special interested groups. For 50% of songs were played less than N = 32 times, for 
# computational convinence, here decide to keep songs played more than 32 times, then we have half numerber of songs
# that should serve majority users. 50% of users played less than 27 songs, that means we still have lots of
# songs to recommand. Even more agressively, we can increase N to a even larger number to ease computation resouse.
# Set M = 11, N = 32

# M = 26 , N = 94 


# get minor songs and minor users
triplets_minor_songs = triplets_by_song_plays.where(triplets_by_song_plays.song_total_plays < 94)		#count = 264993

# +------------------+----------------+
# |           song_id|song_total_plays|
# +------------------+----------------+
# |SOGYCIG12AB018085A|              93|
# |SOXNJCP12A6D4F95D4|              93|
# |SOROQQP12AB01846DA|              93|
# |SOCBXTJ12A8C139E40|              93|
# |SOPOKJC12AB017F4DC|              93|
# +------------------+----------------+


triplets_minor_users = triplets_by_user_songs.where(triplets_by_user_songs.songs_per_user < 26)		#count = 504547

# +--------------------+--------------+
# |             user_id|songs_per_user|
# +--------------------+--------------+
# |53764d30b36edea48...|            25|
# |c53742e2838a3330a...|            25|
# |9cee5136ce22e1421...|            25|
# |02c02ddd4f428a3fd...|            25|
# |21d5a0fa2cac87f71...|            25|
# +--------------------+--------------+




# clean triplets
triplets_clean_minorsongs = triplets.join(triplets_minor_songs, on= 'song_id', how ='left_anti')
triplets_clean = triplets_clean_minorsongs.join(triplets_minor_users, on= 'user_id', how= 'left_anti')

triplets_clean.cache()

triplets_clean.show(5)
# +--------------------+------------------+-----+
# |             user_id|           song_id|plays|
# +--------------------+------------------+-----+
# |00007ed2509128dcd...|SOPFSPZ12A58A78AFB|    1|
# |00007ed2509128dcd...|SORPSOF12AB0188C39|    2|
# |00007ed2509128dcd...|SOAYETG12A67ADA751|    2|
# |00007ed2509128dcd...|SODTGOI12A8C13EBE8|    4|
# |00007ed2509128dcd...|SOHBYIJ12A6D4FB344|    1|
# +--------------------+------------------+-----+

triplets_clean.count()		#35503170

#triplets_clean.write.parquet('hdfs:///user/ych192/outputs/mds/triplets_clean.parquet')



# check the cleaning result
test_clean = (
	triplets_clean
	.groupBy('user_id')
	.agg(F.count('song_id').alias('songs_per_user'))
	.orderBy('songs_per_user', ascending = False)
	)
test_clean.show(5)
# +----------------------------------------+--------------+
# |user_id                                 |songs_per_user|
# +----------------------------------------+--------------+
# |ec6dfcf19485cb011e0b22637075037aae34cf26|2917          |
# |8cb51abc6bf8ea29341cb070fe1e1af5e4c3ffcc|1481          |
# |fef771ab021c200187a419f5e55311390f850a50|1427          |
# |c1255748c06ee3f6440c51c439446886c7807095|1397          |
# |4e73d9e058d2b1f2dba9c1fe4a8f416f9f58364f|1394          |
# +----------------------------------------+--------------+


test_clean.orderBy('songs_per_user', ascending = True).show(5)
triplets_clean.dropDuplicates(['song_id']).count()		#113314

# +--------------------+------------------+-----+
# |             user_id|           song_id|plays|
# +--------------------+------------------+-----+
# |00007ed2509128dcd...|SOPFSPZ12A58A78AFB|    1|
# |00007ed2509128dcd...|SORPSOF12AB0188C39|    2|
# |00007ed2509128dcd...|SOAYETG12A67ADA751|    2|
# |00007ed2509128dcd...|SODTGOI12A8C13EBE8|    4|
# |00007ed2509128dcd...|SOHBYIJ12A6D4FB344|    1|
# +--------------------+------------------+-----+


user_song = spark.read.parquet('hdfs:///user/ych192/outputs/mds/triplets_clean.parquet')


triplets_clean.schema['user_id'].dataType

user_song = triplets_clean

user_song.cache()


songs_by_user = (
	user_song
	.groupBy('user_id')
	.agg(F.count('song_id').alias('songs_per_user'))
	.orderBy('songs_per_user', ascending = True	)
	)
songs_by_user.show()		#count = 939471


minor_users = songs_by_user.where(songs_by_user.songs_per_user < 6)			#count = 54
user_song_clean = user_song.join(minor_users, on = 'user_id', how = 'left_anti')
user_song_clean.cache()		#count = 35502976



user_stringIdx = StringIndexer(inputCol = "user_id", outputCol = "user_index")
song_stringIdx = StringIndexer(inputCol	= 'song_id', outputCol = 'song_index')


pipeline = Pipeline(stages=[user_stringIdx, song_stringIdx])
pipelineFit = pipeline.fit(user_song_clean)
songs_by_user_df = pipelineFit.transform(user_song_clean)


# songs_by_user_df.show()
# +--------------------+------------------+-----+----------+----------+
# |             user_id|           song_id|plays|user_index|song_index|
# +--------------------+------------------+-----+----------+----------+
# |00007ed2509128dcd...|SOPFSPZ12A58A78AFB|    1|  229367.0|   73414.0|
# |00007ed2509128dcd...|SORPSOF12AB0188C39|    2|  229367.0|   48774.0|
# |00007ed2509128dcd...|SOAYETG12A67ADA751|    2|  229367.0|     594.0|
# |00007ed2509128dcd...|SODTGOI12A8C13EBE8|    4|  229367.0|   10227.0|

# train test split
unique_users = user_song_clean.select(F.col('user_id')).distinct().rdd.map(lambda r: r[0]).collect()
fraction_dict = {key:0.21 for key in unique_users }

# get test 
test = songs_by_user_df.sampleBy("user_id", fractions=fraction_dict)		#count = 7458091



train = songs_by_user_df.join(test, on = ((songs_by_user_df.user_id == test.user_id) & (songs_by_user_df.song_id == test.song_id)),\
 how = 'left_anti') 
train.count() 		#28044885

# implicitPrefs = True !!! Rand default = 10, Rank is number of latent factor, increase rank may increase accuracy
als = ALS(maxIter=5, regParam=0.01, userCol="user_index", itemCol="song_index", ratingCol="plays", implicitPrefs=True)
alsModel = als.fit(train)

predictions = alsModel.transform(test)

als.getImplicitPrefs()

predictions.cache()
predictions.show()
# +--------------------+------------------+-----+----------+----------+-----------+
# |             user_id|           song_id|plays|user_index|song_index| prediction|
# +--------------------+------------------+-----+----------+----------+-----------+
# |96f7b4f800cafef33...|SOPUCYA12A8C13A694|    5|       5.0|      12.0| 0.68972623|
# |bc29b378a4751d9b6...|SOPUCYA12A8C13A694|    5|     104.0|      12.0| 0.27929363|
# |fc1d4c6a893c74887...|SOPUCYA12A8C13A694|    1|     167.0|      12.0|  0.7614142|
# |ec46c2eeedfb3b69e...|SOPUCYA12A8C13A694|    2|     454.0|      12.0|  0.5556509|
# |8e9f5c871f48fcb16...|SOPUCYA12A8C13A694|    1|     821.0|      12.0|  0.4998854|
# |aa4671dd2ccc91a98...|SOPUCYA12A8C13A694|    1|     943.0|      12.0| 0.62119555|



evaluator = RegressionEvaluator(metricName="rmse", labelCol="plays", predictionCol="prediction")
evaluator.evaluate(predictions.filter(F.col('prediction') != np.NaN))		#implicitPrefs = False 10.928, implicit = True 6.53

predictions.filter(F.col('prediction') == np.NaN).count()		# 3

predictions.write.parquet('hdfs:///user/ych192/outputs/mds/recommand_test_predictions.parquet')

predictions = spark.read.parquet('hdfs:///user/ych192/outputs/mds/recommand_test_predictions.parquet').cache()

song_recommand = alsModel.recommendForAllItems(10)


# get a sample users to test predictions
test_users = (
	test
	.groupBy('user_id')
	.agg(F.count('song_id').alias('song_per_user'))
	.orderBy('song_per_user')
	)
test_users_30more_songs = test_users.where(test_users.song_per_user > 30)

sample_users = test_users_30more_songs.orderBy(F.rand()).limit(10)
sample_users.show(10, False)
sample_users.cache()

# +----------------------------------------+-------------+
# |user_id                                 |song_per_user|
# +----------------------------------------+-------------+
# |bf6d77cf89e81c267a2cb6e665eab52cd73655cf|38           |
# |dacdb0dd725c4d02cefcc3da417d6267790a50e5|39           |
# |68c421d12a3faad46611b2e71c35f6a9185aa17a|46           |
# |8da6d0d6fcc9c0790036f157224488d85f8e522d|38           |
# |0bae34ce936dbfc0a0c69a78e39ebf6143431011|43           |
# |fb104e750131fdc26397a7d962ed270675e82021|44           |
# |28cd9b47925ce7eb2588ada43ccabae00afc8575|34           |
# |0c9b3ed7e8ae91f6013a49bf97ac3c31224093f8|35           |
# |7f9926173a42c625e3b0c7d6725f12bfb0f1b391|38           |
# |5a8eb7fc8db0de524e26e8dee616db7d7e9a7a0b|56           |
# +----------------------------------------+-------------+



sample_users_df = test.join(sample_users, on = 'user_id', how = 'inner')
sample_users_df.cache()
sample_users_df.show()


sample_predictions = alsModel.transform(sample_users_df.select(F.col('user_index'), F.col('song_index')))
sample_predictions.cache()
sample_predictions.show()

#song_recommand = alsModel.recommendForAllItems(10)
# get recommand for 10 users
song_recommand = alsModel.recommendForUserSubset(sample_users_df.select(F.col('user_index')), 10)
song_recommand.show()
song_recommand.cache()

# +----------+--------------------+
# |user_index|     recommendations|
# +----------+--------------------+
# |     14872|[[58, 0.2514357],...|
# |     32351|[[58, 0.17484158]...|
# |     27733|[[37, 0.61185944]...|
# |     21923|[[134, 0.3560784]...|
# |      8974|[[11, 0.7558242],...|
# |     36056|[[319, 0.19679566...|
# |     25675|[[3, 0.8134617], ...|
# |     38795|[[32, 0.47223717]...|
# |     21633|[[17, 0.2100571],...|
# |     28728|[[17, 0.19925675]...|
# +----------+--------------------+



song_recommand.write.parquet('hdfs:///user/ych192/outputs/mds/song_recommand_top10_1.parquet')

song_recommand = spark.read.parquet('hdfs:///user/ych192/outputs/mds/song_recommand_top10_1.parquet')
song_recommand.cache()
# user index 352996 have listend to following musics
test.where(test.user_index=='352996').orderBy('plays', ascending = False).show(10, False)

# +----------------------------------------+------------------+-----+----------+----------+
# |user_id                                 |song_id           |plays|user_index|song_index|
# +----------------------------------------+------------------+-----+----------+----------+
# |f5f45a5a2a872d93b394cad7f6a2ae9a53613aea|SOTWNDJ12A8C143984|12   |352996.0  |13.0      |
# |f5f45a5a2a872d93b394cad7f6a2ae9a53613aea|SOKTIFI12AB01804AF|6    |352996.0  |19611.0   |
# |f5f45a5a2a872d93b394cad7f6a2ae9a53613aea|SOSROSD12A8C13F519|4    |352996.0  |2989.0    |
# |f5f45a5a2a872d93b394cad7f6a2ae9a53613aea|SOYFRVX12AAF3B3F3E|4    |352996.0  |2995.0    |
# |f5f45a5a2a872d93b394cad7f6a2ae9a53613aea|SOVDTMK12AB01829D3|3    |352996.0  |2058.0    |
# |f5f45a5a2a872d93b394cad7f6a2ae9a53613aea|SOMPTCI12AB017C416|1    |352996.0  |758.0     |
# |f5f45a5a2a872d93b394cad7f6a2ae9a53613aea|SOLRGVL12A8C143BC3|1    |352996.0  |18.0      |
# |f5f45a5a2a872d93b394cad7f6a2ae9a53613aea|SOWWDEF12A8C143C8A|1    |352996.0  |10984.0   |
# |f5f45a5a2a872d93b394cad7f6a2ae9a53613aea|SOIZLKI12A6D4F7B61|1    |352996.0  |38.0      |
# |f5f45a5a2a872d93b394cad7f6a2ae9a53613aea|SODGVGW12AC9075A8D|1    |352996.0  |28.0      |
# +----------------------------------------+------------------+-----+----------+----------+



test.where(test.song_index == '1').show()

song_recommand.collect()[1]



song_recommand_expand  = song_recommand.withColumn('reco1', F.explode(song_recommand.recommendations)).drop('recommendations')
song_recommand_expand = song_recommand_expand.rdd.flatMap(lambda x: [(x[0], x[1][0], x[1][1])]).toDF(['user_index', 'song_index', 'rating'])

# +----------+----------+-------------------+
# |user_index|song_index|             rating|
# +----------+----------+-------------------+
# |      8974|        11| 0.7558242082595825|
# |      8974|        14| 0.7381279468536377|
# |      8974|         9| 0.7355676889419556|
# |      8974|        19| 0.6412095427513123|
# |      8974|        31| 0.6222537755966187|
# |      8974|         4| 0.5968846082687378|
# |      8974|        17| 0.5871473550796509|
# |      8974|        40| 0.5847553014755249|
# |      8974|        25| 0.5695775747299194|
# |      8974|         6| 0.5489161014556885|
# |     36056|       319|0.19679565727710724|
# |     36056|       134|0.16721069812774658|
# |     36056|       156| 0.1627701222896576|
# |     36056|       310|0.15916773676872253|

test.rdd.getNumPartitions()
test.rdd.repartition(32)

sample_recommand = song_recommand_expand.join(
	test
	.select(F.col('user_id'), F.col('user_index')).dropDuplicates(),
	on = 'user_index',
	how = 'left'
	)

sample_recommand_1 = sample_recommand.join(
	test
	.select(F.col('song_id'), F.col('song_index')).dropDuplicates(),
	on = 'song_index',
	how = 'left'
	)

sample_recommand_1.show()

# +----------+----------+-------------------+--------------------+------------------+
# |song_index|user_index|             rating|             user_id|           song_id|
# +----------+----------+-------------------+--------------------+------------------+
# |         4|      8974| 0.5968846082687378|9a5974ca42c64fe82...|SOLFXKT12AB017E3E0|
# |        17|     38795|0.41760435700416565|0c9b3ed7e8ae91f60...|SOPQLBY12A6310E992|
# |        17|     21633| 0.2100570946931839|0bae34ce936dbfc0a...|SOPQLBY12A6310E992|
# |        17|      8974| 0.5871473550796509|9a5974ca42c64fe82...|SOPQLBY12A6310E992|
# |        17|     28728| 0.1992567479610443|da2d257636d8c0117...|SOPQLBY12A6310E992|
# |        17|     36056|0.14134515821933746|591e239cfa8510494...|SOPQLBY12A6310E992|
# |        17|     32351| 0.1473134309053421|dc2de60738c20210d...|SOPQLBY12A6310E992|
# |        20|     25675| 0.6575251817703247|176c70dd1b2917ded...|SOSCIZP12AB0181D2F|
# |        96|     21633|0.15805037319660187|0bae34ce936dbfc0a...|SOMGIYR12AB0187973|
# |       191|     28728| 0.1726546436548233|da2d257636d8c0117...|SOAMIQK12A6701D94F|
# |       191|     36056|0.14024236798286438|591e239cfa8510494...|SOAMIQK12A6701D94F|
# |       238|     27733|0.42235255241394043|aebb22d8e36f24a5b...|SOLARJV12AB018306B|
# |         6|     25675| 0.6128984093666077|176c70dd1b2917ded...|SOWCKVR12A8C142411|
# |         6|      8974| 0.5489161014556885|9a5974ca42c64fe82...|SOWCKVR12A8C142411|
# |        12|     27733| 0.5509281158447266|aebb22d8e36f24a5b...|SOPUCYA12A8C13A694|
# |        58|     14872|0.25143569707870483|68c421d12a3faad46...|SOBUBLL12A58A795A8|
# |        58|     32351|0.17484158277511597|dc2de60738c20210d...|SOBUBLL12A58A795A8|
# |       122|     21923|0.22708842158317566|a7b35c5f5bbc7ee96...|SOWYYUQ12A6701D68D|
# |       145|     14872| 0.1825857013463974|68c421d12a3faad46...|SOEOJHS12AB017F3DC|
# |       145|     32351| 0.1623658537864685|dc2de60738c20210d...|SOEOJHS12AB017F3DC|
# +----------+----------+-------------------+--------------------+------------------+


sample_recommand_1.write.parquet('hdfs:///user/ych192/outputs/mds/sample_recommand.parquet')
sample_recommand = spark.read.parquet('hdfs:///user/ych192/outputs/mds/sample_recommand.parquet').cache()

user_1 = sample_recommand.where(sample_recommand.user_index == 38795)
# +----------+----------+-------------------+--------------------+------------------+
# |song_index|user_index|             rating|             user_id|           song_id|
# +----------+----------+-------------------+--------------------+------------------+
# |        17|     38795|0.41760435700416565|0c9b3ed7e8ae91f60...|SOPQLBY12A6310E992|
# |       156|     38795| 0.4302408695220947|0c9b3ed7e8ae91f60...|SONQEYS12AF72AABC9|
# |       199|     38795| 0.4245038330554962|0c9b3ed7e8ae91f60...|SOVQJRY12A81C210C0|
# |        41|     38795|0.44869598746299744|0c9b3ed7e8ae91f60...|SOLLNTU12A6701CFDC|
# |       269|     38795|0.39239516854286194|0c9b3ed7e8ae91f60...|SOUFWEW12AB0180EB7|
# |        32|     38795|0.47223716974258423|0c9b3ed7e8ae91f60...|SOGPBAW12A6D4F9F22|
# |       101|     38795| 0.4006105065345764|0c9b3ed7e8ae91f60...|SOKMHKY12AF72AB079|
# |        66|     38795|0.45880722999572754|0c9b3ed7e8ae91f60...|SOQGVCS12AF72A078D|
# |        91|     38795| 0.4423770606517792|0c9b3ed7e8ae91f60...|SOSJRJP12A6D4F826F|
# |        60|     38795|0.41787484288215637|0c9b3ed7e8ae91f60...|SOEBOWM12AB017F279|

useful_meta = metadata.select(F.col('song_id'), F.col('artist_name'), F.col('song_hotttnesss'),\
 F.col('title'),F.col('genre'), F.col('release'))
user_1_songs = user_1.join(useful_meta, on= 'song_id', how = 'inner').select(F.col('rating'), \
	F.col('artist_name'), F.col('song_hotttnesss'), F.col('title'), F.col('release'))

# +--------------+------------------+------------------+-----+--------------------+
# |   artist_name|   song_hotttnesss|             title|genre|             release|
# +--------------+------------------+------------------+-----+--------------------+
# |  3 Doors Down|0.9151810636939853|        Kryptonite| null|           Total 90s|
# | Guns N' Roses|              null|     Paradise City| null|       Greatest Hits|
# |Counting Crows|               1.0|         Mr. Jones| null|Films About Ghost...|
# |     Asia 2001|0.2998774882739778|          Epilogue| null|             Amnesia|
# |      Bon Jovi|               1.0|Livin' On A Prayer| null|          Cross Road|
# |Lynyrd Skynyrd|              null|Sweet home Alabama| null|          To Die For|
# |     Metallica|0.9655075340177467| Master Of Puppets| null|   Master Of Puppets|
# |     Radiohead|              null|  Creep (Explicit)| null|         Pablo Honey|
# |        Eagles|0.9406301589603857|  Hotel California| null|Hotel California ...|
# |    Nickelback|              null| How You Remind Me| null|FETENHITS - New P...|
# +--------------+------------------+------------------+-----+--------------------+


test_user1 = test.where(test.user_index == 38795).join(useful_meta, on = 'song_id', how= 'inner').orderBy('artist_name')\
.select(F.col('plays'), F.col('artist_name'), F.col('song_hotttnesss'), F.col('title'), F.col('release'))
test_user1.orderBy('plays', ascending= False).show(50)
# +-----+--------------------+-------------------+--------------------+--------------------+
# |plays|         artist_name|    song_hotttnesss|               title|             release|
# +-----+--------------------+-------------------+--------------------+--------------------+
# |   20|      C.V. Jørgensen|               null|Pak Dit Grej (199...|         De 2 Første|
# |   10|        Billy Squier|               null|          The Stroke|Let's Go To Priso...|
# |   10|        Billy Squier| 0.7646204000035967|         In The Dark|        Don't Say No|
# |    8|          Nickelback|  0.936575245043614|Never Gonna Be Al...|          Dark Horse|
# |    5|        Damn Yankees| 0.8158882089961863|High Enough (Albu...|Rhino Hi-Five: Da...|
# |    4|       Robert Palmer|0.26586104921065007|Your Mother Shoul...|The Essential Sel...|
# |    4|            Bon Jovi|                1.0|You Give Love A B...|Slippery When Wet...|
# |    4|         Miley Cyrus|                1.0| Party In The U.S.A.| Party In The U.S.A.|
# |    4|            Bon Jovi| 0.8709776265271216|        It's My Life|               Crush|
# |    4|           Metallica|                1.0|       Enter Sandman|           Metallica|
# |    4|    Billy Currington| 0.7268112207729489|That's How Countr...|Little Bit Of Eve...|
# |    4|           Colt Ford| 0.4368224699038839|   Huntin' The World|Country Is As Cou...|
# |    3|          Nickelback| 0.7830859657398406|         Never Again|      Silver Side Up|
# |    3|          Toby Keith| 0.5944639098781762|      She's A Hottie|Toby Keith 35 Big...|
# |    3|           Colt Ford|0.46417399167415674|             Like Me|Ride Through The ...|
# |    3|           Colt Ford| 0.3972653548644609|             Twisted|Ride Through The ...|
# |    2|      Britney Spears|                1.0|               Toxic|The Singles Colle...|
# |    2|           Liz Phair|  0.519832962528063|Why Can't I? (Exp...|           Liz Phair|
# |    2|        3 Doors Down| 0.8175613303456302|    Let Me Be Myself|        3 Doors Down|
# |    2|             Alabama| 0.6448037051731974|My Home's In Alabama|        Alabama Live|
# |    2|           Colt Ford| 0.4191634755594802|            Buck 'Em|Country Is As Cou...|
# |    2|                 Kix|               null|Don't Close Your ...|                 Kix|
# |    1|            Kid Rock| 0.6072560951352716|So Hott (Amended ...|   Rock N Roll Jesus|
# |    1|       Joe Christmas| 0.2998774882739778|       Bedroom Suite|Upstairs Overlooking|
# |    1|         Miley Cyrus| 0.7865534054973415|Kicking And Screa...|The Time Of Our L...|
# |    1|Usher Featuring L...| 0.7813811676365547|               Yeah!|         Confessions|
# |    1|     The Wallflowers| 0.8696612354539689|       One Headlight|Bringing Down The...|
# |    1|             Journey| 0.5456765460250699|If He Should Brea...|       Trial By Fire|
# |    1|         Linkin Park| 0.8722290226088538|Crawling (Album V...|       Hybrid Theory|
# +-----+--------------------+-------------------+--------------------+--------------------+


train_user1 = train.where(train.user_index == 38795).join(useful_meta, on = 'song_id', how= 'inner').orderBy('artist_name')\
	.select(F.col('plays'), F.col('artist_name'), F.col('song_hotttnesss'), F.col('title'), F.col('release'))
train_user1.show(50)


user_2 = sample_recommand.where(sample_recommand.user_index == 14872)
user_2_songs = user_2.join(useful_meta, on= 'song_id', how = 'inner').select(F.col('song_id'),F.col('rating'), \
	F.col('artist_name'), F.col('song_hotttnesss'), F.col('title'), F.col('release'))
user_2_songs.show()

# +------------------+-------------------+--------------------+-------------------+--------------------+--------------------+
# |           song_id|             rating|         artist_name|    song_hotttnesss|               title|             release|
# +------------------+-------------------+--------------------+-------------------+--------------------+--------------------+
# |SOOXJDU12A8AE47ECB|  0.160763680934906|the bird and the bee|  0.756813842652999|       Again & Again|the bird and the bee|
# |SOAKMDU12A8C1346A9| 0.1528574377298355|  The Postal Service| 0.9041696612136142|  Such Great Heights|Grey's Anatomy Or...|
# |SOUGCDK12AC95F075F|0.15160781145095825|       Justin Bieber| 0.8589194788293776|    Never Let You Go|           My Worlds|
# |SOEOJHS12AB017F3DC| 0.1825857013463974|              Metric|               null|      Help I'm Alive|           Fantasies|
# |SOFCPOU12A8C13BF40| 0.1715606451034546|      Postal Service| 0.8161552405876549|Nothing Better (A...|             Give Up|
# |SODLAPJ12A8C142002|0.22106920182704926|      Emmy The Great|  0.685878801899692|                 Mia|          First Love|
# |SORAHAG12AB0182BD0|0.18231359124183655|             Soltero|0.48796141728262293| Songs Of The Season|          Hell Train|
# |SOBUBLL12A58A795A8|0.25143569707870483|         Tiny Vipers| 0.7513734115085566|They Might Follow...|         Tiny Vipers|
# |SOUDLVN12AAFF43658|0.15573202073574066|        Bill Withers|   0.55449365448418|Make Love To Your...|Playlist: The Ver...|
# |SOTGHQR12A8C1406C5|0.15378372371196747|      Chris Bathgate| 0.6331196341746719|                Coda|    A Cork Tale Wake|
# +------------------+-------------------+--------------------+-------------------+--------------------+--------------------+


test_user2 = test.where(test.user_index == 14872).join(useful_meta, on = 'song_id', how= 'inner').orderBy('artist_name')\
.select(F.col('song_id'),F.col('plays'), F.col('artist_name'), F.col('song_hotttnesss'), F.col('title'), F.col('release'))
test_user2.orderBy('plays', ascending= False).show(20)

# +------------------+-----+-------------------+------------------+--------------------+--------------------+
# |           song_id|plays|        artist_name|   song_hotttnesss|               title|             release|
# +------------------+-----+-------------------+------------------+--------------------+--------------------+
# |SOXJLLK12A8C139E3D|    2|              Foals|0.7920759819012835|Balloons (Single ...|            Balloons|
# |SODLAPJ12A8C142002|    2|     Emmy The Great| 0.685878801899692|                 Mia|          First Love|
# |SODTXJH12A8C134DD1|    2| Immortal Technique|0.5082891059677606|    Blackout Special|        Loose Canons|
# |SOKTHOE12AB01882F0|    2|              Kelis|0.8131284711485357|            Acapella|            Acapella|
# |SOAEKQS12A67AE0287|    1| Bedouin Soundclash|0.5957540542599369|            Criminal|    Sounding Amosaic|
# |SOOWMTD12AC468AB21|    1|      Faith No More|0.7098479113671987|      Ashes To Ashes|This Is It: The B...|
# |SOGWMDT12A6701F5C5|    1|        Cat Stevens|0.4634896622372766|         Whistlestar|             Numbers|
# |SORYRJI12A8C132747|    1|       Keyshia Cole|0.6247996943281579|         Work It Out|       Just Like You|
# |SOVWMOI12A67021386|    1|     Brian McKnight|0.6516648558948737|Show Me The Way B...|             Anytime|
# |SOAHPIG12A6D4F8F7F|    1|Kelis Featuring Nas|0.5920543875413302|        Blindfold Me|      Kelis Was Here|
# |SOGIDSA12A8C142829|    1|      Kings Of Leon|0.9051328362284463|              Camaro|Because Of The Times|
# |SOFZYKR12A6D4F72A5|    1|       Eric Clapton|0.5466354885721183|     Floating Bridge|      Another Ticket|
# |SOTGHQR12A8C1406C5|    1|     Chris Bathgate|0.6331196341746719|                Coda|    A Cork Tale Wake|
# |SOTLDCX12AAF3B1356|    1|           Dru Hill|0.7604743397513587|We're Not Making ...|       Master Peaces|
# |SOBJOSC12A8C137A74|    1|      Amy Winehouse|0.6450385130033194|               Cupid| Valentine's Day OST|
# |SOVFATL12AB0180F91|    1|          Busdriver|0.5930956904665208|        Reheated Pop|Fear Of A Black T...|
# |SOBGDOL12A6D4F6F21|    1|       Mariah Carey|   0.6580875659085|Don't Forget Abou...|Don't Forget Abou...|
# |SOMJLKL12AF72A4508|    1|              Cream|0.9186186661849578|         I'm So Glad|The Very Best Of ...|
# |SORJBEI12A8C14396D|    1|         Air France|0.7218916593024672|       June Evenings|         No Way Down|
# |SOLZKLE12AF729F385|    1|         Rilo Kiley|0.8447114518669253|   Spectacular Views|The Execution Of ...|
# +------------------+-----+-------------------+------------------+--------------------+--------------------+




# (c)
predictions.rdd.getNumPartitions()

predictions.rdd.repartition(32)


from pyspark.sql import *

from pyspark.mllib.evaluation import RankingMetrics

windowSpec = Window.partitionBy('user_id').orderBy(F.col('prediction').desc())

test_prediction = (
	predictions
	.select('user_id','song_id','prediction',F.rank().over(windowSpec).alias('rank'))
	.where('rank <= 10')
	.groupBy('user_id')
	.agg(F.expr('collect_list(song_id) as songs'))
	)

windowSpec = Window.partitionBy('user_id').orderBy(F.col('plays').desc())
test_actualLabel = (
	predictions 
	.select('user_id','song_id','plays',F.rank().over(windowSpec).alias('rank'))
	.where('rank <=10')
	.groupBy('user_id')
	.agg(F.expr('collect_list(song_id) as songs'))
	)


user_plays_rdd = test_prediction.join(F.broadcast(test_actualLabel),'user_id','inner').rdd.map(lambda x: (x[1],x[2]))
test_metrics = RankingMetrics(user_plays_rdd)

print(test_metrics.precisionAt(5))	#0.8889504523945604
print(test_metrics.ndcgAt(10))		#0.9171596854566892
print(test_metrics.meanAveragePrecision)	#0.7691694702138988


# [(['SOVMWUC12A8C13750B',
#    'SOAYETG12A67ADA751',
#    'SOKHEEY12A8C1418FE',
#    'SOOKJWB12A6D4FD4F8',
#    'SOTOAAN12AB0185C68',
#    'SOYZNPE12A58A79CAD',
#    'SOROLRE12A8C13943A',
#    'SOGEWRX12AB0189432',
#    'SORREOD12AB0189427',
#    'SOGRNDU12A3F1EB51F'],
#   ['SOTOAAN12AB0185C68',
#    'SOAYETG12A67ADA751',
#    'SOGRNDU12A3F1EB51F',
#    'SOKHEEY12A8C1418FE',
#    'SOGEWRX12AB0189432',
#    'SOYZNPE12A58A79CAD',
#    'SOVMWUC12A8C13750B',
#    'SOOKJWB12A6D4FD4F8',
#    'SOROLRE12A8C13943A',
#    'SORREOD12AB0189427']),
#  (['SOGDDKR12A6701E8FA',
#    'SOVXUCJ12A6701FBC2',
#    'SOYCXBA12A6701E35B',
#    'SOYRVSP12A6D4F907A',
#    'SOUHBAC12AB01816FE',
#    'SOGDTQS12A6310D7D1',
#    'SORYCTJ12A8C138AC3',
#    'SOXSQCC12A8C13D78E',
#    'SOMFLEZ12A6D4F907C',
#    'SOUBUWB12A6701E9F0'],
#   ['SOSOPQD12A6D4F8D32',
#    'SORJGVR12A58A7C3C8',
#    'SODVYCM12A6D4F7F2A',
#    'SOUBUWB12A6701E9F0',
#    'SOSPIXE12AB018B422',
#    'SOZPGLO12A8C139A1E',
#    'SOUHBAC12AB01816FE',
#    'SOMDZQT12A58A7DC91',
#    'SORYCTJ12A8C138AC3',
#    'SOEHDIY12A58A78EE6',
#    'SOTXAPB12A6310E092',
#    'SOVXUCJ12A6701FBC2',
#    'SOYOFJE12A58A77A23',
#    'SOCDKAZ12A8C13C80F',
#    'SOGDDKR12A6701E8FA',
#    'SOWULAI12A67020A1E',
#    'SOJAYEY12AB0185304',
#    'SOMFLEZ12A6D4F907C',
#    'SOYCXBA12A6701E35B',
#    'SOGDTQS12A6310D7D1',
#    'SOYRVSP12A6D4F907A',
#    'SOJHEQO12A670203C4',
#    'SOXSQCC12A8C13D78E',
#    'SOJHVCP12A6701D0F3',
#    'SOQZARZ12A6D4F814D'])]

