# STAT420-19S1 Assignment 2
# Data Processing Q2

# Imports

from pyspark import SparkContext
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import *

# Required to allow the file to be submitted and run using spark-submit instead
# of using pyspark interactively

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()



# Processing Q2(a)

#hdfs dfs -cat /data/msd/tasteprofile//mismatches/sid_mismatches.txt | head




# mismatches 

mismaches_schema = StructType([
    StructField('song_id', StringType(), True),
    StructField('song_artist', StringType(), True),
    StructField('song_title', StringType(), True),
    StructField('track_id', StringType(), True),
    StructField('track_artist', StringType(), True),
    StructField('track_title', StringType(), True),
])

# copy to local
# hdfs dfs -copyToLocal hdfs:///data/msd/tasteprofile/mismatches/sid_matches_manually_accepted.txt

 # test = open('hdfs:///node0:9000/data/msd/tasteprofile/mismatches/sid_matches_manually_accepted.txt')



with open('sid_matches_manually_accepted.txt') as f:
	lines = f.readlines()
	sid_matches_manually_accepted = []
	for line in lines:
		if line.startswith('< ERROR: '):
			a = line[10:28]
			b = line[29:47]
			c, d = line[49:-1].split('  !=  ')
			e, f = c.split('  -  ')
			g, h = d.split('  -  ')
			sid_matches_manually_accepted.append((a, e, f, b, g, h))

matches_manually_accepted = spark.createDataFrame(sc.parallelize(sid_matches_manually_accepted, 64), schema=mismaches_schema)
matches_manually_accepted.cache()
matches_manually_accepted.show(10,50)

print(matches_manually_accepted.count())  #488

with open('sid_mismatches.txt') as f:
	lines = f.readlines()
	sid_mismatches = []
	for line in lines:
		if line.startswith('ERROR: '):
			a = line[8:26]
			b = line[27:45]
			c, d = line[47:-1].split('  !=  ')
			e, f = c.split('  -  ')
			g, h = d.split('  -  ')
			sid_mismatches.append((a, e, f, b, g, h))

mismatches = spark.createDataFrame(sc.parallelize(sid_mismatches, 64), schema=mismaches_schema)
mismatches.cache()
mismatches.show(10,50)

print(mismatches.count())	#19094



# load triplet
triplets_schema = StructType([
    StructField('user_id', StringType(), True),
    StructField('song_id', StringType(), True),
    StructField('plays', IntegerType(), True)
])
triplets = (
	spark.read.format("com.databricks.spark.csv")
	.option('header', 'false')
	.option('delimiter', '\t')
	.option('codec', 'gzip')
	.schema(triplets_schema)
	.load('hdfs:////data/msd/tasteprofile/triplets.tsv/')
	.cache()
	)

triplets.cache()
triplets.show(10)

# +--------------------+------------------+-----+
# |             user_id|           song_id|plays|
# +--------------------+------------------+-----+
# |f1bfc2a4597a3642f...|SOQEFDN12AB017C52B|    1|
# |f1bfc2a4597a3642f...|SOQOIUJ12A6701DAA7|    2|
# |f1bfc2a4597a3642f...|SOQOKKD12A6701F92E|    4|
# |f1bfc2a4597a3642f...|SOSDVHO12AB01882C7|    1|

# join mismatch

mismatches_not_accepted = mismatches.join(matches_manually_accepted, on='song_id', how = 'left_anti')
triplets_not_mismatchet = triplets.join(mismatches_not_accepted, on='song_id', how='left_anti')

print(triplets.count())		# 48373586
print(triplets_not_mismatchet.count())		#45795111

triplets_not_mismatchet.write.parquet('hdfs:///user/ych192/outputs/mds/triplets_not_mismatchet.parquet')


# Processing Q2(b)

# hdfs dfs -cat "/data/msd/audio/attributes/*" | awk -F',' '{print $2}' | sort | uniq
# NUMERIC
# real
# real
# string
# string
# STRING

audio_attribute_type_mapping = {
	'NUMERIC':DoubleType(),
	'real':DoubleType(),
	'string':StringType(),
	'STRING':StringType()
}

audio_dataset_names = [
'msd-jmir-area-of-moments-all-v1.0',
'msd-jmir-lpc-all-v1.0',
'msd-jmir-methods-of-moments-all-v1.0',
'msd-jmir-mfcc-all-v1.0',
'msd-jmir-spectral-all-all-v1.0',
'msd-jmir-spectral-derivatives-all-all-v1.0',
'msd-marsyas-timbral-v1.0',
'msd-mvd-v1.0',
'msd-rh-v1.0',
'msd-rp-v1.0',
'msd-ssd-v1.0',
'msd-trh-v1.0',
'msd-tssd-v1.0'
]


# copy to local
# hdfs dfs -copyToLocal hdfs:///data/msd/audio/attributes


audio_dataset_schemas = {}
for audio_dataset_name  in audio_dataset_names:
	print(audio_dataset_name)
	audio_dataset_path = f"attributes/{audio_dataset_name}.attributes.csv"
	with open(audio_dataset_path, 'r') as f:
		rows = [line.strip().split(',') for line in f.readlines()]
	audio_dataset_schemas[audio_dataset_name] = StructType([StructField(row[0], audio_attribute_type_mapping[row[1]],True) for row in rows])


# 'http://lcoalhost:50070/data/msd/audio/attributes'
