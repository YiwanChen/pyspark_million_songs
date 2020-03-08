
# Imports

from pyspark import SparkContext
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import *

# from pyspark.sql.functions import udf#, col, lit
import re
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StandardScaler

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.classification import *

import pandas as pd

# Required to allow the file to be submitted and run using spark-submit instead
# of using pyspark interactively

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()

# Audio Similarity Q1
# (a) Use /msd-jmir-methods-of-moments-all-v1.0.csv

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

def get_audio_dataset_path(name):
	audio_dataset_path = f"hdfs:///data/msd/audio/features/{name}.csv"
	return audio_dataset_path

get_audio_dataset_path(audio_dataset_names[2])

# load 3rd dataset from audio features 'msd-jmir-methods-of-moments-all-v1.0'

jimir_mtd = (
	spark.read.format("com.databricks.spark.csv")
	.option('header', 'false')
	.option("inferSchema", "false")
	.option('codec', 'gzip')
	.schema(audio_dataset_schemas[audio_dataset_names[2]])
	.load(get_audio_dataset_path(audio_dataset_names[2]))
	)

jimir_mtd.cache()
jimir_mtd.show(10)

print(jimir_mtd.count()) 	#994623

# (a) descriptive statistics for jimir_mtd




jimir_mtd.describe().show()

# +-------+----------------------------------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+--------------------+
# |summary|Method_of_Moments_Overall_Standard_Deviation_1|Method_of_Moments_Overall_Standard_Deviation_2|Method_of_Moments_Overall_Standard_Deviation_3|Method_of_Moments_Overall_Standard_Deviation_4|Method_of_Moments_Overall_Standard_Deviation_5|Method_of_Moments_Overall_Average_1|Method_of_Moments_Overall_Average_2|Method_of_Moments_Overall_Average_3|Method_of_Moments_Overall_Average_4|Method_of_Moments_Overall_Average_5|         MSD_TRACKID|
# +-------+----------------------------------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+--------------------+
# |  count|                                        994623|                                        994623|                                        994623|                                        994623|                                        994623|                             994623|                             994623|                             994623|                             994623|                             994623|              994623|
# |   mean|                           0.15498176001746342|                            10.384550576952307|                             526.8139724398096|                             35071.97543290272|                             5297870.369577217|                 0.3508444432531317|                 27.463867987840707|                 1495.8091812075545|                 143165.46163257837|                2.396783048473542E7|                null|
# | stddev|                           0.06646213086143025|                            3.8680013938746836|                             180.4377549977526|                            12806.816272955562|                            2089356.4364558065|                0.18557956834383815|                  8.352648595163764|                  505.8937639190231|                 50494.276171032274|                  9307340.299219666|                null|
# |    min|                                           0.0|                                           0.0|                                           0.0|                                           0.0|                                           0.0|                                0.0|                                0.0|                                0.0|                          -146300.0|                                0.0|'TRAAAAK128F9318786'|
# |    max|                                         0.959|                                         55.42|                                        2919.0|                                      407100.0|                                       4.657E7|                              2.647|                              117.0|                             5834.0|                           452500.0|                            9.477E7|'TRZZZZO128F428E2D4'|
# +-------+----------------------------------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+--------------------+



from pyspark.mllib.stat import Statistics

jimir_numeric = jimir_mtd.drop('MSD_TRACKID')

features = jimir_numeric.rdd.map(lambda row: row[0])
corr_mat=Statistics.corr(features)

df = jimir_numeric
col_names = df.columns
features = df.rdd.map(lambda row: row[0:])
corr_mat=Statistics.corr(features, method="pearson")
corr_df = pd.DataFrame(corr_mat)
#corr_df.index, corr_df.columns = col_names, col_names

round(corr_df,2)

# correlation matrix
#        0      1      2      3      4      5      6      7      8      9
# 0  1.000  0.426  0.296  0.061 -0.055  0.754  0.498  0.448  0.167  0.100
# 1  0.426  1.000  0.858  0.610  0.434  0.025  0.407  0.396  0.016 -0.041
# 2  0.296  0.858  1.000  0.803  0.683 -0.082  0.126  0.185 -0.088 -0.135
# 3  0.061  0.610  0.803  1.000  0.942 -0.328 -0.223 -0.158 -0.245 -0.221
# 4 -0.055  0.434  0.683  0.942  1.000 -0.393 -0.355 -0.286 -0.260 -0.212
# 5  0.754  0.025 -0.082 -0.328 -0.393  1.000  0.549  0.519  0.347  0.279
# 6  0.498  0.407  0.126 -0.223 -0.355  0.549  1.000  0.903  0.516  0.423
# 7  0.448  0.396  0.185 -0.158 -0.286  0.519  0.903  1.000  0.773  0.686
# 8  0.167  0.016 -0.088 -0.245 -0.260  0.347  0.516  0.773  1.000  0.985
# 9  0.100 -0.041 -0.135 -0.221 -0.212  0.279  0.423  0.686  0.985  1.000



# (b)
# load -MAGD-genreAssignment.tsv
genres_assign_schema = StructType([
    StructField('track_id', StringType(), True),
    StructField('genres', StringType(), True)
])

genres_assign = (
	spark.read.format("com.databricks.spark.csv")
	.option('header', 'false')
	.option('delimiter', '\t')
	.option('codec', 'gzip')
	.schema(genres_assign_schema)
	.load('hdfs:///data/msd/genre/msd-MAGD-genreAssignment.tsv')
	)

genres_assign.cache()
genres_assign.show(10)

print(genres_assign.count())	#422714


genres_group = (
	genres_assign
	.groupBy('genres')
	.agg(F.count('genres'))
	.orderBy('count(genres)', ascending = False)
	)



# Visualization

genres_group.toPandas().to_csv('genresgroup.csv')


# (c) Merge the genres dataset and the audio features dataset // use 3rd Dataset

# remove comma in jimir_mtd column 'MSD_TRACKID'
commaRep = F.udf(lambda x: re.sub("'","", x))
jimir_mtd_rmcomma = jimir_mtd.withColumn('MSD_TRACKID', commaRep('MSD_TRACKID'))
jimir_mtd_rmcomma.show()


features_genres = (
	jimir_mtd_rmcomma
#	.select([jimir_mtd[c] for c in jimir_mtd.columns])
	.join(
		genres_assign
		.select(
			F.col('genres'),
			F.col('track_id').alias('MSD_TRACKID')
			)
		,
	on = 'MSD_TRACKID',
	how = 'inner'
	)
	)
features_genres.cache()
features_genres.show()

# Q2 
# (b) # Convert genre to Electronic and others.




features_genres_Elec = features_genres.withColumn('genres', F.when(features_genres.genres == 'Electronic', 'Electronic').otherwise('Other')) #.cast("double")

# features_genres_binary.describe()
# test if converstion is right
print(features_genres_Elec.filter(features_genres_binary['genres']=='Electronic').count())		#40666
print(features_genres_Elec.filter(features_genres_binary['genres']=='Other').count())		#379954	 total with above = 420620
features_genres.filter(features_genres['genres']=='Electronic').count() # check if the number match

# features_genres_Elec = sc.parallelize(features_genres_Elec)

# repartition
conf = sc.getConf()
N = int(conf.get("spark.executor.instances"))
M = int(conf.get("spark.executor.cores"))
partitions = 4 * N * M

features_genres_Elec =  features_genres_Elec.repartition(partitions)

# save the data frame
# features_genres_Elec.write.parquet('hdfs:///user/ych192/outputs/mds/features_genres_Elec.parquet')

features_genres_Elec = spark.read.parquet('hdfs:///user/ych192/outputs/mds/features_genres_Elec.parquet')
features_genres_Elec.cache()
#
seed = 1

######################################
# define function to report recall and precision
def evaluate_manual(predictionAndLabels):
    log = {}

    # Show Validation Score (AUROC)
    evaluator = BinaryClassificationEvaluator(metricName='areaUnderROC')
    log['AUROC'] = "%f" % evaluator.evaluate(predictionAndLabels)    
    print("Area under ROC = {}".format(log['AUROC']))

    # Show Validation Score (AUPR)
    evaluator = BinaryClassificationEvaluator(metricName='areaUnderPR')
    log['AUPR'] = "%f" % evaluator.evaluate(predictionAndLabels)
    print("Area under PR = {}".format(log['AUPR']))

    # Metrics
    predictionRDD = predictionAndLabels.select(['label', 'prediction']) \
                            .rdd.map(lambda line: (line[1], line[0]))
    metrics = MulticlassMetrics(predictionRDD)

    # Confusion Matrix
    print(metrics.confusionMatrix().toArray())

    # Overall statistics
    log['precision'] = "%s" % metrics.precision()
    log['recall'] = "%s" % metrics.recall()
    log['F1 Measure'] = "%s" % metrics.fMeasure()
    print("[Overall]\tprecision = %s | recall = %s | F1 Measure = %s" % \
            (log['precision'], log['recall'], log['F1 Measure']))

    # Statistics by class
    labels = [0.0, 1.0]
    for label in sorted(labels):
        log[label] = {}
        log[label]['precision'] = "%s" % metrics.precision(label)
        log[label]['recall'] = "%s" % metrics.recall(label)
        log[label]['F1 Measure'] = "%s" % metrics.fMeasure(label, beta=1.0)
        print("[Class %s]\tprecision = %s | recall = %s | F1 Measure = %s" \
                  % (label, log[label]['precision'], 
                    log[label]['recall'], log[label]['F1 Measure']))

    return log




# train_electronic = features_genres_Elec.sampleBy("genres", fractions={'Electronic': 0.7, 'Other': 0.7})  # count = 294160

# test_electronic = features_genres_Elec.join(train_electronic, on='MSD_TRACKID', how= 'left_anti')		# count =	126460
# train_electronic.cache()
# test_electronic.cache()

# (d)


# train test split by 0.7/ 0.3 by Electrinoic and others.
train_raw = features_genres_Elec.sampleBy("genres", fractions={'Electronic': 0.7, 'Other': 0.7})
test_raw = features_genres_Elec.join(train_raw, on='MSD_TRACKID', how= 'left_anti')	

#pipe line
label_stringIdx = StringIndexer(inputCol = "genres", outputCol = "label")
input_columns = [c for c in features_genres_Elec.columns if c not in ['MSD_TRACKID', 'genres', 'label']]
assemblr = VectorAssembler(inputCols = input_columns , outputCol = 'features_original')
scaler = StandardScaler(inputCol="features_original", outputCol="features",
                        withStd=True, withMean=False)
pipeline = Pipeline(stages=[label_stringIdx, assemblr, scaler])


#pipelineFit = pipeline.fit(features_genres_Elec)

# dataset = pipelineFit.transform(features_genres_Elec)

# train = dataset.sampleBy("genres", fractions={'Electronic': 0.7, 'Other': 0.7})  # count = 294505
# test = dataset.join(train, on='MSD_TRACKID', how= 'left_anti')		# count =	126115


pipelineFit = pipeline.fit(train_raw)
train = pipelineFit.transform(train_raw)

pipelineFit = pipeline.fit(test_raw)
test = pipelineFit.transform(test_raw)


train.cache()
test.cache()


features_genres_Elec.where(train.genres == 'Electronic').count()	#40666 in train test total
train.where(train.genres == 'Electronic').count()	#Count of Electrinoic 28440
train.where(train.genres != 'Electronic').count()	#Count of others 265952 


# subsampling
sub_train_other = train.where(train.genres == 'Other').sample(28440/265952) 	#count = 28559
sub_train = sub_train_other.union(train.where(train.genres == 'Electronic'))	#count = 56999

# oversampling
over_train_elec = train.where(train.genres == 'Electronic').sample(True, 265952/28440) 	#count = 266285
over_train = over_train_elec.union(train.where(train.genres == 'Other'))	#count = 532237


lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
lrModel = lr.fit(train)
lr_predictions = lrModel.transform(test)


# lrModel.save('hdfs:///user/ych192/outputs/mds/lrModel')
#lrModel = LogisticRegressionModel.load('hdfs:///user/ych192/outputs/mds/lrModel')

lr_predictions.select('MSD_TRACKID',"genres","label",'rawPrediction',"probability","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)


# |       MSD_TRACKID|    genres|label|                 rawPrediction|                   probability|prediction|
# +------------------+----------+-----+------------------------------+------------------------------+----------+
# |TRAXIBQ128F149B03A|     Other|  0.0|[3.421742453903383,-3.42174...|[0.9683771740476963,0.03162...|       0.0|
# |TRCATFU128F423B64C|     Other|  0.0|[3.421742453903383,-3.42174...|[0.9683771740476963,0.03162...|       0.0|
# |TRHOYLY128F9329865|     Other|  0.0|[3.421742453903383,-3.42174...|[0.9683771740476963,0.03162...|       0.0|
# |TRRRANQ12903CED7A5|     Other|  0.0|[3.421742453903383,-3.42174...|[0.9683771740476963,0.03162...|       0.0|
# |TRPDLIO128F931513A|     Other|  0.0|[3.421742453903383,-3.42174...|[0.9683771740476963,0.03162...|       0.0|

evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol='label', metricName='areaUnderROC')
evaluator.evaluate(lr_predictions) 	# areaUnderROC 0.695


print(lr_predictions.where((lr_predictions.label == 1) & (lr_predictions.prediction == 1)).count())
print(lr_predictions.where((lr_predictions.label == 0) & (lr_predictions.prediction == 0)).count())

print(lr_predictions.where((lr_predictions.label == 1) & (lr_predictions.prediction == 0)).count())
print(lr_predictions.where((lr_predictions.label == 0) & (lr_predictions.prediction == 1)).count())

# 			Predict
#			P 			N
#Actual P 	3			12317
#Actual N 	23			114251


# Metrics


# Negative / 0 are on top left corner ####################
# [1.14251e+05 2.30000e+01]
#  [1.23170e+04 3.00000e+00]

metrics.confusionMatrix().toArray()
    # Overall statistics

print(metrics.precision())
print(metrics.recall())
print(metrics.fMeasure())

# evaluate lr prediction model, stratifiled sampling
evaluate_manual(lr_predictions)
# Area under ROC = 0.695546
# Area under PR = 0.218714
# [[1.13918e+05 3.80000e+01]
#  [1.22700e+04 2.00000e+00]]
# [Overall]       precision = 0.902493899927116 | recall = 0.902493899927116 | F1 Measure = 0.902493899927116
# [Class 0.0]     precision = 0.9027641297112245 | recall = 0.9996665379620204 | F1 Measure = 0.9487474182157373
# [Class 1.0]     precision = 0.05 | recall = 0.00016297262059973924 | F1 Measure = 0.00032488628979857043

# alternate prediction threshold
lr_predictions = lrModel.transform(test, {lrModel.threshold:0.05})
evaluate_manual(lr_predictions)

# Area under ROC = 0.695546
# Area under PR = 0.218714
# [[   770. 113186.]
#  [   182.  12090.]]
# [Overall]       precision = 0.10187913933517127 | recall = 0.10187913933517127 | F1 Measure = 0.10187913933517127
# [Class 0.0]     precision = 0.8088235294117647 | recall = 0.0067569939274807825 | F1 Measure = 0.013402025968600968
# [Class 1.0]     precision = 0.09650691273667741 | recall = 0.9851694915254238 | F1 Measure = 0.1757931776543461



# Test subsampling result ---------------------------------------------------------------------------------------------------
lrModel = lr.fit(sub_train)
lr_predictions = lrModel.transform(test)

evaluate_manual(lr_predictions)

# Area under ROC = 0.701369
# Area under PR = 0.225076
# [[77103. 36853.]
#  [ 4476.  7796.]]
# [Overall]       precision = 0.6725845295813924 | recall = 0.6725845295813924 | F1 Measure = 0.6725845295813924
# [Class 0.0]     precision = 0.9451329386239106 | recall = 0.6766032503773386 | F1 Measure = 0.7886363055207508
# [Class 1.0]     precision = 0.1746063741629152 | recall = 0.6352672750977836 | F1 Measure = 0.2739235080198872

lr_predictions = lrModel.transform(test, {lrModel.threshold:0.05})
evaluate_manual(lr_predictions)

# Area under ROC = 0.701369
# Area under PR = 0.225076
# [[     0. 113956.]
#  [     0.  12272.]]
# [Overall]       precision = 0.09722090186012612 | recall = 0.09722090186012612 | F1 Measure = 0.09722090186012612
# [Class 0.0]     precision = 0.0 | recall = 0.0 | F1 Measure = 0.0
# [Class 1.0]     precision = 0.09722090186012612 | recall = 1.0 | F1 Measure = 0.17721299638989169

lr_predictions = lrModel.transform(test, {lrModel.threshold:0.4})
evaluate_manual(lr_predictions)
# Area under ROC = 0.701369
# Area under PR = 0.225076
# [[29342. 84614.]
#  [ 1324. 10948.]]
# [Overall]       precision = 0.3191843331115125 | recall = 0.3191843331115125 | F1 Measure = 0.3191843331115125
# [Class 0.0]     precision = 0.9568251483727908 | recall = 0.2574853452209625 | F1 Measure = 0.40577505497088967
# [Class 1.0]     precision = 0.11456436658923003 | recall = 0.8921121251629727 | F1 Measure = 0.2030528404770295

# Test oversampling result ---------------------------------------------------------------------------------------------------
lrModel = lr.fit(over_train)
lr_predictions = lrModel.transform(test)

evaluate_manual(lr_predictions)

# Area under ROC = 0.700946
# Area under PR = 0.225668
# [[76547. 37409.]
#  [ 4442.  7830.]]
# [Overall]       precision = 0.6684491554964034 | recall = 0.6684491554964034 | F1 Measure = 0.6684491554964034
# [Class 0.0]     precision = 0.9451530454753114 | recall = 0.6717241742426902 | F1 Measure = 0.7853189361101849
# [Class 1.0]     precision = 0.17308074891133757 | recall = 0.6380378096479792 | F1 Measure = 0.27229573472900837

lr_predictions = lrModel.transform(test, {lrModel.threshold:0.4})
evaluate_manual(lr_predictions)

# Area under ROC = 0.700946
# Area under PR = 0.225668
# [[28652. 85304.]
#  [ 1294. 10978.]]
# [Overall]       precision = 0.3139556992109516 | recall = 0.3139556992109516 | F1 Measure = 0.3139556992109516
# [Class 0.0]     precision = 0.9567888866626595 | recall = 0.25143037663659656 | F1 Measure = 0.39821545218273546
# [Class 1.0]     precision = 0.11401923516337426 | recall = 0.8945567144719687 | F1 Measure = 0.20225878364684857


## Cross Validation with LR
lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0)
evaluator = BinaryClassificationEvaluator()
# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.1, 0.3, 0.5]) # regularization parameter
             #.addGrid(lr.elasticNetParam, [0, 0.01, 0.1, 0.5, 0.8]) # Elastic Net Parameter (Ridge = 0)
            .addGrid(lr.maxIter, [20]) #Number of iterations
             .build())
# Create 5-fold CrossValidator
cv = CrossValidator(estimator=lr, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, \
                    numFolds=5,
                    parallelism=2)
cvModel = cv.fit(sub_train)

cv_lr_predictions = cvModel.transform(test)

cvModel.bestModel.extractParamMap()

# Best model
# regParam = 0.1, elasticNetParam = 0.00, maxIter = 20

# Evaluate best model(

evaluate_manual(cv_lr_predictions)

# Area under ROC = 0.732438
# Area under PR = 0.253928
# [[77197. 36759.]
#  [ 3970.  8302.]]
# [Overall]       precision = 0.6773378331273568 | recall = 0.6773378331273568 | F1 Measure = 0.6773378331273568
# [Class 0.0]     precision = 0.9510884965564823 | recall = 0.6774281301554986 | F1 Measure = 0.7912649969506415
# [Class 1.0]     precision = 0.184239142495728 | recall = 0.6764993481095176 | F1 Measure = 0.2896063349205519


# Gradient-boosted tree classifier

gbTree = GBTClassifier(maxIter=20, stepSize=0.1, maxDepth=5, maxBins=32)

#gbTree.explainParams()
gbTreeModel = gbTree.fit(train)
gbTree_predictions = gbTreeModel.transform(test)
evaluate_manual(gbTree_predictions)

# Area under ROC = 0.807213
# Area under PR = 0.376005
# [[113017.    939.]
#  [ 10736.   1536.]]
# [Overall]       precision = 0.9075086351681085 | recall = 0.9075086351681085 | F1 Measure = 0.9075086351681085
# [Class 0.0]     precision = 0.913246547558443 | recall = 0.9917599775351891 | F1 Measure = 0.950885326176123
# [Class 1.0]     precision = 0.6206060606060606 | recall = 0.12516297262059975 | F1 Measure = 0.20831355529938295

# gbTree on subsampling 
gbTreeModel = gbTree.fit(sub_train)
gbTree_predictions = gbTreeModel.transform(test)
evaluate_manual(gbTree_predictions)

# Area under ROC = 0.810135
# Area under PR = 0.373752
# [[85897. 28059.]
#  [ 3603.  8669.]]
# [Overall]       precision = 0.7491681718794562 | recall = 0.7491681718794562 | F1 Measure = 0.7491681718794562
# [Class 0.0]     precision = 0.9597430167597766 | recall = 0.7537733862192425 | F1 Measure = 0.8443791286568103
# [Class 1.0]     precision = 0.23603245480287519 | recall = 0.7064048239895697 | F1 Measure = 0.35383673469387755

# gbTree on oversampling 
gbTreeModel = gbTree.fit(over_train)
gbTree_predictions = gbTreeModel.transform(test)
evaluate_manual(gbTree_predictions)

# Area under ROC = 0.810396
# Area under PR = 0.365030
# [[86052. 27904.]
#  [ 3568.  8704.]]
# [Overall]       precision = 0.7506733846690117 | recall = 0.7506733846690117 | F1 Measure = 0.7506733846690117
# [Class 0.0]     precision = 0.9601874581566615 | recall = 0.7551335603215276 | F1 Measure = 0.8454041733799662
# [Class 1.0]     precision = 0.23776223776223776 | recall = 0.7092568448500652 | F1 Measure = 0.35613747954173486


gbTreeModel.featureImportances()
# SparseVector(10, {0: 0.1012, 1: 0.1261, 2: 0.0589, 3: 0.1159, 4: 0.074, 5: 0.0595, 6: 0.1035, 7: 0.1282, 8: 0.0818, 9: 0.1508})

# GbTree cross validation 
gbTree = GBTClassifier(maxIter=20, stepSize=0.1, maxDepth=5, maxBins=32)
paramGrid = (ParamGridBuilder()
             .addGrid(gbTree.stepSize, [0.01, 0.1, 0.2])
             .addGrid(gbTree.maxDepth, [3, 4 ,5]) 
             .build())
evaluator = BinaryClassificationEvaluator()
cv = CrossValidator(estimator=gbTree, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, \
                    numFolds=5)

cvGbTreeModel = cv.fit(sub_train)
cv_gbTree_predictions = cvGbTreeModel.transform(test)
evaluate_manual(gbTree_predictions)

cvGbTreeModel.bestModel.extractParamMap()	#stepsize = 0.2, maxDepth = 5

# Area under ROC = 0.810396
# Area under PR = 0.365030
# [[86052. 27904.]
#  [ 3568.  8704.]]
# [Overall]       precision = 0.7506733846690117 | recall = 0.7506733846690117 | F1 Measure = 0.7506733846690117
# [Class 0.0]     precision = 0.9601874581566615 | recall = 0.7551335603215276 | F1 Measure = 0.8454041733799662
# [Class 1.0]     precision = 0.23776223776223776 | recall = 0.7092568448500652 | F1 Measure = 0.35613747954173486




# LinearSVC model

LSVC = LinearSVC(maxIter=5, regParam=0.01, standardization=False)
# axIter=100, regParam=0.0, tol=1e-06,  standardization=True, threshold=0.0, weightCol=None, aggregationDepth=2
LSVCmodel = LSVC.fit(train)
LSVCpredictions = LSVCmodel.transform(test)

evaluate_manual(LSVCpredictions)

# Area under ROC = 0.318782
# Area under PR = 0.075114
# [[113956.      0.]
#  [ 12272.      0.]]
# [Overall]       precision = 0.9027790981398739 | recall = 0.9027790981398739 | F1 Measure = 0.9027790981398739
# [Class 0.0]     precision = 0.9027790981398739 | recall = 1.0 | F1 Measure = 0.9489058388568764
# [Class 1.0]     precision = 0.0 | recall = 0.0 | F1 Measure = 0.0

# Subsample
LSVCmodel = LSVC.fit(sub_train)
LSVCpredictions = LSVCmodel.transform(test)
evaluate_manual(LSVCpredictions)

# Area under ROC = 0.626657
# Area under PR = 0.177135
# [[66662. 47294.]
#  [ 5253.  7019.]]
# [Overall]       precision = 0.5837135976170105 | recall = 0.5837135976170105 | F1 Measure = 0.5837135976170105
# [Class 0.0]     precision = 0.9269554334978795 | recall = 0.584980167784057 | F1 Measure = 0.7172931764503339
# [Class 1.0]     precision = 0.12923241212969272 | recall = 0.5719524119947849 | F1 Measure = 0.21082826462416462

# oversample
LSVCmodel = LSVC.fit(over_train)
LSVCpredictions = LSVCmodel.transform(test)
evaluate_manual(LSVCpredictions)

# Area under ROC = 0.621334
# Area under PR = 0.172697
# [[63561. 50395.]
#  [ 5022.  7250.]]
# [Overall]       precision = 0.5609769623221472 | recall = 0.5609769623221472 | F1 Measure = 0.5609769623221472
# [Class 0.0]     precision = 0.9267748567429246 | recall = 0.5577679104215663 | F1 Measure = 0.6964100822290031
# [Class 1.0]     precision = 0.12576979790094545 | recall = 0.5907757496740548 | F1 Measure = 0.2073887609594233


## Cross Validation with LSVC
LSVC = LinearSVC(maxIter=5, regParam=0.01, standardization=False)
paramGrid = (ParamGridBuilder()
             .addGrid(LSVC.regParam, [0.001, 0.01, 0.1, 1])
             .addGrid(LSVC.maxIter, [5, 10,20]) 
             .build())

cv = CrossValidator(estimator=LSVC, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, \
                    numFolds=5)
cvModel = cv.fit(sub_train)

cv_lsvc_predictions = cvModel.transform(test)
# Evaluate best model(
evaluator.evaluate(cv_lsvc_predictions)
evaluate_manual(cv_lsvc_predictions)

# Area under ROC = 0.676170
# Area under PR = 0.208303
# [[77685. 36271.]
#  [ 5171.  7101.]]
# [Overall]       precision = 0.6716893240802357 | recall = 0.6716893240802357 | F1 Measure = 0.6716893240802357
# [Class 0.0]     precision = 0.9375905184899102 | recall = 0.6817104847484994 | F1 Measure = 0.7894335711237119
# [Class 1.0]     precision = 0.1637231393525777 | recall = 0.5786342894393742 | F1 Measure = 0.2552296743584214

cvModel.bestModel.extractParamMap()
# regParam = 1, maxIter = 20

# Random Forest


rf = RandomForestClassifier(labelCol="label", \
                            featuresCol="features", \
                            numTrees = 3, \
                            maxDepth = 4, \
                            maxBins = 32)
# Train model with Training Data
rfModel = rf.fit(train)
rf_predictions = rfModel.transform(test)

evaluate_manual(rf_predictions)

# Area under ROC = 0.500000
# Area under PR = 0.097221
# [[113956.      0.]
#  [ 12272.      0.]]
# [Overall]       precision = 0.9027790981398739 | recall = 0.9027790981398739 | F1 Measure = 0.9027790981398739
# [Class 0.0]     precision = 0.9027790981398739 | recall = 1.0 | F1 Measure = 0.9489058388568764
# [Class 1.0]     precision = 0.0 | recall = 0.0 | F1 Measure = 0.0



rfModel = rf.fit(sub_train)
rf_predictions = rfModel.transform(test)
evaluate_manual(rf_predictions)

# Area under ROC = 0.764170
# Area under PR = 0.276076
# [[86371. 27585.]
#  [ 4557.  7715.]]
# [Overall]       precision = 0.7453655290426847 | recall = 0.7453655290426847 | F1 Measure = 0.7453655290426847
# [Class 0.0]     precision = 0.9498834242477565 | recall = 0.7579328863771982 | F1 Measure = 0.8431209855332774
# [Class 1.0]     precision = 0.21855524079320113 | recall = 0.6286668839634941 | F1 Measure = 0.32435045825275377


rfModel = rf.fit(over_train)
rf_predictions = rfModel.transform(test)
evaluate_manual(rf_predictions)

# Area under ROC = 0.765009
# Area under PR = 0.292741
# [[80136. 33820.]
#  [ 3887.  8385.]]
# [Overall]       precision = 0.7012786386538644 | recall = 0.7012786386538644 | F1 Measure = 0.7012786386538644
# [Class 0.0]     precision = 0.9537388572176666 | recall = 0.7032187861981818 | F1 Measure = 0.8095404058006153
# [Class 1.0]     precision = 0.19867314299253644 | recall = 0.6832627118644068 | F1 Measure = 0.3078363345999229




## Cross Validation with Random forest
paramGrid = (ParamGridBuilder()
             .addGrid(rf.numTrees, [2,3,4])
             .addGrid(rf.maxDepth, [4,5]) 
             .build())

cv = CrossValidator(estimator=rf, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, \
                    numFolds=5)
cvModel = cv.fit(train)

cv_rf_predictions = cvModel.transform(test)
# Evaluate best model(
evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(cv_rf_predictions)

evaluate_manual(cv_rf_predictions)






# Naive Bayes 

scaler = MinMaxScaler(inputCol="features_original", outputCol="features")
pipeline = Pipeline(stages=[label_stringIdx, assemblr, scaler])
pipelineFit = pipeline.fit(features_genres_Elec)

train = pipelineFit.transform(train_raw)
test = pipelineFit.transform(test_raw)


# subsampling
sub_train_other = train.where(train.genres == 'Other').sample(28440/265952) 	#count = 28559
sub_train = sub_train_other.union(train.where(train.genres == 'Electronic'))	#count = 56999

# oversampling
over_train_elec = train.where(train.genres == 'Electronic').sample(True, 265952/28440) 	#count = 266285
over_train = over_train_elec.union(train.where(train.genres == 'Other'))	#count = 532237


nb = NaiveBayes(smoothing=1, modelType="multinomial")
nbmodel = nb.fit(train)
nbpredictions = nbmodel.transform(test)
evaluate_manual(nbpredictions)

# Area under ROC = 0.344596
# Area under PR = 0.076973
# [[113956.      0.]
#  [ 12272.      0.]]
# [Overall]       precision = 0.9027790981398739 | recall = 0.9027790981398739 | F1 Measure = 0.9027790981398739
# [Class 0.0]     precision = 0.9027790981398739 | recall = 1.0 | F1 Measure = 0.9489058388568764
# [Class 1.0]     precision = 0.0 | recall = 0.0 | F1 Measure = 0.0


nb = NaiveBayes(smoothing=1, modelType="multinomial")
nbmodel = nb.fit(sub_train)
nbpredictions = nbmodel.transform(test)
evaluate_manual(nbpredictions)


# Area under ROC = 0.344596
# Area under PR = 0.076973
# [[73195. 40761.]
#  [ 6249.  6023.]]
# [Overall]       precision = 0.6275786671736857 | recall = 0.6275786671736857 | F1 Measure = 0.6275786671736857
# [Class 0.0]     precision = 0.9213408186898948 | recall = 0.6423093123661764 | F1 Measure = 0.7569286452947259
# [Class 1.0]     precision = 0.1287405950752394 | recall = 0.4907920469361147 | F1 Measure = 0.20397588729341642



## Cross Validation with Random forest
paramGrid = (ParamGridBuilder()
             .addGrid(rf.numTrees, [2,3,4])
             .addGrid(rf.maxDepth, [4,5]) 
             .build())

cv = CrossValidator(estimator=rf, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, \
                    numFolds=5)
cvModel = cv.fit(train)