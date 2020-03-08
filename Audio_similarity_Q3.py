from pyspark import SparkContext
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import *

# from pyspark.sql.functions import udf#, col, lit
import re
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StandardScaler

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.classification import *

import pandas as pd


seed = 1


# save the data frame
# features_genres.write.parquet('hdfs:///user/ych192/outputs/mds/features_genres.parquet')

features_genres = spark.read.parquet('hdfs:///user/ych192/outputs/mds/features_genres.parquet')
features_genres.cache()


features_genres.rdd.getNumPartitions()

conf = sc.getConf()
N = int(conf.get("spark.executor.instances"))
M = int(conf.get("spark.executor.cores"))
partitions = 4 * N * M

features_genres =  features_genres.repartition(partitions)

label_stringIdx = StringIndexer(inputCol = "genres", outputCol = "label")
encoder = OneHotEncoder(inputCol="indexed", outputCol="label")
input_columns = [c for c in features_genres.columns if c not in ['MSD_TRACKID', 'genres', 'label']]
assemblr = VectorAssembler(inputCols = input_columns , outputCol = 'features_original')

scaler = StandardScaler(inputCol="features_original", outputCol="features",
                        withStd=True, withMean=False)

pipeline = Pipeline(stages=[label_stringIdx, assemblr, scaler])
pipelineFit = pipeline.fit(features_genres)
dataset = pipelineFit.transform(features_genres)


dataset.select(F.col('genres')).distinct().collect()

genres_type = dataset.select(F.col('genres')).distinct().rdd.map(lambda r: r[0]).collect()

fraction_dict = {key:0.7 for key in genres_type }


train = dataset.sampleBy("genres", fractions=fraction_dict)  

test = dataset.join(train, on='MSD_TRACKID', how= 'left_anti')		

train.cache()
test.cache()


# logistic Regression
lr = LogisticRegression(maxIter=20, regParam=0.1, elasticNetParam=1, family="multinomial")
lrModel = lr.fit(train)
predictions = lrModel.transform(test)
predictions.filter(predictions['prediction'] == 0) \
    .select("genres","label","features","probability","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)

# +-----------+-----+------------------------------+------------------------------+----------+
# |     genres|label|                      features|                   probability|prediction|
# +-----------+-----+------------------------------+------------------------------+----------+
# |   Pop_Rock|  0.0|[5.5345453470351895,1.06155...|[0.9655663761891948,0.00888...|       0.0|
# |   Pop_Rock|  0.0|[4.985728699493991,1.232366...|[0.938916866997794,0.015058...|       0.0|
# |   Pop_Rock|  0.0|[4.172552483700552,1.123692...|[0.9379245190770236,0.01459...|       0.0|
# | Electronic|  1.0|[10.319298936161982,2.84960...|[0.9351055118133464,0.02223...|       0.0|
# |   Pop_Rock|  0.0|[5.573194406721189,1.064838...|[0.9256322203636289,0.01833...|       0.0|
# | Electronic|  1.0|[7.198000875920627,2.641019...|[0.925010639116745,0.022428...|       0.0|
# |   Pop_Rock|  0.0|[7.092875433574707,2.200575...|[0.923547322119503,0.022387...|       0.0|
# |   Pop_Rock|  0.0|[5.2810075154950304,1.24358...|[0.9212237300696415,0.01913...|       0.0|
# |Avant_Garde| 17.0|[3.357830305519674,1.466685...|[0.9183981824919586,0.01808...|       0.0|
# |   Pop_Rock|  0.0|[4.5961461778591115,1.17433...|[0.9181019213450791,0.01929...|       0.0|
# +-----------+-----+------------------------------+------------------------------+----------+



evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator.evaluate(predictions)		# Accuracy 0.410

predictionRDD = predictions.select(['label', 'prediction']) \
                        .rdd.map(lambda line: (line[1], line[0]))
metrics = MulticlassMetrics(predictionRDD)

print('recall = {:.3f} '.format(metrics.recall()))	#0.567
print('precision = {:.3f} '.format(metrics.precision()))	#0.567

# change threshold  (not useful)
lr_predictions = lrModel.transform(test, {lrModel.threshold:0.05})

predictionRDD = lr_predictions.select(['label', 'prediction']).rdd.map(lambda line: (line[1], line[0]))
metrics = MulticlassMetrics(predictionRDD)

print('recall = {:.3f} '.format(metrics.recall()))	#0.564
print('precision = {:.3f} '.format(metrics.precision()))	#0.564


## Cross Validation with LR
lr = LogisticRegression(maxIter=20, regParam=0.1, elasticNetParam=0, family="multinomial")

# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.1, 0.3, 0.5]) # regularization parameter
             .addGrid(lr.elasticNetParam, [0.01, 0.1, 0.2]) # Elastic Net Parameter (Ridge = 0)
            .addGrid(lr.maxIter, [10, 20]) #Number of iterations
             .build())
# Create 5-fold CrossValidator
cv = CrossValidator(estimator=lr, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, \
                    numFolds=5,
                    parallelism=2)
cv_lr_multi_model = cv.fit(train)

cv_lr_predictions = cv_lr_multi_model.transform(test)


#cv_lr_multi_model.save('hdfs:///user/ych192/outputs/mds/cv_lr_multi_model')

cv_lr_multi_model = LogisticRegressionModel.load('hdfs:///user/ych192/outputs/mds/cv_lr_multi_model')


# Evaluate best model(

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator.evaluate(cv_lr_predictions)		# Accuracy 0.4235

predictionRDD = cv_lr_predictions.select(['label', 'prediction']) \
                        .rdd.map(lambda line: (line[1], line[0]))
metrics = MulticlassMetrics(predictionRDD)

print('recall = {:.3f} '.format(metrics.recall()))	#0.566
print('precision = {:.3f} '.format(metrics.precision()))	#0.566



print('recall = {:.3f} '.format(metrics.recall(0)))		#0.986

# In [173]: print('recall = {:.3f} '.format(metrics.recall(0)))
# recall = 0.986

# In [174]: print('recall = {:.3f} '.format(metrics.recall(1)))
# recall = 0.058

# In [175]: print('recall = {:.3f} '.format(metrics.recall(2)))
# recall = 0.014

# In [176]: print('recall = {:.3f} '.format(metrics.recall(3)))
# recall = 0.017

# In [177]: print('recall = {:.3f} '.format(metrics.recall(4)))
# recall = 0.000

evaluator.evaluate(cv_lr_predictions, {evaluator.metricName: "accuracy"})