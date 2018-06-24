from pyspark.ml import PipelineModel
from pyspark.ml.feature import CountVectorizer
from pyspark.sql.functions import col
from pyspark.sql import SQLContext, SparkSession
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from collections import namedtuple

# from pyspark.sql.functions import desc

sc = SparkContext("local[2]", "Streaming App")

ssc = StreamingContext(sc, 10)
sqlContext = SQLContext(sc)

socket_stream = ssc.socketTextStream("127.0.0.1", 5555)  # Internal ip of  the tweepy streamer

lines = socket_stream.window(20)
fields = "SentimentText"
Tweet = namedtuple('Tweet', fields)


def getSparkSessionInstance(sparkConf):
    if "sparkSessionSingletonInstance" not in globals():
        globals()["sparkSessionSingletonInstance"] = SparkSession \
            .builder \
            .config(conf=sparkConf) \
            .getOrCreate()
    return globals()["sparkSessionSingletonInstance"]


def do_something(time, rdd):
    print("========= %s =========" % str(time))
    try:
        # Get the singleton instance of SparkSession
        spark = getSparkSessionInstance(rdd.context.getConf())

        # Convert RDD[String] to RDD[Tweet] to DataFrame
        rowRdd = rdd.map(lambda w: Tweet(w))
        linesDataFrame = spark.createDataFrame(rowRdd)
        pipelineModel = PipelineModel.load("logreg.model")
        prediction = pipelineModel.transform(linesDataFrame)
        print("########################################################3")
        #prediction.show(5)
        #result = prediction.select("SentimentText", "probability", "prediction")
        #print("result"+"======================="+type(result))
        type(prediction)
        # print("########################################################3")
        # print(result.collect())

        # Creates a temporary view using the DataFrame
        # result.collect().createOrReplaceTempView("tweets")
        # result.createOrReplaceTempView("tweets")

        # Do tweet character count on table using SQL and print it
        # lineCountsDataFrame = spark.sql("select SentimentText,probability,prediction from tweets")
        # lineCountsDataFrame = spark.sql("select * from tweets")
        # lineCountsDataFrame.show()
        # lineCountsDataFrame.coalesce(1).write.format("com.databricks.spark.csv").save("dirwithcsv")
        # lineCountsDataFrame.write.coalesce(1).format("com.databricks.spark.csv").mode("append").save("dirwithcsv.csv")
        # lineCountsDataFrame.write.format("com.databricks.spark.csv").save("dirwithcsv")
        result.write.format("com.databricks.spark.csv").save("dirwithcsv")

        # predictions_pdf = prediction.toPandas()
    except:
        pass


# key part!
lines.foreachRDD(do_something)

ssc.start()
ssc.awaitTermination()
