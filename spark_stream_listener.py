from pyspark import SparkContext, SQLContext
from pyspark.streaming import StreamingContext
# from pyspark.sql.functions import desc

sc = SparkContext("local[2]", "Tweet Streaming App")


ssc = StreamingContext(sc, 10)
sqlContext = SQLContext(sc)
ssc.checkpoint( "file:/Users/yiweizhang/Desktop/CSCI5408/assignment3/spark/gitA3/CSCI5408A3/checkpoint/")

socket_stream = ssc.socketTextStream("127.0.0.1", 5555) # Internal ip of  the tweepy streamer

lines = socket_stream.window(20)

lines.count().map(lambda x:'Tweets in this batch: %s' % x).pprint()
# If we want to filter hashtags only
# .filter( lambda word: word.lower().startswith("#") )
words = lines.flatMap( lambda twit: twit.split(" ") )
pairs = words.map( lambda word: ( word.lower(), 1 ) )
wordCounts = pairs.reduceByKey( lambda a, b: a + b ) #.transform(lambda rdd:rdd.sortBy(lambda x:-x[1]))
#wordCounts.pprint()
words.pprint()

ssc.start()
ssc.awaitTermination()
