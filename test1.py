from pyspark import SparkContext

dataFile = "/Users/yiweizhang/Desktop/CSCI5408/assignment3/spark/gitA3/CSCI5408A3/stops.csv"
sc = SparkContext("spark://Yiweis-MacBook-Pro.local:7077", "Simple App")
a = sc.textFile(dataFile)
print ("Count of records: ", a.count())
print ("Count of 2005 records: ", a.filter(lambda x: 'c' in x).count())
sc.stop()