from pyspark import SparkContext

dataFile = "stops.csv"
sc = SparkContext("spark://ip-172-31-46-37.us-east-2.compute.internal:7077", "Simple App")
a = sc.textFile(dataFile)
print ("Count of records: ", a.count())
print ("Count of 2005 records: ", a.filter(lambda x: 'c' in x).count())
sc.stop()
