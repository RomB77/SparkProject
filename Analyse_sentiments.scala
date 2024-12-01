import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.StopWordsRemover 
import org.apache.spark.ml.feature.CountVectorizer

val spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()
val tweetsDF = spark.read.option("header", "true").csv("Data3.csv")
tweetsDF.show(5)

val nettoyerUDF = udf((text: String) => {
  if (text != null) {
	  var cleanedText = text.replaceAll("http[\\S]+", "")
	  cleanedText = cleanedText.replaceAll("@\\w+", "")
	  cleanedText = cleanedText.replaceAll("#\\w+", "")
	  cleanedText = cleanedText.replaceAll("[^a-zA-Z\\s]", "")
	  cleanedText = cleanedText.toLowerCase()
	  cleanedText
  } else ""
})
 
val tweetsDF = spark.read.option("header", "true").csv("Data3.csv")
val cleanedTweetsDF = tweetsDF.withColumn("cleaned_text", nettoyerUDF(col("selected_text")))
cleanedTweetsDF.select("selected_text", "cleaned_text").show(5)

val tokenizer = new Tokenizer().setInputCol("cleaned_text").setOutputCol("tokens") 
val tokenizedDF = tokenizer.transform(cleanedTweetsDF)
tokenizedDF.select("cleaned_text", "tokens").show(5)

val remover = new StopWordsRemover()
  .setInputCol("tokens") 
  .setOutputCol("filtered_tokens") 
 
val finalDF = remover.transform(tokenizedDF)
finalDF.select("tokens", "filtered_tokens").show(5)

val vectorizer = new CountVectorizer().setInputCol("filtered_tokens").setOutputCol("features")
val vectorizedDF = vectorizer.fit(finalDF).transform(finalDF)
vectorizedDF.select("filtered_tokens", "features").show(5)

val updatedDF = tweetsDF.withColumn("sentiment_label", 
  when(col("sentiment") === "positive", 1)
  .when(col("sentiment") === "negative", 0)
  .when(col("sentiment") === "neutral", 2)
  .otherwise(-1))

updatedDF.select("sentiment", "sentiment_label").show(5)
