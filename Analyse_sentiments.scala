import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{Tokenizer, StopWordsRemover, CountVectorizer, HashingTF, IDF, VectorAssembler}
import org.apache.spark.ml.classification.LogisticRegression

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

val cleanedTweetsDF = tweetsDF.withColumn("cleaned_text", nettoyerUDF(col("selected_text")))

val tokenizer = new Tokenizer().setInputCol("cleaned_text").setOutputCol("tokens")
val tokenizedDF = tokenizer.transform(cleanedTweetsDF)

val remover = new StopWordsRemover()
  .setInputCol("tokens")
  .setOutputCol("filtered_tokens")

val filteredDF = remover.transform(tokenizedDF)

val hashingTF = new HashingTF().setInputCol("filtered_tokens").setOutputCol("raw_features").setNumFeatures(5000)
val featurizedDF = hashingTF.transform(filteredDF)

val idf = new IDF().setInputCol("raw_features").setOutputCol("features")
val idfModel = idf.fit(featurizedDF)
val rescaledDF = idfModel.transform(featurizedDF)

val updatedDF = tweetsDF.withColumn("sentiment_label",
  when(col("sentiment") === "positive", 1)
    .when(col("sentiment") === "negative", 0)
    .when(col("sentiment") === "neutral", 2)
    .otherwise(3)
)

val finalDF = rescaledDF.join(updatedDF.select("sentiment_label", "selected_text"), Seq("selected_text"))

val assembler = new VectorAssembler()
  .setInputCols(Array("features"))
  .setOutputCol("final_features")

val finalDataDF = assembler.transform(finalDF)

val lr = new LogisticRegression().setLabelCol("sentiment_label").setFeaturesCol("final_features")
val lrModel = lr.fit(finalDataDF)

val predictions = lrModel.transform(finalDataDF)
predictions.select("selected_text", "prediction").show(5)



val newTweetsDF = spark.read.option("header", "true").csv("Data4.csv")
val cleanedNewTweetsDF = newTweetsDF.withColumn("cleaned_text", nettoyerUDF(col("text")))
val tokenizedNewTweetsDF = tokenizer.transform(cleanedNewTweetsDF)
val finalNewTweetsDF = remover.transform(tokenizedNewTweetsDF)

val featurizedNewTweetsDF = hashingTF.transform(finalNewTweetsDF)
val rescaledNewTweetsDF = idfModel.transform(featurizedNewTweetsDF)

val assemblerNew = new VectorAssembler()
  .setInputCols(Array("features"))
  .setOutputCol("final_features")

val finalNewData = assemblerNew.transform(rescaledNewTweetsDF)
val predictionsNew = lrModel.transform(finalNewData)

predictionsNew.select("text", "cleaned_text", "prediction","sentiment").show(20)



val totalTweets = predictions.count()
val positiveTweets = predictions.filter(col("prediction") === 1).count()
val negativeTweets = predictions.filter(col("prediction") === 0).count()
val positivePercentage = (positiveTweets.toDouble / totalTweets) * 100
val negativePercentage = (negativeTweets.toDouble / totalTweets) * 100
println(s"Pourcentage de tweets positifs : $positivePercentage%")
println(s"Pourcentage de tweets négatifs : $negativePercentage%")
