import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{Tokenizer, StopWordsRemover, CountVectorizer, HashingTF, IDF, VectorAssembler}
import org.apache.spark.ml.classification.LogisticRegression

val spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()

val tweetsDF = spark.read.option("header", "true").csv("H:/Documents/Ing2/S1/Programmation fonctionnelle/Projet/Data3.csv")
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

val updatedDF = tweetsDF.withColumn("sentiment_label",
  when(col("sentiment") === "positive", 2)
    .when(col("sentiment") === "negative", 0)
    .when(col("sentiment") === "neutral", 1)
    .otherwise(3)
)
val Array(trainingTweets, testTweets) = filteredDF.randomSplit(Array(0.8, 0.2), seed = 42)
trainingTweets.show(5)
testTweets.show(5)

val hashingTF = new HashingTF().setInputCol("filtered_tokens").setOutputCol("raw_features").setNumFeatures(5000)
val featurizedDF = hashingTF.transform(trainingTweets)

val idf = new IDF().setInputCol("raw_features").setOutputCol("features")
val idfModel = idf.fit(featurizedDF)
val rescaledDF = idfModel.transform(featurizedDF)

val finalDF = rescaledDF.join(updatedDF.select("sentiment_label", "selected_text").dropDuplicates("selected_text"), "selected_text") 
val recoveredDF = rescaledDF.join(updatedDF.select("sentiment_label", "selected_text"), Seq("selected_text"), "left")
val finalDFWithRecovered = finalDF.union(recoveredDF.filter(col("sentiment_label").isNull))

val assembler = new VectorAssembler()
  .setInputCols(Array("features"))
  .setOutputCol("final_features")

val finalDataDF = assembler.transform(finalDF)

val lr = new LogisticRegression().setLabelCol("sentiment_label").setFeaturesCol("final_features")
val lrModel = lr.fit(finalDataDF)

val predictions = lrModel.transform(finalDataDF)
predictions.select("selected_text", "prediction").show(5)


val predictionsWithLabels = predictions.select(col("selected_text"), col("sentiment_label"), col("prediction").cast("int"))
val confusionMatrixDF = predictionsWithLabels.groupBy("sentiment_label", "prediction").count()
confusionMatrixDF.show()
val pivotDF = confusionMatrixDF.groupBy("sentiment_label").pivot("prediction").sum("count").na.fill(0)
val orderedPivotDF = pivotDF.orderBy(expr("CASE WHEN sentiment_label = 0 THEN 0 WHEN sentiment_label = 1 THEN 1 ELSE 2 END"))
orderedPivotDF.show()
val totalCorrect = predictionsWithLabels.filter("sentiment_label = prediction").count()
val total = predictionsWithLabels.count()
val accuracy = totalCorrect.toDouble / total * 100
println(s"Précision du modèle : $accuracy %")

val featurizedNewTweetsDF = hashingTF.transform(testTweets)
val idfModelNew = idf.fit(featurizedNewTweetsDF)
val rescaledNewTweetsDF = idfModel.transform(featurizedNewTweetsDF)

val finalDFNew = rescaledNewTweetsDF.join(updatedDF.select("sentiment", "selected_text"), Seq("selected_text"))

val assemblerNew = new VectorAssembler()
  .setInputCols(Array("features"))
  .setOutputCol("final_features")

val finalNewData = assemblerNew.transform(rescaledNewTweetsDF)
val predictionsNew = lrModel.transform(finalNewData)

predictionsNew.select("selected_text", "cleaned_text", "prediction","sentiment").show(20)

val sentiment_labelAdded = predictionsNew.withColumn("sentiment_label",
  when(col("sentiment") === "positive", 2)
    .when(col("sentiment") === "negative", 0)
    .when(col("sentiment") === "neutral", 1)
    .otherwise(3)
)

val predictionsWithLabelsNew = sentiment_labelAdded.select(col("text"), col("sentiment_label"), col("prediction").cast("int"))
val confusionMatrixDFNew = predictionsWithLabelsNew.groupBy("sentiment_label", "prediction").count()
confusionMatrixDFNew.show()
val pivotDFNew = confusionMatrixDFNew.groupBy("sentiment_label").pivot("prediction").sum("count").na.fill(0)
val orderedPivotDFNew = pivotDFNew.orderBy(expr("CASE WHEN sentiment_label= 0 THEN 0 WHEN sentiment_label = 1 THEN 1 ELSE 2 END"))
orderedPivotDFNew.show()
val totalCorrectNew = predictionsWithLabelsNew.filter("sentiment_label = prediction").count()
val totalNew = predictionsWithLabelsNew.count()
val accuracyNew = totalCorrectNew.toDouble / totalNew * 100
println(s"Précision du modèle : $accuracyNew %")

val totalTweets = predictionsNew.count()
val positiveTweets = predictionsNew.filter(col("prediction") === 1).count()
val negativeTweets = predictionsNew.filter(col("prediction") === 0).count()
val positivePercentage = (positiveTweets.toDouble / totalTweets) * 100
val negativePercentage = (negativeTweets.toDouble / totalTweets) * 100
