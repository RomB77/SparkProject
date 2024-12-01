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
  .otherwise(3))

updatedDF.select("sentiment", "sentiment_label").show(5)

// Préparation des données pour l'entraînement du modèle
val labeledDF = vectorizedDF.join(updatedDF, Seq("textID"))  // On suppose que chaque tweet a un identifiant unique (textID)
val assembler = new VectorAssembler().setInputCols(Array("features")).setOutputCol("assembledFeatures")
val trainingData = assembler.transform(labeledDF)

// Définition du modèle Logistic Regression
val lr = new LogisticRegression().setFeaturesCol("assembledFeatures").setLabelCol("sentiment_label")

// Entraînement du modèle
val lrModel = lr.fit(trainingData)

// Prédiction sur les données d'entraînement
val predictions = lrModel.transform(trainingData)
predictions.select("textID", "cleaned_text", "sentiment_label", "prediction").show(5)
