import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.{Tokenizer}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

object SentimentClassification {
  def writeMetricsToFile(predictionAndLabels: RDD[(Double, Double)],printContent: StringBuilder,modelName: String ) : StringBuilder = {

    // Instantiate Multimetrics object
    val multimetrics = new MulticlassMetrics(predictionAndLabels)
    printContent.append("\n")
    // Confusion matrix
    printContent.append("Classification Model: "+modelName+"\n")
    printContent.append("Confusion matrix:")
    printContent.append("\n")
    printContent.append(multimetrics.confusionMatrix)
    printContent.append("\n")
    // High Level Statistics
    val accuracy = multimetrics.accuracy
    printContent.append("Summary Statistics")
    printContent.append("\n")
    printContent.append(s"Accuracy = $accuracy")
    printContent.append("\n")

    val labels = multimetrics.labels
    // FPR
    labels.foreach { l =>
      printContent.append(s"FPR($l) = " + multimetrics.falsePositiveRate(l))
      printContent.append("\n")
    }

    // F-measure by label
    labels.foreach { l =>
      printContent.append(s"F1-Score($l) = " + multimetrics.fMeasure(l))
      printContent.append("\n")
    }
    // Precision

    labels.foreach { l =>
      printContent.append(s"Precision($l) = " + multimetrics.precision(l))
      printContent.append("\n")
    }

    // Recall
    labels.foreach { l =>
      printContent.append(s"Recall($l) = " + multimetrics.recall(l))
      printContent.append("\n")
    }


    // Weighted statistics
    printContent.append(s"Precision: ${multimetrics.weightedPrecision}")
    printContent.append("\n")
    printContent.append(s"Recall: ${multimetrics.weightedRecall}")
    printContent.append("\n")
    printContent.append(s" F1 score: ${multimetrics.weightedFMeasure}")
    printContent.append("\n")
    printContent.append(s"False positive rate: ${multimetrics.weightedFalsePositiveRate}")
    printContent.append("\n")
    return printContent
  }
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.appName("Simple App").getOrCreate()
    if (args.length != 2) {
      println("Provide Input Output filepath")
    }
    Logger.getLogger("labAsst").setLevel(Level.OFF)
    spark.sparkContext.setLogLevel("ERROR")
    import spark.implicits._
    val tweets = spark.read.option("header","true")
      .csv(args(0))
    val cols = Array("text")
    val filteredTweets = tweets.na.drop(cols)
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")

    val remover = new StopWordsRemover().setInputCol(tokenizer.getOutputCol).setOutputCol("filtered")

    val hashingTF = new HashingTF().setInputCol(remover.getOutputCol).setOutputCol("features")

    val indexer = new StringIndexer().setInputCol("airline_sentiment").setOutputCol("categoryIndex")

    val pipeline = new Pipeline().setStages(Array(tokenizer, remover,hashingTF, indexer))

    val preProcessedData = pipeline.fit(filteredTweets)
    val tweetPreProcessedData = preProcessedData.transform(filteredTweets)
    val Array(training, test) = tweetPreProcessedData.randomSplit(Array(0.8, 0.2))
    val lr = new LogisticRegression().setMaxIter(10).setLabelCol("categoryIndex").setFeaturesCol("features")

    val paramGrid_logistic = new ParamGridBuilder().addGrid(hashingTF.numFeatures, Array(10, 50, 100)).addGrid(lr.regParam, Array(0.1,0.3)).build()

    val lr_evaluate = new MulticlassClassificationEvaluator().setLabelCol("categoryIndex").setPredictionCol("prediction").setMetricName("accuracy")

    val lr_crossvalidate = new CrossValidator().setEstimator(lr).setEvaluator(lr_evaluate).setEstimatorParamMaps(paramGrid_logistic).setNumFolds(3)

    val lr_model = lr_crossvalidate.fit(training)

    val lr_prediction = lr_model.transform(test)
    //convert dataset to RDD[(Double,Double)]
    val lr_result = lr_prediction.select("categoryIndex","prediction").map{ case Row(l:Double,p:Double) => (l,p) }

    val lr_predictionLabels = lr_result.rdd
    var printContent = new StringBuilder()
    printContent = writeMetricsToFile(lr_predictionLabels,printContent,"Logistic Regression")

    val bayes = new NaiveBayes().setLabelCol("categoryIndex").setFeaturesCol("features")

    val paramGrid_NaiveBayes = new ParamGridBuilder().addGrid(hashingTF.numFeatures, Array(20,60,150)).addGrid(bayes.smoothing,Array(0.1,0.2,0.3)).build()

    val bm_evaluate = new MulticlassClassificationEvaluator().setLabelCol("categoryIndex").setPredictionCol("prediction").setMetricName("accuracy")

    val bm_crossvalidate = new CrossValidator().setEstimator(bayes).setEvaluator(bm_evaluate).setEstimatorParamMaps(paramGrid_NaiveBayes).setNumFolds(3)

    val bmModel = bm_crossvalidate.fit(training)

    val bm_prediction = bmModel.transform(test)
    //convert dataset to RDD[(Double,Double)]
    val bm_result = bm_prediction.select("categoryIndex","prediction").map{ case Row(l:Double,p:Double) => (l,p) }

    val bm_predictionLabels = bm_result.rdd

    printContent = writeMetricsToFile(bm_predictionLabels,printContent,"Naive Bayes")
    val printRdd = spark.sparkContext.parallelize(Seq(printContent))
    printRdd.saveAsTextFile(args(1))
  }
}