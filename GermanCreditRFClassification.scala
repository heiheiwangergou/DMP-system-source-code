package creditscoring

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.immutable.Map
import scala.util.control.NonFatal

object GermanCreditRFClassification extends App {
  val sparkConf = new SparkConf()
  sparkConf.setAppName("localApp")
  sparkConf.setMaster("local")
  val sc = new SparkContext(sparkConf)

  def prepareData(): RDD[LabeledPoint] = {
    val creditText = scala.io.Source.fromFile("/home/yarn/sparkdatas/german_credit.csv").mkString
    val allLines = creditText.split("\n")
    val recordsOnly = allLines.slice(1, allLines.size-1)
    val allDataset = sc.parallelize(recordsOnly).map{
      case line =>
        val arr = line.split(",").map(t=>t.trim()).map(p=>p.toDouble)
        new LabeledPoint(arr(0).asInstanceOf[Int],Vectors.dense(arr.slice(1,arr.length)))
    }
    allDataset
  }

  val trainingData: RDD[LabeledPoint] = prepareData()

  val split = trainingData.randomSplit(List(0.8, 0.2).toArray)
  val trainingSet = split(0)
  val testSet = split(1)
  val numClasses = 2
  val categoricalFeaturesInfo = Map[Int, Int]()
  val numTrees = 5
  val featureSubsetStrategy = "auto"
  val impurity = "gini"
  val maxDepth = 6
  val maxBins = 32

  val model: RandomForestModel = RandomForest.trainClassifier(
    trainingSet, numClasses, categoricalFeaturesInfo,
    numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

  val testData = testSet.map(labeledPoint => labeledPoint.features)
  val gt = testSet.map(labeledPoint => labeledPoint.label)
  val result = model.predict(testData)

  val unmatchedPredictions = gt.zip(result).map(pair => pair._1 - pair._2).filter(_ != 0d).count()
  val matchedPredictions = gt.zip(result).map(pair => pair._1 - pair._2).filter(_ == 0d).count()

  val predictionAndLabels = gt.zip(result)

  // Instantiate metrics object
  val metrics = new BinaryClassificationMetrics(predictionAndLabels)

  // Precision by threshold
  val precision = metrics.precisionByThreshold
  precision.foreach { case (t, p) =>
    println(s"Threshold: $t, Precision: $p")

  }

  // Recall by threshold
  val recall = metrics.recallByThreshold
  recall.foreach { case (t, r) =>
    println(s"Threshold: $t, Recall: $r")
  }

  // Precision-Recall Curve
  val PRC = metrics.pr

  // F-measure
  val f1Score = metrics.fMeasureByThreshold
  f1Score.foreach { case (t, f) =>
    println(s"Threshold: $t, F-score: $f, Beta = 1")
  }

  val beta = 0.5
  val fScore = metrics.fMeasureByThreshold(beta)
  f1Score.foreach { case (t, f) =>
    println(s"Threshold: $t, F-score: $f, Beta = 0.5")
  }

  // AUPRC
  val auPRC = metrics.areaUnderPR
  println("Area under precision-recall curve = " + auPRC)

  // Compute thresholds used in ROC and PR curves
  val thresholds = precision.map(_._1)

  // ROC Curve
  val roc = metrics.roc

  // AUROC
  val auROC = metrics.areaUnderROC
  println("Area under ROC = " + auROC)

  println(model.toDebugString)

  println(matchedPredictions)
  println(unmatchedPredictions)

}
