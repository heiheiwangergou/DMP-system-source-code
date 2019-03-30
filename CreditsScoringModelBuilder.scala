package creditscoring

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by zxk on 2017/12/5.
 */
class CreditsScoringModelBuilder {

}
object CreditsScoringModelBuilder{
   def main (args: Array[String]){

     val sparkConf: SparkConf = new SparkConf().setAppName("creditsScoringModelBuilderTest")
     val sc: SparkContext = new SparkContext(sparkConf)
     val sqlContext = new SQLContext(sc)
    // val hiveContext: HiveContext = new HiveContext(sc)

     import org.apache.spark.sql.SQLContext
     import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType}

     // val sqlContext = new SQLContext(sc)
     val customSchema = StructType(Array(
       StructField("creditability", IntegerType, true),
       StructField("balance", IntegerType, true),
       StructField("duration", IntegerType, true),
       StructField("history", IntegerType, true),
       StructField("purpose", IntegerType, true),
       StructField("amount", IntegerType, true),
       StructField("savings", IntegerType, true),
       StructField("employment", IntegerType, true),
       StructField("instalment", IntegerType, true),
       StructField("sexmarital", IntegerType, true),
       StructField("guarantors", IntegerType, true),
       StructField("residenceduration", IntegerType, true),
       StructField("asset", IntegerType, true),
       StructField("age", IntegerType, true),
       StructField("concredits", IntegerType, true),
       StructField("apartment", IntegerType, true),
       StructField("credits", IntegerType, true),
       StructField("occupation", IntegerType, true),
       StructField("dependents", IntegerType, true),
       StructField("telephone", IntegerType, true),
       StructField("foreign", IntegerType, true)))

     val creditDf = sqlContext.read
    .format("com.databricks.spark.csv")
    .option("header", "true") // Use first line of all files as header
    .schema(customSchema)
       .load("german_credit_data.csv")

     creditDf.registerTempTable("credit")
     creditDf.printSchema()
     sqlContext.sql("select Creditability ,avg(balance)as avg_balance ," +
       " avg(amount) as avg_amount from credit group by Creditability ")

     val featureCol=Array("creditability","balance", "duration", "history", "purpose",
       "amount", "savings", "employment", "instalment", "sexmarital", "guarantors", "residenceduration",
       "asset", "age", "concredits", "apartment", "credits", "occupation", "dependents", "telephone", "foreign")
     val assembler = new VectorAssembler()
       .setInputCols(featureCol)
       .setOutputCol("features")

     val df2=assembler.transform(creditDf)
     val indexer = new StringIndexer()
       .setInputCol("Creditability")
       .setOutputCol("lable")

     val indexed = indexer.fit(df2).transform(df2)
     //val output = assembler.transform(dataset)
     val splitSeed=5043
     val Array(trainingData, testData) = indexed.randomSplit(Array(0.7, 0.3))

     val rf = new RandomForestClassifier()
       .setLabelCol("indexedLabel")
       .setFeaturesCol("indexedFeatures")
       .setNumTrees(10)

     val model = rf.fit(trainingData)
     // Make predictions.
     val predictions = model.transform(testData)

     // Select example rows to display.
     predictions.select("predictedLabel", "label", "features").show(5)

     // Select (prediction, true label) and compute test error.
     val evaluator = new MulticlassClassificationEvaluator()
       .setLabelCol("indexedLabel")
       .setPredictionCol("prediction")
       .setMetricName("accuracy")
     val accuracy = evaluator.evaluate(predictions)
     println("Test Error = " + (1.0 - accuracy))
  }


}