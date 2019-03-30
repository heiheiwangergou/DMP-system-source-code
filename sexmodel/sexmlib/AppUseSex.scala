package sexmodel.sexmlib

import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature.{StandardScaler, StringIndexer, VectorIndexer}
import org.apache.spark.ml.{Pipeline, PipelineStage, Transformer}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable

/**
 * Created by pc on 2015/9/11.
 */
class AppUseSex {

}

object AppUseSex {
  def main(args: Array[String]) {
    val sc = new SparkContext(new SparkConf().setAppName("we"))
    sc.hadoopConfiguration.set("mapred.output.compress", "false")

    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    val hiveContext = new HiveContext(sc)
    val pathsave = "/tmp/xxx/sex/01/"
    val path = "/tmp/xxx/sex/01/"
    val tainDataDF = getKnownSex(hiveContext, pathsave)
    tainDataDF.count()
    // tainDataDF.write.
    val trandatas = tainDataDF.map {
      case Row(lable: Int, features: String) =>
        (lable, features)
    }
    val trandatas2 = trandatas.filter {
      line =>
        if (line._2 != null && line._2.nonEmpty) true else false
    }.map {
      line =>
        line._1 + " " + line._2
    }
    println(s"train data====$tainDataDF.count()")
    trandatas2.saveAsTextFile(pathsave)
    trandatas2.count()
    //train_data
    val datas = loadDatasets(sc, pathsave).toDF()
   //val datas=trainingdrops
    val stages = new mutable.ArrayBuffer[PipelineStage]()
    val lableIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLablel").fit(datas)
    //lableIndexer.t
    stages += lableIndexer
    /* val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(datas)
     stages += featureIndexer*/
    //标准化 操作
    val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(false)
    //正则化 特征值
    val scalerModel = scaler.fit(datas)
    stages += scalerModel
    //数据划分
    //trainingdrop  0.8作为训练 0.2作为测试
    val splits = datas.randomSplit(Array(0.8, 0.2))
    val training = splits(0)
    val test = splits(1)
/*    test.show(10)
    val test2 = datas.map {
      case Row(label: Double, features: Vector) =>
        label
    }.countByValue()*/
     val sexModel = new DecisionTreeClassifier().setFeaturesCol("scaledFeatures").setLabelCol("indexedLablel").setMaxDepth(15).setMaxBins(100)
    //sexModel.tr
     //val sexModel= new LogisticRegression().setFeaturesCol("scaledFeatures").setLabelCol("indexedLablel").setRegParam(0.01)setMaxIter(30)
    // val sexModel = new DecisionTreeClassifier().setFeaturesCol("features").setLabelCol("label").setMaxDepth(10).setMaxBins(10)
    //datas.printSchema()
    stages += sexModel
    val pipline = new Pipeline().setStages(stages.toArray)
    val startime = System.nanoTime()

    val pipelineModel = pipline.fit(training)
    training.printSchema()
    // Make predictions.
    val predictions = pipelineModel.transform(test)

    predictions.printSchema()
    // Select example rows to display.
    predictions.select("prediction", "indexedLablel", "features").show(100)
    //    val pipelineModel = pipline.fit(testdata)
    training.printSchema()
    val elapsedTime = (System.nanoTime() - startime) / 1e9
    println(s"Training time: $elapsedTime seconds")

    val sexModelLR = pipelineModel.stages.last.asInstanceOf[DecisionTreeClassificationModel]
    println("Learned classification tree model:\n" + sexModelLR.toDebugString)


    val aucc = evaluateModel(pipelineModel, test, "indexedLablel")
    val auc = aucc._1
    val acc = aucc._2
    println(s"Model AUC: $auc\t Model ACC $acc")
    val aucc2 = evaluateModelFM(pipelineModel, test, "indexedLablel")

    aucc2.map { line =>
      if (line._1 == line._2) 1 else 0
    }.sum / aucc2.count()

    training.count() //51283   64220   51058  51051  50847
    //dropMaleErroDF(sc,)
    var trainingdrops = training
    var modeldrops = pipelineModel
    var num=0
    while(num<2){
      //var ss = dropMaleErroDF(sc, modeldrops, trainingdrops, "indexedLablel")
      var sslr = dropMaleErroDFLR(sc, modeldrops, trainingdrops, "indexedLablel")
      val sslr2=dropMaleErroDFLR2(sc, modeldrops, trainingdrops, "indexedLablel")
      println("=========="+  trainingdrops.count())
      trainingdrops.unpersist(blocking = false)
       trainingdrops = sslr._1
       modeldrops = sslr._2
      num+=1
    }
      trainingdrops.count()

    val aucc22 = evaluateModelFM(modeldrops, trainingdrops, "indexedLablel")
    aucc22.map { line =>
      if (line._1 == line._2) 1 else 0
    }.sum / aucc22.count()
  }


  def dropMaleErroDFLR2(sc: SparkContext,
                       model: Transformer,
                       data: DataFrame,
                       lableColName: String) = {
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    val testdataPrediction = model.transform(data)
    //    val testdataPredictionMale=testdataPrediction.where(($"indexedLablel"===0.0) &&($"prediction"!==$"indexedLablel"))
   //取一半男的数据
    val testdataPredictionMale2 = testdataPrediction.where(($"indexedLablel" === 0.0) && ($"prediction" === $"indexedLablel")).select("label", "features").randomSplit(Array(0.3,0.7))

    val testdataPredictionMale =testdataPredictionMale2(0)
    val testdataPredictionFMale = testdataPrediction.where(($"indexedLablel" === 1.0)).select("label", "features")

    val trainingdrop = testdataPredictionMale.unionAll(testdataPredictionFMale).repartition(400)
   /* trainingdrop.count()//50914  50847
    //val dd=trainingdrop.repartition(400)
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLablel").fit(trainingdrop)
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").fit(trainingdrop)
    // val sss = splits2(0).show(10)
    val Array(trainingData, testData) = trainingdrop.randomSplit(Array(0.8, 0.2))
    // Train a DecisionTree model.
    val lr= new LogisticRegression().setFeaturesCol("indexedFeatures").setLabelCol("indexedLablel").setRegParam(0.01)setMaxIter(10)
    val sexModel = new DecisionTreeClassifier().setFeaturesCol("indexedFeatures").setLabelCol("indexedLablel").setMaxDepth(15).setMaxBins(20)

    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, lr))
    val modeldroplr = pipeline.fit(trainingdrop)*/
    // val dropdata=testdataPredictionMale.unionAll(testdataPredictionFMale)
    (trainingdrop)
  }

  def dropMaleErroDFLR(sc: SparkContext,
                     model: Transformer,
                     data: DataFrame,
                     lableColName: String) = {
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    val testdataPrediction = model.transform(data)
    //    val testdataPredictionMale=testdataPrediction.where(($"indexedLablel"===0.0) &&($"prediction"!==$"indexedLablel"))
    val testdataPredictionMale = testdataPrediction.where(($"indexedLablel" === 0.0) && ($"prediction" === $"indexedLablel")).select("label", "features")
    val testdataPredictionFMale = testdataPrediction.where(($"indexedLablel" === 1.0)).select("label", "features")

    val trainingdrop = testdataPredictionMale.unionAll(testdataPredictionFMale).repartition(400)
    trainingdrop.count()//50914  50847
    //val dd=trainingdrop.repartition(400)
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLablel").fit(trainingdrop)
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").fit(trainingdrop)
    // val sss = splits2(0).show(10)
    val Array(trainingData, testData) = trainingdrop.randomSplit(Array(0.8, 0.2))
    // Train a DecisionTree model.
    val lr= new LogisticRegression().setFeaturesCol("indexedFeatures").setLabelCol("indexedLablel").setRegParam(0.01)setMaxIter(10)
    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, lr))
    val modeldroplr = pipeline.fit(trainingdrop)
    // val dropdata=testdataPredictionMale.unionAll(testdataPredictionFMale)
    (trainingdrop, modeldroplr)
  }
  def dropMaleErroDF(sc: SparkContext,
                     model: Transformer,
                     data: DataFrame,
                     lableColName: String) = {
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    val testdataPrediction = model.transform(data)
    //    val testdataPredictionMale=testdataPrediction.where(($"indexedLablel"===0.0) &&($"prediction"!==$"indexedLablel"))
    val testdataPredictionMale = testdataPrediction.where(($"indexedLablel" === 0.0) && ($"prediction" === $"indexedLablel")).select("label", "features")
    val testdataPredictionFMale = testdataPrediction.where(($"indexedLablel" === 1.0)).select("label", "features")

    val trainingdrop = testdataPredictionMale.unionAll(testdataPredictionFMale)
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLablel").fit(trainingdrop)
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").fit(trainingdrop)
    val splits2 = trainingdrop.randomSplit(Array(0.8, 0.2))
   // val sss = splits2(0).show(10)
    val Array(trainingData, testData) = trainingdrop.randomSplit(Array(0.8, 0.2))
    // Train a DecisionTree model.
    val dt = new DecisionTreeClassifier().setFeaturesCol("indexedFeatures").setLabelCol("indexedLablel").setMaxDepth(10)
    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt))
    val modeldrop = pipeline.fit(trainingdrop)
    // val dropdata=testdataPredictionMale.unionAll(testdataPredictionFMale)
    (trainingdrop, modeldrop)
  }

  def evaluateModel(model: Transformer,
                    data: DataFrame,
                    lableColName: String) = {
    //val testdataPrediction = pipelineModel.transform(test)
    val testdataPrediction = model.transform(data)
    val predictions = testdataPrediction.select("prediction").map(_.getDouble(0))
    val lables = testdataPrediction.select(lableColName).map(_.getDouble(0))
    val lablesAndpredictions = predictions.zip(lables)
    val metrics = new BinaryClassificationMetrics(lablesAndpredictions)
    val auc = metrics.areaUnderROC()
    val accs = lablesAndpredictions.map {
      line =>
        if (line._1 == line._2) 1 else 0
    }.sum
    val acc = accs / lablesAndpredictions.count()
    (auc, acc)
    //aucc: (Double, Double) = (0.5849461445602667,0.924936788874842)
    //  println(s"Model AUC: $auc")
  }

  def evaluateModelFM(model: Transformer,
                      data: DataFrame,
                      lableColName: String) = {
    val testdataPrediction = model.transform(data)
    val predictions = testdataPrediction.select("prediction").map(_.getDouble(0))
    val lables = testdataPrediction.select(lableColName).map(_.getDouble(0))

    val lablesAndpredictions = predictions.zip(lables)

    val lablesAndpredictionsFM = lablesAndpredictions.filter {
      line =>
        if (line._2 == 1.0) true else false
    }
    /* val accsFM = lablesAndpredictionsFM.map {
       line =>
         if (line._1 == line._2) 1 else 0
     }.sum
     val accFM = accsFM / lablesAndpredictionsFM.count()
     val lablesAndpredictionsM=lablesAndpredictions.filter{
       line=>
         if(line._2==1.0) true else false
     }
     val accsM = lablesAndpredictionsM.map {
       line =>
         if (line._1 == line._2) 1 else 0
     }.sum
     val accM = accsM / lablesAndpredictionsFM.count()
     (accFM,accM)*/
    //aucc: (Double, Double) = (0.5849461445602667,0.924936788874842)
    //  println(s"Model AUC: $auc")
    lablesAndpredictionsFM
  }

  def loadDatasets(sc: SparkContext, path: String) = {
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    val datas = Utils.loadSVMFile(sc, path, -1)
    // val splits: Array[RDD[LabeledPoint]] = datas.randomSplit(Array(0.8, 0.2))
    // val dataframes = splits.map(_.toDF())
    /*.map {
      line =>
        line.withColumn("lableString", line("label").cast(StringType))
    }*/
    val dataframes = datas.toDF()
    val numFeatures = dataframes.select("features").first().getAs[Vector](0).size
    println(s"特征数据量为 $numFeatures")
    datas
  }

  def getKnownSex(hiveContext: HiveContext, path: String) = {
    hiveContext.setConf("mapred.output.compress", "false")
    hiveContext.setConf("hive.exec.compress.output", "false")
    hiveContext.setConf("mapreduce.output.fileoutputformat.compress", "false")
    val knowImei = hiveContext.sql("select imei,sex from imei_sex_for_build_sexmodel")
    println("====")
    val unkownImei = hiveContext.sql("select  imei,value  from  adl_fdt_app_adv_model_install")
    //取得训练的数据
    val tainDataDF = knowImei.join(unkownImei, knowImei("imei") === unkownImei("imei"), "left").select("sex", "value")
    val rrr = tainDataDF.select()
    // val tainDataDF2 = tainDataDF.where((tainDataDF("feature") !== ""))
    val tainDataDF2 = tainDataDF.where((tainDataDF("value").isNotNull))
    tainDataDF2
  }
}
