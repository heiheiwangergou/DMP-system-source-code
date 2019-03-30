package sexmodel

import java.net.URI
import java.text.SimpleDateFormat
import java.util.Calendar

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer

/**
 * 对性别进行预测
 * Created by zhouxiaoke on 2015/8/5.
 */
object AppUseActionBased {

  case class Item(itemCode: String, itemName: String, itemColIndex: Long)
  case class Imei_feature(imei: String, features: String)
  case class Imei_sex(imei: String, sex: String)

  def main(args: Array[String]): Unit = {
    val sparkConf: SparkConf = new SparkConf()
    System.setProperty("user.name", "algo")
    System.setProperty("HADOOP_USER_NAME", "algo")
    sparkConf.setAppName("ZXK_ALGO_sex_mode_appUse_actionBased")
    val sc: SparkContext = new SparkContext(sparkConf)
    sc.hadoopConfiguration.set("mapred.output.compress", "false")
    val calendar: Calendar = Calendar.getInstance
    calendar.add(Calendar.DATE, -1)
    val yestoday_Date: String = new SimpleDateFormat("yyyyMMdd").format(calendar.getTime)

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    val splitChar: String = "\u0001"
    val model_save_dir: String = "hdfs://hd-nn-1.meizu.gz:9000/user/mzsip/zxk/algo/sex_model/models/app_use/" + yestoday_Date + "/"
    val hiveContext: HiveContext = new HiveContext(sc)
    hiveContext.setConf("mapred.output.compress", "false")
    hiveContext.setConf("hive.exec.compress.output", "false")
    hiveContext.setConf("mapreduce.output.fileoutputformat.compress", "false")

    //原始维表的表名称
    val source_dim_table_name: String = "app_center.adl_sdt_adv_dim_app_boot"
    //原始的用户行为表的名称
    val source_feature_table_name: String = "app_center.adl_fdt_app_adv_model_boot"
    //年龄已知的数据
    val sex_know_table_name: String = "algo.imei_sex_for_build_sexmodel"
    //筛选出安装量最大的topK个item
    val topK: Int = 1000
    //item的code前缀
    val code_prefix: String = "12"
    //将要创建的新的维表名称
    val create_dim_table_name: String = "algo.sex_model_app_use_items_on_" + topK.toString + "dims"
    //将要创建的新的用户行为表名称
    val create_feature_table_name: String = "algo.sex_model_imei_app_use_features_on_" + topK.toString + "dims"
    val create_predict_table_name: String = "algo.sex_model_appUse_ActionBased_prediction_on_" + topK.toString + "dims"

    //(itemCode, (itemName, installNum))
    val source_topK_rdd: RDD[(String, (String, Long))] = get_topK_rdd(hiveContext, topK, code_prefix, source_dim_table_name, splitChar, yestoday_Date)
    //(itemCode, (itemName, value))
    val source_feature_rdd: RDD[(String, (String, Double))] = get_features_rdd(hiveContext, source_feature_table_name, splitChar, yestoday_Date)

    println(" \n\n**************  count of source_dim_rdd: " + source_topK_rdd.count() + "  *********************")
    println(" **************  count of source_feature_rdd: " + source_feature_rdd.count() + "  *********************\n\n")

    val use_topK_rdd: RDD[(String, (String, Long))] = get_use_item_rdd(hiveContext, create_dim_table_name, source_feature_rdd, source_topK_rdd, yestoday_Date)
    val imei_feature_rdd: RDD[(String, String)] = get_user_feature_data(hiveContext, create_feature_table_name, source_feature_rdd, use_topK_rdd, yestoday_Date)

    val dim_num: Int = use_topK_rdd.count().toInt

    val train_pre: (RDD[(String, LabeledPoint)], RDD[(String, Vector)]) = getTrainSet_predictSet(hiveContext, sex_know_table_name, imei_feature_rdd, dim_num, splitChar)
    val threshold_model: (Double, LogisticRegressionModel) = buildModel(hiveContext.sparkContext, train_pre._1, model_save_dir)

    val pre: RDD[(String, String)] = predict(threshold_model, train_pre)

    import hiveContext.implicits._
    val pre_df: DataFrame = pre.map(v => Imei_sex(v._1, v._2)).toDF

    pre_df.registerTempTable("prediction")
    hiveContext.sql(
      "create table if not exists " +
        create_predict_table_name +
        " (imei string, sex string) partitioned by (stat_date string) stored as textfile")

    hiveContext.sql(
      "insert overwrite table " +
        create_predict_table_name +
        " partition(stat_date = " +
        yestoday_Date +
        " ) select * from prediction")

  }

  def predict(threshold_model: (Double, LogisticRegressionModel), train_pre: (RDD[(String, LabeledPoint)], RDD[(String, Vector)])): RDD[(String, String)] = {

    val prediction: RDD[(String, String)] = train_pre._2.map(v => {
      if (threshold_model._2.predict(v._2) > threshold_model._1)
        (v._1, "male")
      else
        (v._1, "female")
    })
    val pre: RDD[(String, String)] = train_pre._1.map(v => (v._1, if (v._2.label == 1d) "male" else "female")).union(prediction)
    pre
  }

  /**
   * 将用户的的行为数据整理成两部分数据，一部分是性别标签已知的样本，一部分是性别标签位置的样本
   * @param hiveContext                用于读取hive库中的表
   * @param know_sex_table_name        性别标签已知的imei，格式是: (imei, sex)  sex=1表示男   sex=0表示女
   * @param imei_feature_rdd           用户的行为数据，格式是：（imei, feature)
   * @param dim_num                    用户行为对应的维度
   * @param splitChar                  字段临时分隔符
   * @return                           返回两部分数据，一个是标签位置的样本，一个是标签已知的样本，
   *                                   用标签已知的样本建立模型，然后预测标签位置的样本
   */
  def getTrainSet_predictSet(
                              hiveContext: HiveContext,
                              know_sex_table_name: String,
                              imei_feature_rdd: RDD[(String, String)],
                              dim_num: Int,
                              splitChar: String): (RDD[(String, LabeledPoint)], RDD[(String, Vector)]) = {
    val select_know_sex_sql: String = "select * from " + know_sex_table_name
    val imei_sexLabel: RDD[(String, Double)] =
      hiveContext.sql(select_know_sex_sql)
        .map(_.mkString(splitChar))
        .filter(_.split(splitChar).length == 2)
        .map(v => {
        val array: Array[String] = v.trim.split(splitChar)
        (array(0).trim, array(1).trim.toDouble)
      })
    //(imei, (sexLabel, nullFeature, sexKnowTag))
    val imei_sexLabel_tail: RDD[(String, (Double, String, Int))] = imei_sexLabel.map(v => (v._1, (v._2, "", 1)))
    //(imei, (nuxLabel, feature, sexUnKnowTag))
    val imei_features_tail: RDD[(String, (Double, String, Int))] = imei_feature_rdd.map(v => (v._1, (0d, v._2, 2)))

    val union_tail: RDD[(String, (Double, String, Int))] = imei_sexLabel_tail
      .union(imei_features_tail)
      .reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3))

    //(imei, (sexLabel, feature, sexKnowTag))
    val imei_hasfeature_rdd: RDD[(String, (Double, String, Int))] = union_tail.filter(_._2._3 != 1)
    val imei_hasfeature_rdd1: RDD[(String, (Double, Array[Int], Array[Double], Int))] = imei_hasfeature_rdd.map(v => {
      val features: Array[String] = v._2._2.trim.split(" ")
      val index_array: ArrayBuffer[Int] = new ArrayBuffer[Int]()
      val value_array: ArrayBuffer[Double] = new ArrayBuffer[Double]()
      for (feature <- features) {
        val columnIndex_value: Array[String] = feature.trim.split(":")
        if (columnIndex_value.length == 2) {
          index_array += columnIndex_value(0).trim.toInt
          value_array += columnIndex_value(1).trim.toDouble
        }
      }
      (v._1, (v._2._1, index_array.toArray, value_array.toArray, v._2._3))
    }).filter(_._2._2.length > 0)
    val predictSet: RDD[(String, Vector)] = imei_hasfeature_rdd1
      .filter(_._2._4 == 2)
      .map(v => (v._1, Vectors.sparse(dim_num, v._2._2, v._2._3)))
    println("\n\n********************  count of predictSet: " + predictSet.count + "  *********************")
    val trainSet: RDD[(String, LabeledPoint)] = imei_hasfeature_rdd1
      .filter(_._2._4 == 3)
      .map(v => (v._1, new LabeledPoint(v._2._1, Vectors.sparse(dim_num, v._2._2, v._2._3))))
    println("********************  count of trainSet: " + trainSet.count + "  *********************\n\n")
    (trainSet, predictSet)
  }

  def deleteIfexists(model_save_dir: String): Unit = {
    val conf: Configuration = new Configuration()
    val fs: FileSystem = FileSystem.get(URI.create(model_save_dir), conf, "mzsip")
    val path: Path = new Path(model_save_dir)
    if (fs.exists(path))
      fs.delete(path, true)
  }

  /**
   * 将标签已知的数据分为训练集和测试集，训练集用于建立模型，测试集用于检验模型效果
   * @param sc                       当前的spark运行上下文
   * @param trainSet                 标签已知的样本
   * @param model_save_dir           模型的保存路径
   * @return                         返回一个logistic二分类模型和这个模型的对应的分类阈值
   */
  def buildModel(sc: SparkContext, trainSet: RDD[(String, LabeledPoint)], model_save_dir: String): (Double, LogisticRegressionModel) = {
    deleteIfexists(model_save_dir)
    var ret_model: LogisticRegressionModel = null
    val rdds: Array[RDD[(String, LabeledPoint)]] = trainSet.randomSplit(Array(0.8, 0.2))
    var trainRDD: RDD[(String, LabeledPoint)] = rdds(0).cache()
    val predictRDD: RDD[(String, LabeledPoint)] = rdds(1).cache()
    val num_1: Long = predictRDD.filter(_._2.label == 1d).count()
    val num_0: Long = predictRDD.filter(_._2.label == 0d).count()

    val thresholds: Array[Double] = (for (i <- 1 until 100) yield i.toDouble / 100).toArray
    val split: String = "\t"
    var model: LogisticRegressionModel = null
    var prediction: RDD[(String, (Double, Double))] = sc.emptyRDD[(String, (Double, Double))]
    var maxVar: Double = 0
    var goodthreshold: Double = 0
    var threshold: Double = 0
    val iterationNum: Int = 10
    var k: Int = 0
    while (k < iterationNum) {
      var predictInfo: ArrayBuffer[String] = new ArrayBuffer[String]()
      val train1: Long = trainRDD.filter(_._2.label == 1d).count()
      val train0: Long = trainRDD.filter(_._2.label == 0d).count()
      val train_info: String = "train1: " + train1 + split + "train0: " + train0
      predictInfo += train_info
      var tmpVariance: Double = 0
      model = new LogisticRegressionWithLBFGS().run(trainRDD.map(_._2))
      model.clearThreshold()
      model.save(sc, model_save_dir + "model_" + k.toString + "/model")
      prediction = predictRDD.map(v => (v._1, (v._2.label, model.predict(v._2.features)))).cache()
      println(s"\n\n&&&&&&&&&&&&&&&&&  第 $k 次模型效果！！ &&&&&&&&&&&&&&&&&")
      println(train_info)
      for (i <- 0 until thresholds.length) {
        val num_11: Long = prediction.filter(v => v._2._1 == 1d && v._2._2 > thresholds(i)).count()
        val num_10: Long = prediction.filter(v => v._2._1 == 1d && v._2._2 <= thresholds(i)).count()
        val num_00: Long = prediction.filter(v => v._2._1 == 0d && v._2._2 <= thresholds(i)).count()
        val num_01: Long = prediction.filter(v => v._2._1 == 0d && v._2._2 > thresholds(i)).count()
        val pre1: Double = if (num_11 + num_01 == 0d) 0d else num_11.toDouble / (num_11 + num_01)
        val pre0: Double = if (num_00 + num_10 == 0d) 0d else num_00.toDouble / (num_00 + num_10)
        val cov1: Double = if (num_1 == 0) 0d else num_11.toDouble / num_1
        val cov0: Double = if (num_0 == 0) 0d else num_00.toDouble / num_0
        val variance: Double = math.sqrt(cov1 * cov0)
        if (variance > tmpVariance) {
          tmpVariance = variance
          goodthreshold = thresholds(i)
        }
        val info: String =
          "threshold: " + thresholds(i) + split +
            "num_11:" + num_11 + split +
            "num_10: " + num_10 + split +
            "num_00: " + num_00 + split +
            "num_01: " + num_01 + split +
            "pre1: " + pre1 + split +
            "pre_0: " + pre0 + split +
            "cov1: " + cov1 + split +
            "cov0: " + cov0
        predictInfo += info
        println(info)
      }
      val pre_info: String =
        "num_1: " + num_1 + split +
          "num_0: " + num_0 + split +
          "tmpVariance: " + tmpVariance + split +
          "goodthreshold: " + goodthreshold
      predictInfo += pre_info
      println(pre_info)

      sc.parallelize(predictInfo.toSeq)
        .repartition(1)
        .saveAsTextFile(model_save_dir + "model_" + k.toString + "/model_analysis")

      if (tmpVariance > maxVar) {
        maxVar = tmpVariance
        threshold = goodthreshold
        ret_model = model
      }

      val trainPrediction: RDD[(String, (Double, Double))] = trainRDD.map(v => {
        (v._1, (v._2.label, model.predict(v._2.features)))
      })
      val correctUid: RDD[(String, Double)] = trainPrediction.filter(v => {
        (v._2._1 == 1.0 && v._2._2 >= goodthreshold) || v._2._1 == 0.0
      }).map(v => (v._1, v._2._1))
      val newTrainRDD: RDD[(String, LabeledPoint)] = correctUid.join(trainRDD).map(v => (v._1, v._2._2)).cache()
      trainRDD.unpersist()
      trainRDD.count()
      trainRDD = newTrainRDD
      prediction.unpersist()
      prediction.count()
      k += 1
    }
    (threshold, ret_model)
  }

  /**
   * 根据真正使用的app和出现在行为表中的app，关联得到用户的有效行为数据，并写入hive表中
   * @param hiveContext                    用于从hive中读取和写入数据
   * @param create_feature_table_name      将要创建的新的用户行为表名称
   * @param source_feature_rdd             用户的行为数据，格式为：RDD[(itemCode, (itemName, value))]
   * @param use_item_rdd                   真正使用到的app，格式为：RDD[(itemCode, (itemName, columnIndex))]
   * @param yestoday_Date                  数据日期
   * 写入hive表中的用户新的行为数据格式为：imei columnIndex1:value1 columnIndex2:value2 ...
   */
  def get_user_feature_data(
                             hiveContext: HiveContext,
                             create_feature_table_name: String,
                             source_feature_rdd: RDD[(String, (String, Double))],
                             use_item_rdd: RDD[(String, (String, Long))],
                             yestoday_Date: String): RDD[(String, String)] = {
    val create_feature_table_sql: String =
      "create table if not exists " +
        create_feature_table_name +
        " (imei string, feature string) partitioned by (stat_date bigint) stored as textfile"
    val insertInto_feature_table_sql: String = "insert overwrite table " +
      create_feature_table_name +
      " partition(stat_date = " + yestoday_Date  + ") select * from "

    //(imei, columnIndex:value)             //(imei, columnIndex:value)            (code, (imei, value))  join  (code, (itemName, columnIndex))
    val join_rdd: RDD[(String, Array[(Long, String)])] = source_feature_rdd.join(use_item_rdd).map(v => (v._2._1._1, Array((v._2._2._2, "1"))))

    val imei_feature: RDD[(String, String)] = join_rdd.reduceByKey(_ ++ _).map(v => {
      val array: Array[(Long, String)] = v._2.sortWith((a, b) => a._1 < b._1)
      val feature_str: String = array.map(v => v._1.toString + ":" + v._2).mkString(" ")
      (v._1, feature_str)
    })

    import hiveContext.implicits._
    val imei_feature_df: DataFrame = imei_feature.map(v => Imei_feature(v._1, v._2)).toDF()
    val tmp_table_name: String = "imei_feature_table"
    imei_feature_df.registerTempTable(tmp_table_name)
    hiveContext.sql(create_feature_table_sql)
    hiveContext.sql(insertInto_feature_table_sql + tmp_table_name)

    imei_feature
  }

  /**
   * 根据筛选出来的app和出现在行为表里面的app的code进行关联，得到将要使用的app，并将得到的结果写入维度表中
   * @param hiveContext                    用于从hive读取数据和写入数据
   * @param create_dim_table_name          将要创建的新的维度表名称
   * @param source_feature_rdd             用户的行为数据，格式为：RDD[(itemCode, (itemName, value))]
   * @param source_dim_rdd                 筛选出来的app，格式为：RDD[(itemCode, itemName))
   * @param yestoday_Date                  数据日期
   * @return
   */
  def get_use_item_rdd(
                        hiveContext: HiveContext,
                        create_dim_table_name: String,
                        source_feature_rdd: RDD[(String, (String, Double))],
                        source_dim_rdd: RDD[(String, (String, Long))],
                        yestoday_Date: String): RDD[(String, (String, Long))] = {
    val create_dim_table_sql: String =
      "create table if not exists " +
        create_dim_table_name +
        " (code string, name string, column_index bigint) partitioned by (stat_date bigint) stored as textfile"
    val insertInto_dim_table_sql: String = "insert overwrite table " +
      create_dim_table_name +
      " partition(stat_date = " + yestoday_Date  + ") select * from "

    val code_rdd: RDD[(String, Int)] = source_feature_rdd
      .map(_._1)
      .distinct()
      .map(v => (v.trim, 1))

    println(" \n\n***************  count of code_rdd: " + code_rdd.count() + "  *****************")

    val join_rdd: RDD[(String, (String, Long))] = source_dim_rdd
      .join(code_rdd)
      .map(v => (v._1, v._2._1))

    //(code, (itemName, columnIndex))
    val use_items: RDD[(String, (String, Long))] = join_rdd.zipWithIndex().map(v => (v._1._1, (v._1._2._1, v._2)))
    println("  *******************  count of use_items: " + use_items.count() + "*******************\n\n")
    import hiveContext.implicits._
    val use_items_df: DataFrame = use_items.map(v => Item(if (v._1.length > 2) v._1.substring(2, v._1.length) else v._1, v._2._1, v._2._2)).toDF()

    val tmp_table_name: String = "item_table"
    use_items_df.registerTempTable(tmp_table_name)
    hiveContext.sql(create_dim_table_sql)
    hiveContext.sql(insertInto_dim_table_sql + tmp_table_name)
    use_items
  }

  /**
   * 将用户的行为记录转换成：（code, (imei, value))的格式
   * @param hiveContext                    用于从hive中读取item的维度表
   * @param source_feature_table_name      原始用户行为表
   * @param splitChar                      用于分隔字段的临时分隔符
   * @param yestoday_Date                  数据日期
   * @return
   */
  def get_features_rdd(
                        hiveContext: HiveContext,
                        source_feature_table_name: String,
                        splitChar: String,
                        yestoday_Date: String): RDD[(String, (String, Double))] = {
    val select_source_feature_table_sql: String = "select * from " +
      source_feature_table_name +
      " where stat_date = " +
      yestoday_Date
    val imei_features_df: DataFrame = hiveContext.sql(select_source_feature_table_sql)
    //println("count of imei_features_df for " + sqls_dataType(i)._2 + ": " + imei_features_df.count())
    val imei_features_rdd1 = imei_features_df
      .map(v => v.mkString(splitChar))
      .map(v => {
      val array: Array[String] = v.trim.split(splitChar)
      (array(0).trim, array(1).trim)
    }).map(v => (v._1, v._2.trim.split(" ")))
    val imei_features_rdd2 = imei_features_rdd1.filter(_._2.length > 0)
    //println("count of imei_features_rdd2 for " + sqls_dataType(i)._2 + ": " + imei_features_rdd2.count)
    val imei_features_rdd3 = imei_features_rdd2.mapPartitions(iter => {
      new Iterator[(String, String)]() {
        var count: Int = 0
        var value: (String, Array[String]) = iter.next()
        override def hasNext: Boolean = {
          if (count < value._2.length)
            true
          else {
            if (iter.hasNext) {
              count = 0
              value = iter.next()
              true
            }
            else
              false
          }
        }

        override def next(): (String, String) = {
          count += 1
          (value._2(count - 1), value._1)
        }
      }
    })
    //println("count of imei_features_rdd3 for " + sqls_dataType(i)._2 + ": " +  + imei_features_rdd3.count())
    val imei_features_rdd4 = imei_features_rdd3.filter(_._1.trim.split(":").length == 2)
      .map(v => {
      val array: Array[String] = v._1.trim.split(":")
      (array(0).trim, (v._2, array(1).trim.toDouble))
    })
    //println("count of imei_features_rdd4 for" + sqls_dataType(i)._2 + ": " + + imei_features_rdd4.count())
    imei_features_rdd4
  }

  /**
   * 读取各个类型数据的维度表，将itemID转换成code,后面会用code作为key进行关联
   * @param hiveContext                  用于从hive中读取item的维度表
   * @param topK                         筛选出安装量最大的topK个app
   * @param code_prefix                  添加的code前缀
   * @param source_dim_table_name        原始维表
   * @param splitChar                    用于分隔字段的临时分隔符
   * @param yestoday_Date                数据日期
   * @return                             RDD[(itemCode, (itemName, installNum))]
   */
  def get_topK_rdd(
                    hiveContext: HiveContext,
                    topK: Int,
                    code_prefix: String,
                    source_dim_table_name: String,
                    splitChar: String,
                    yestoday_Date: String): RDD[(String, (String, Long))] = {
    val select_source_dim_table_sql: String = "select * from " +
      source_dim_table_name +
      " where stat_date = " +
      yestoday_Date
    val item_df: DataFrame = hiveContext.sql(select_source_dim_table_sql)
    //(itemCode, (itemName, installNum))
    val rdd: RDD[(String, (String, Long))] = item_df
      .map(v => v.mkString(splitChar).replace("\n", ""))
      .map(v => {
      val array: Array[String] = v.trim.split(splitChar)
      (code_prefix.trim + array(0).trim, (array(1).trim, array(2).trim.toLong))
    })
    //(installNum, (itemCode, itemName))
    val topK_array: Array[(Long, (String, String))] = rdd.map(v => (v._2._2, (v._1, v._2._1))).top(topK)
    //(itemCode, (itemName, installNum))
    hiveContext.sparkContext.parallelize(topK_array).map(v => (v._2._1, (v._2._2, v._1)))
  }

}
