/**
 * Created by thinkpad on 2018/11/7.
 */

/**
 * @author xubo
 * sparkCSV learning
 * @time 20160419
 * reference https://github.com/databricks/spark-csv
 * blog http://blog.csdn.net/xubo245/article/details/51184946
  */
package com.apache.spark.sparkCSV.learning


import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.{SaveMode, SQLContext}

object readCsvBySparkSQLLoad {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("SparkLearning:SparkCSV").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext._
    val df = sqlContext.load("com.databricks.spark.csv", Map("path" -> "file:///home/yarn/sparkdatas/imei_sex_for_build_sexmodel.csv", "header" -> "true"))
    // df.select("year", "model").save("file/data/sparkCSV/output/newcars.csv", "com.databricks.spark.csv")
   // sql("create table adl_fdt_app_adv_model_install (imei String, value String,stat_date String)")
    df.show
    //df.write.mode(SaveMode.Append).insertInto("default.adl_fdt_app_adv_model_install")
    df.write.saveAsTable("imei_sex_for_build_sexmodel")
    sc.stop
  }
}