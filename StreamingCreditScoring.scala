package creditscoring



import org.apache.spark.SparkConf
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.flume.FlumeUtils
import org.apache.spark.streaming.{Time, StreamingContext, Milliseconds}
import org.apache.spark.util.IntParam

/**
 * Created by thinkpad on 2017/12/5.
 */
class StreamingCreditScoring {

}
case class Record(word: String)
object StreamingCreditScoring{
  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println(
        "Usage: FlumeEventCount <host> <port>")
      System.exit(1)
    }
  //  StreamingCreditScoring.setStreamingLogLevels()
   // val Array(host, IntParam(port)) = args
    val batchInterval = Milliseconds(2000)

    // Create the context and set the batch size
    val sparkConf = new SparkConf().setAppName("FlumeEventCount")
    val ssc = new StreamingContext(sparkConf, batchInterval)

    // Create a flume stream
    val stream = FlumeUtils.createStream(ssc, "127.0.0.1", 8088, StorageLevel.MEMORY_ONLY_SER_2)

    val streamEvent=stream.map(e=>new String(e.event.getBody.array()))
    val creditRDD= streamEvent.flatMap(_.split(" "))
    import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}

    val creditModel = CrossValidatorModel.load("/home/spark/pipilinemodel/")
    stream.count().map(cnt => "Received " + cnt + " flume events." ).print()
    creditRDD.foreachRDD{
      (rdds:RDD[String],time:Time)=>
     //   import org.apache.spark.sql.SparkSession
        val spark  = SparkSessionSingleton.getInstance(rdds.sparkContext.getConf)

        val requestArray=rdds.map(r=>r.asInstanceOf[String]).collect()
        if(requestArray.size>0){
         val  requestRDD=  spark.sparkContext.parallelize(requestArray)
          import spark.implicits._
          // Convert RDD[String] to RDD[case class] to DataFrame
          val  credutDataFrame = rdds.map(w => Record(w)).toDF()
          val  trainData=creditModel.transform(credutDataFrame)
       //   credutDataFrame.createOrReplaceTempView("words")
        }

    }

    ssc.start()
    ssc.awaitTermination()
  }
  object SparkSessionSingleton {

    @transient  private var instance: SparkSession = _

    def getInstance(sparkConf: SparkConf): SparkSession = {
      if (instance == null) {
        instance = SparkSession
          .builder
          .config(sparkConf)
          .getOrCreate()
      }
      instance
    }
  }
}
