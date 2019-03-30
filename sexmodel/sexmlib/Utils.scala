package sexmodel.sexmlib

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.storage.StorageLevel

/**
 * Created by pc on 2015/9/12.
 */
class Utils extends  Serializable {

}

object Utils  extends  Serializable{
  def loadSVMFile(sc: SparkContext, path: String, numFeatures: Int) = {
    val parsed = sc.textFile(path)
      .map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#")))
      .map { line =>
      val items = line.split(' ')
      val label = items.head.toDouble
      val (indices, values) = items.tail.filter(_.nonEmpty).map { item =>
        val indexAndValue = item.split(':')
        val index = indexAndValue(0).toInt
        val value = indexAndValue(1).toDouble
        (index, value)
      }.unzip

      // 检查数据的顺序
      var previous = -1
      var i = 0
      val indicesLength = indices.length
      while (i < indicesLength) {
        val current = indices(i)
        require(current > previous, "数字得排序")
        previous = current
        i += 1
      }
      (label, indices.toArray, values.toArray)
    }
    // 定义数据feature的数量
    val d = if (numFeatures > 0) {
      numFeatures
    } else {
      parsed.persist(StorageLevel.MEMORY_ONLY)
      parsed.map { case (label, indices, values) =>
        indices.lastOption.getOrElse(0)
      }.reduce(math.max) + 1
    }
    parsed.map { case (label, indices, values) =>
      LabeledPoint(label, Vectors.sparse(d, indices, values))
    }
  }
}

