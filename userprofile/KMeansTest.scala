/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.examples.mllib

import org.apache.spark.{SparkConf, SparkContext}
// $example on$
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

object KMeansTest {

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("KMeansUserActive")
    val sc = new SparkContext(conf)
    val data=sc.textFile("/tmp/kmeans_data.txt")
    val parsesData=data.map{
      line=>
        org.apache.spark.mllib.linalg.Vectors.dense(line.split(" ").map(_.toDouble))
    }
   val numCluster=4
   val numIterations=10
   val clusterModule=KMeans.train(parsesData,numCluster,numIterations)
   val clusterCenter=clusterModule.clusterCenters.map(_.toArray.mkString(":")).foreach(println)

   val ecaluations=for(cluster<-Array(2,3,4,5,6)) yield {
     val clusterModule=KMeans.train(parsesData,cluster,numIterations)
     val WSSSE=clusterModule.computeCost(parsesData)
     (cluster,WSSSE)
   }
    ecaluations.sortBy(_._2).reverse.foreach(println)

  }
}
