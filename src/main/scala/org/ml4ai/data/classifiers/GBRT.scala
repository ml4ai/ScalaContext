package org.ml4ai.data.classifiers

import smile.classification.{GradientTreeBoost, gbm}

object GBRT {
  private final val NTREES:Int = 300
  def classifierInstance(x: Array[Array[Double]], y: Array[Int], ntrees:Int = NTREES): GradientTreeBoost = {
    gbm(x,y, null, ntrees = ntrees)
  }
}
