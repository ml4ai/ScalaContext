package org.ml4ai.data

object Utils {
  def argMax(values:Map[Int, Double]):Int = {
    var bestK = Integer.MIN_VALUE
    var bestF1 = Double.MinValue
    values.map(x => {if (x._2 > bestF1) {bestK = x._1; bestF1 = x._2}})
    bestK
  }

  def f1(preds: Map[String, Int]): Double = {
    val p = precision(preds)
    val r = recall(preds)
    if (p + r == 0) 0.0
    else ((2 * (p * r))/(p + r))
  }

  def precision(preds: Map[String, Int]): Double = {
    if (preds("TP") + preds("FP") == 0) 0.0
    else (preds("TP") / (preds("TP") + preds("FP")))
  }

  def recall(preds: Map[String, Int]): Double = {
    if (preds("TP") + preds("FN") == 0) 0.0
    else (preds("TP")/(preds("TP") + preds("FN")))
  }

  def accuracy(preds:Map[String, Int]): Double = {
    if ((preds("TP") + preds("FP") + preds("FN") + preds("TN")) == 0) 0.0
    else (preds("TP") + preds("TN"))/(preds("TP") + preds("TN") + preds("FP") + preds("FN"))
  }

  private def predictCounts(yTrue: Array[Int], yPred: Array[Int]): Map[String, Int] = {
    val indexValuePair = yTrue.zipWithIndex
    var TP = 0; var FP = 0; var TN = 0; var FN = 0
    for((v,i) <- indexValuePair) {
      if (v == yPred(i) == 1) TP+=1
      if (v == 1 && v!=yPred(i)) FP +=1
      if (v == yPred(i) == 0) TN +=1
      if (v == 0 && v!=yPred(i)) FN +=1
    }
    Map(("TP" -> TP), ("FP" -> FP), ("TN" -> TN), ("FN" -> FN))
  }

  def deterministicSentDist(yTrue: Array[Int], sentDistValues: Array[Double], k:Int = 3):Map[String, Double] = {
    val tempo = sentDistValues.map(ele => {if (ele <= k) 1 else 0})
    val countMap = predictCounts(yTrue, tempo)
    val p =precision(countMap)
    val recall = recall(countMap)
    val f1 = f1(countMap)
    val ac = accuracy(countMap)
    Map(("precision_score" -> p), ("recall_score" -> r), ("f1_score" -> f1), ("accuracy_score" -> ac))
  }
}
