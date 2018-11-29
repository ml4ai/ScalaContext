package org.ml4ai

import java.util.zip._

import data.Balancer
import data.AggregatedRowNew
import data.FoldMaker
import data.DummyClassifier
import scala.collection.mutable
import data.Utils
import data.Baseline
import scala.io.Source
object Main extends App {
  val (allFeatures,rows) = AggregatedRowNew.fromStream(new GZIPInputStream(getClass.getResourceAsStream("/grouped_features.csv.gz")))
  val rows2 = rows.filter(_.PMCID != "b'PMC4204162'")
  val bufferedFoldIndices = Source.fromFile("./src/main/resources/cv_folds_val_4.csv")
  // in the specific set of features, sentenceDist_min is always at index 18 for all thr aggregated rows
  val sentDistMinIndex = 18
  val foldsFromCSV = FoldMaker.getFoldsPerPaper(bufferedFoldIndices)
  val giantTruthTestLabel = new mutable.ArrayBuffer[Int]()
  val giantPredTestLabel = new mutable.ArrayBuffer[Int]()
  val giantTruthValLabel = new mutable.ArrayBuffer[Int]()
  val giantPredValLabel = new mutable.ArrayBuffer[Int]()
  for((train,_,test) <- foldsFromCSV) {
      val trainingData = train.collect{case x:Int => rows2(x)}
      val balancedTrainingData = Balancer.balanceByPaperAgg(trainingData, 1)

      val sentDistTrainFrame = FoldMaker.extractData(balancedTrainingData, sentDistMinIndex)
      val trainingLabels = DummyClassifier.convertOptionalToBool(balancedTrainingData)
      val labelsToInt = DummyClassifier.convertBooleansToInt(trainingLabels)
      val kToF1Map = new mutable.HashMap[Int, Double]
      for(k_val <- 0 until 51) {
        val trainInstance = new Baseline(k_val)
        val pred = trainInstance.predict(sentDistTrainFrame)
        val counts = Utils.predictCounts(labelsToInt, pred)
        val f1score = Utils.f1(counts)
        kToF1Map += (k_val -> f1score)
      }

      val bestK = Utils.argMax(kToF1Map.toMap)
      val testingData = test.collect{case x:Int => rows2(x)}
      val testSentFrame = FoldMaker.extractData(testingData, sentDistMinIndex)
      val currentTruthTest = DummyClassifier.convertOptionalToBool(testingData)
      val currentTruthTestInt = DummyClassifier.convertBooleansToInt(currentTruthTest)
      giantTruthTestLabel ++= currentTruthTestInt
      val testInstance = new Baseline(bestK)
      val predTestLabel = testInstance.predict(testSentFrame)
      giantPredTestLabel ++= predTestLabel

  }

  // converts boolean mappings to 1's and 0's in test set and then calculates metrics

  val testCounts = Utils.predictCounts(giantTruthTestLabel.toArray, giantPredTestLabel.toArray)
  val precisionTest = Utils.precision(testCounts)
  val recallTest = Utils.recall(testCounts)
  val f1Test = Utils.f1(testCounts)


  var scoreDictionary = collection.mutable.Map[String, ((String, Double, Double, Double), (String, Double, Double, Double))]()
  val baselineResults = (("validation", 0.0,0.0,0.0), ("test", precisionTest, recallTest, f1Test))
  scoreDictionary += ("baseline" -> baselineResults)
  println(scoreDictionary.values)

}
