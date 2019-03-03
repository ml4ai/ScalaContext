package org.ml4ai.data.utils.correctDataPrep

import org.ml4ai.data.classifiers.{Baseline, DummyClassifier, GradTreeBoost, LinearSVMWrapper, SVM}

import scala.collection.mutable
import scala.io.BufferedSource
case class FoldMaker(groupedFeatures: Map[(String, String, String), AggregatedRowNew]) extends Iterable[(Array[Int], Array[Int], Array[Int])]{
  def toFolds:Iterable[(Array[Int], Array[Int], Array[Int])] = new mutable.HashSet[(Array[Int], Array[Int], Array[Int])]()
  override def iterator:Iterator[(Array[Int], Array[Int], Array[Int])] = this.toFolds.iterator
}

object FoldMaker {

  def extractData(rows: Seq[AggregatedRowNew], sentMinIndex: Int): Array[Array[Double]] = {
    val returnValue = new mutable.ListBuffer[Array[Double]]()
    rows.map(r => {
      val temp = r.featureGroups(sentMinIndex)
      val array = Array(temp)
    returnValue += array })
    returnValue.toArray
  }

  def getFoldsPerPaper(bufSource:BufferedSource):Array[(Array[Int], Array[Int], Array[Int])] = {
    val perPaperLines = bufSource.getLines()
    val toReturn = collection.mutable.ListBuffer[(Array[Int], Array[Int], Array[Int])]()
    perPaperLines.foreach(p => {
      val sets = p.split("\\]\",\"\\[")
      val cleanStrSets = sets.map(s => s.replace("\"[",""))
      val clean2 = cleanStrSets.map(c => c.replace("]\"",""))
      val trainIndex = splitAndExtract(clean2(0))
      val valIndex = splitAndExtract(clean2(1))
      val testIndex = splitAndExtract(clean2(2))
      val tup = (trainIndex, valIndex, testIndex)
      toReturn += tup
    })

    def splitAndExtract(arr: String):Array[Int] = {
      val strArr = arr.split(", ")
      val intVals = strArr.map(s => Integer.parseInt(s))
      intVals
    }
    toReturn.toArray
  }

  def baselineController(foldsFromCSV: Array[(Array[Int], Array[Int], Array[Int])], rows2: Seq[AggregatedRowNew]): (Array[Int], Array[Int]) = {
    val sentDistMinIndex = 18
    val giantTruthTestLabel = new mutable.ArrayBuffer[Int]()
    val giantPredTestLabel = new mutable.ArrayBuffer[Int]()
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
    (giantTruthTestLabel.toArray, giantPredTestLabel.toArray)
  }

  def svmControllerLinearSVM(svmInstance: LinearSVMWrapper, foldsFromCSV: Array[(Array[Int], Array[Int])], rows2: Seq[AggregatedRowNew]): (Array[Int], Array[Int]) = {
    val giantTruthTestLabel = new mutable.ArrayBuffer[Int]()
    val giantPredTestLabel = new mutable.ArrayBuffer[Int]()
    for((train,test) <- foldsFromCSV) {
      val trainingData = train.collect{case x:Int => rows2(x)}
      val balancedTrainingData = Balancer.balanceByPaperAgg(trainingData, 1)

      val trainingLabels = DummyClassifier.convertOptionalToBool(balancedTrainingData)
      val labelsToInt = DummyClassifier.convertBooleansToInt(trainingLabels)

      val tups = svmInstance.constructTupsForRVF(balancedTrainingData)
      val (trainDataSet, _) = svmInstance.mkRVFDataSet(labelsToInt,tups)
      svmInstance.fit(trainDataSet)



      val testingData = test.collect{case t: Int => rows2(t)}
      val testLabels = DummyClassifier.convertOptionalToBool(testingData)
      val testLabelsTruth = DummyClassifier.convertBooleansToInt(testLabels)
      giantTruthTestLabel ++= testLabelsTruth
      val tupsTruth = svmInstance.constructTupsForRVF(testingData)
      val (_, testDatumCollect) = svmInstance.mkRVFDataSet(testLabelsTruth, tupsTruth)
      val testLabelsPred = testDatumCollect.map(td => svmInstance.predict(td))
      giantPredTestLabel ++= testLabelsPred
    }

    (giantTruthTestLabel.toArray, giantPredTestLabel.toArray)

  }


  def gradBoostController(foldsFromCSV: Array[(Array[Int], Array[Int])], rows2: Seq[AggregatedRowNew]):(Array[Int], Array[Int]) = {
    val giantTruthTestLabel = new mutable.ArrayBuffer[Int]()
    val giantPredTestLabel = new mutable.ArrayBuffer[Int]()
    for((train,test) <- foldsFromCSV) {
      val trainingData = train.collect{case x:Int => rows2(x)}
      val balancedTrainingData = Balancer.balanceByPaperAgg(trainingData, 1)
      val (maxSize, dataFrame) = extractTrainingData(balancedTrainingData)

      val trainingLabels = DummyClassifier.convertOptionalToBool(balancedTrainingData)
      val labelsToInt = DummyClassifier.convertBooleansToInt(trainingLabels)


      val gradTreeB = new GradTreeBoost(dataFrame, labelsToInt)


      val testingData = test.collect{case t: Int => rows2(t)}
      val testLabels = DummyClassifier.convertOptionalToBool(testingData)
      val testLabelsTruth = DummyClassifier.convertBooleansToInt(testLabels)
      giantTruthTestLabel ++= testLabelsTruth
      val testDataFrame = extractTestingData(testingData, maxSize)
      val testLabelsPred = gradTreeB.predict(testDataFrame)
      giantPredTestLabel ++= testLabelsPred
    }

    (giantTruthTestLabel.toArray, giantPredTestLabel.toArray)
  }

  def extractTrainingData(dataSet:Seq[AggregatedRowNew]):(Int, Array[Array[Double]]) = {
    val featureValues = collection.mutable.ArrayBuffer[Array[Double]]()
    val toReturn = collection.mutable.ArrayBuffer[Array[Double]]()
    dataSet.map(featureValues+=_.featureGroups)
    val sizeList = featureValues.map(_.size)
    val maxSize = sizeList.reduceLeft(_ max _)
    featureValues.map(f => {
      val missing = maxSize - f.size
      if(missing == 0) toReturn += f
      else{
        val zeroArray = List.fill(missing)(0.0).toArray
        val tempo = Array(f ++ zeroArray)
        toReturn ++= tempo
      }
    })
    (maxSize, toReturn.toArray)
  }

  def extractTestingData(dataSet:Seq[AggregatedRowNew], missingArraySize:Int):Array[Array[Double]] = {
    val featureValues = collection.mutable.ArrayBuffer[Array[Double]]()
    val toReturn = collection.mutable.ArrayBuffer[Array[Double]]()
    dataSet.map(featureValues+=_.featureGroups)
    featureValues.map(f => {
      val miss = missingArraySize - f.size
      if(miss == 0) toReturn += f
      else{
        val zeroArray = List.fill(miss)(0.0).toArray
        val tempo = Array(f ++ zeroArray)
        toReturn ++= tempo
      }
    })
    toReturn.toArray
  }

  def gbmScoreMaker(name: String, truthTest:Array[Int], predTest:Array[Int]): Map[String,  (String, Double, Double, Double)] = {

    val countsTest = Utils.predictCounts(truthTest, predTest)
    val precTest = Utils.precision(countsTest)
    val recallTest = Utils.recall(countsTest)
    val f1Test = Utils.f1(countsTest)
    val testTup = ("test", precTest, recallTest, f1Test)
    val mapToReturn = Map(name -> testTup)
    mapToReturn
  }


}
