package org.ml4ai.data.classifiers
import com.typesafe.config.ConfigFactory
import java.util.zip.GZIPInputStream

import org.clulab.learning.LinearSVMClassifier
import org.ml4ai.data.utils.{AggregatedRow, CodeUtils}


object SVMTrainSaveInstance extends App {
  //data preprocessing
  val config = ConfigFactory.load()
  val fileName = config.getString("svm.trainedModel")
  val SVMClassifier = new LinearSVMClassifier[Int, String](C = 0.001, eps = 0.001, bias = false)
  val svmInstance = new LinearSVMWrapper(SVMClassifier)
  val groupedFeatures = config.getString("features.groupedFeatures")
  val hardCodedFeaturePath = config.getString("features.hardCodedFeatures")
  val (allFeatures,rows) = CodeUtils.loadAggregatedRowsFromFile(groupedFeatures, hardCodedFeaturePath)
  val nonNumericFeatures = Seq("PMCID", "label", "EvtID", "CtxID", "")
  val numericFeatures = allFeatures.toSet -- nonNumericFeatures.toSet
  val featureDict = createFeaturesLists(numericFeatures.toSeq)
  val bestFeatureSet = featureDict("NonDep_Context")
  val trainingDataPrior = rows.filter(_.PMCID != "b'PMC4204162'")
  val trainingData = extractDataByRelevantFeatures(bestFeatureSet, trainingDataPrior)

  // training the machine learning model and writing it to file
  val trainingLabels = DummyClassifier.convertOptionalToBool(trainingData)
  val labelsToInt = DummyClassifier.convertBooleansToInt(trainingLabels)
  val tups = svmInstance.constructTupsForRVF(trainingData)
  val (trainDataSet, _) = svmInstance.mkRVFDataSet(labelsToInt,tups)
  svmInstance.fit(trainDataSet)
  svmInstance.saveModel(fileName)


  def createFeaturesLists(numericFeatures: Seq[String]):Map[String, Seq[String]] = {
    val contextDepFeatures = numericFeatures.filter(_.startsWith("ctxDepTail"))
    val eventDepFeatures = numericFeatures.filter(_.startsWith("evtDepTail"))
    val nonDepFeatures = numericFeatures.toSet -- (contextDepFeatures.toSet ++ eventDepFeatures.toSet)
    val map = collection.mutable.Map[String, Seq[String]]()
    map += ("All_features" -> numericFeatures)
    map += ("Non_Dependency_Features" -> nonDepFeatures.toSeq)
    map += ("NonDep_Context" -> (nonDepFeatures ++ contextDepFeatures.toSet).toSeq)
    map += ("NonDep_Event" -> (nonDepFeatures ++ eventDepFeatures.toSet).toSeq)
    map += ("Context_Event" -> (contextDepFeatures.toSet ++ eventDepFeatures.toSet).toSeq)
    map.toMap
  }

  def extractDataByRelevantFeatures(featureSet:Seq[String], data:Seq[AggregatedRow]):Seq[AggregatedRow] = {
    val result = data.map(d => {
      val currentSent = d.sentenceIndex
      val currentPMCID = d.PMCID
      val currentEvtId = d.EvtID
      val currentContextID = d.CtxID
      val currentLabel = d.label
      val currentFeatureName = d.featureGroupNames
      val currentFeatureValues = d.featureGroups
      val indexList = collection.mutable.ListBuffer[Int]()
      // we need to check if the feature is present in the current row. Only if it is present should we try to access its' value.
      // if not, i.e. if the feature is not present and we try to access it, then we get an ArrayIndexOutOfBound -1 error/
      featureSet.map(f => {
        if(currentFeatureName.contains(f)) {
          val tempIndex = currentFeatureName.indexOf(f)
          indexList += tempIndex
        }
      })
      val valueList = indexList.map(i => currentFeatureValues(i))
      AggregatedRow(currentSent, currentPMCID, currentEvtId, currentContextID, currentLabel, valueList.toArray, featureSet.toArray)
    })
    result
  }
}
