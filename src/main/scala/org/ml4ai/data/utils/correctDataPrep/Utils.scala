package org.ml4ai.data.utils.correctDataPrep
import java.io._

import com.typesafe.scalalogging.LazyLogging

import scala.io.Source
object Utils extends LazyLogging {
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
    //println((preds("TP")))
    //println((preds("TP") + preds("FP")))
    if(!(preds("TP") + preds("FP") == 0)) preds("TP").toDouble / (preds("TP") + preds("FP")).toDouble
    else 0.0
  }

  def recall(preds: Map[String, Int]): Double = {
    if (!(preds("TP") + preds("FN") == 0)) preds("TP").toDouble/(preds("TP") + preds("FN")).toDouble
    else 0.0
  }

  def accuracy(preds:Map[String, Int]): Double = {
    if ((preds("TP") + preds("FP") + preds("FN") + preds("TN")) == 0) 0.0
    else (preds("TP") + preds("TN"))/(preds("TP") + preds("TN") + preds("FP") + preds("FN"))
  }

  def predictCounts(yTrue: Array[Int], yPred: Array[Int]): Map[String, Int] = {
    val indexValuePair = yTrue zip yPred
    var TP = 0; var FP = 0; var TN = 0; var FN = 0
    for((gt,pr) <- indexValuePair) {
      if (gt == 1 && pr == 1) TP+=1
      if (gt == 1 && pr == 0) FN +=1
      if (gt == 0 && pr == 0) TN +=1
      if (gt == 0 && pr == 1) FP +=1
    }
    Map(("TP" -> TP), ("FP" -> FP), ("TN" -> TN), ("FN" -> FN))
  }

  def scoreMaker(name: String, truthTest:Array[Int], predTest:Array[Int]):Map[String, (String, Double, Double, Double)] = {
    val countsTest = Utils.predictCounts(truthTest, predTest)
    val precTest = Utils.precision(countsTest)
    val recallTest = Utils.recall(countsTest)
    val f1Test = Utils.f1(countsTest)
    val testTup = ("test", precTest, recallTest, f1Test)
    val mapToReturn = Map(name -> testTup)
    mapToReturn
  }

  def combineTrainVal(folds: Array[(Array[Int], Array[Int], Array[Int])]):Array[(Array[Int], Array[Int])] = {
    val trainValCombined = collection.mutable.ListBuffer[(Array[Int], Array[Int])]()
    for((train,validate,test) <- folds) {
      val trainVal = train ++ validate
      val toAdd = (trainVal, test)
      trainValCombined += toAdd
    }
    trainValCombined.toArray
  }

  def createFeatureDictionary(numericFeatures: Seq[String]):Map[String, Seq[String]] = {
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

  def writeAllFeaturesToFile(allFeatures:Seq[String], fileName:String="./src/main/resources/allFeaturesFile.txt"):Unit = {
    val printWriter = new PrintWriter(new File(fileName))
    allFeatures.map(a => if(allFeatures.indexOf(a)!= allFeatures.size-1) {
      printWriter.write(a+",")
    }
    else {
      printWriter.write(a)
    })
  }

  def createStats(nums: Iterable[Double]): (Double, Double, Double) = {
    val min = nums.min
    val max = nums.max
    val avg = nums.sum / nums.size
    (min, max, avg)
  }

  def extendFeatureName(f:String):(String, String, String) = {

      val feat_min = s"${f}_min"
      val feat_max = s"${f}_max"
      val feat_avg = s"${f}_avg"
      (feat_min, feat_max, feat_avg)

  }

  def featureConstructor(file:String):(Seq[String], Seq[String]) = {
    val allFeatures = collection.mutable.ListBuffer[String]()
    for(l <- Source.fromFile(file).getLines) {
      val contents = l.split(",")
      contents.map(allFeatures+=_)
    }
    (allFeatures, createBestFeatureSet(allFeatures))
  }

  def createBestFeatureSet(allFeatures:Seq[String]):Seq[String] = {
    val nonNumericFeatures = Seq("PMCID", "label", "EvtID", "CtxID", "")
    val numericFeatures = allFeatures.toSet -- nonNumericFeatures.toSet
    val featureDict = Utils.createFeatureDictionary(numericFeatures.toSeq)
    val bestFeatureSet = featureDict("NonDep_Context")
    bestFeatureSet
  }
  // for every new feature, we add to a map of (featureName, (_min, _max, _sum, size))
  def aggregateInputRowFeats(rows:Seq[String]):Map[String,(Double,Double, Double, Int)] = {
    val resultingMap = collection.mutable.Map[String,(Double,Double, Double, Int)]()
    for(r <- rows) {
      if(resultingMap.contains(r)) {
        val valueToBeAdded = 1.0
        val currentFeatDetails = resultingMap(r)
        val tupReplace = (Math.min(currentFeatDetails._1, valueToBeAdded),
                        Math.max(currentFeatDetails._2, valueToBeAdded),
                        currentFeatDetails._3 + valueToBeAdded,
                        currentFeatDetails._4+1)
        resultingMap(r) = tupReplace

      }
      else {
        val entry = (r -> (1.0,1.0,1.0,1))
        resultingMap += entry
      }
    }
    resultingMap.toMap
  }

  // in the given map, key is the feature name that we will extend to name_min, name_max, name_avg
  // value is (_min, _max, total, size) wherein we extract the following tup: (_min, _max, total/size)
  // important part is to store them in the same order, in harmony with AggregatedRowNew
  def finalFeatValuePairing(aggr: Map[String,(Double,Double, Double, Int)]): Seq[((String,String,String), (Double,Double,Double))] = {
    val finalPairings = collection.mutable.ListBuffer[((String,String,String), (Double,Double,Double))]()
    for((key,value)<- aggr){
      val extendedKey = extendFeatureName(key)
      val nameTup = (extendedKey._1, extendedKey._2, extendedKey._3)
      val valueTup = (value._1, value._2, (value._3/value._4))
      val currentTup = (nameTup, valueTup)
      finalPairings+= currentTup
    }

    finalPairings
  }

  def writeFrequenciesToFile(input: Seq[AggregatedRowNew], bestFeatureSet:Seq[String], filename:String):Unit = {
    logger.info("inside frequency count function")
    val mut = collection.mutable.HashMap[String,Int]()
    val printWriter = new PrintWriter(new File(filename))
    printWriter.write("Size of input: " + input.size + "\n")
    logger.info((input.size).toString)
    for(i <- input){
      logger.info("inside for loop for input")
      val currentFeatureSet = i.featureGroupNames
      //val currentFeatureValue = i.featureGroups
      //val currentIndex = currentFeatureSet.indexOf(i)
      for(c<-currentFeatureSet) {
        if(bestFeatureSet.contains(c)){
        //if(bestFeatureSet.contains(c) && currentFeatureValue(currentIndex)!=0.0) {
        if(mut.contains(c)){
          val freq = mut(c)+1
          mut += (c -> freq)
        }
        else mut+=(c->1)}}
      //}




    for((k,v)<- mut.toMap) {
      val string = k + " : " + v + "\n"
      logger.info("Checking string to write")
      logger.info(string)
      printWriter.write(string)
    }
  }}



}
