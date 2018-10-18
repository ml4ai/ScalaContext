package org.ml4ai.data
import scala.collection.mutable


case class FoldMaker(groupedFeatures: Map[(String, String, String), AggregatedRow]) extends Iterable[(Array[Int], Array[Int], Array[Int])]{
  def toFolds:Iterable[(Array[Int], Array[Int], Array[Int])] = new mutable.HashSet[(Array[Int], Array[Int], Array[Int])]()
  override def iterator:Iterator[(Array[Int], Array[Int], Array[Int])] = this.toFolds.iterator
}

object FoldMaker {
  def getFolds(groupedFeatures: Map[(String, String, String), AggregatedRow], valSize:Int = 4): mutable.HashSet[(Array[Int], Array[Int], Array[Int])] = {

    val degenRemoved = groupedFeatures.filter(_._1._1 != "b'PMC4204162'")
    val rowLabels = degenRemoved.map(x => x._1._1)
    val paperLabels = rowLabels.toSet
    val toReturn = paperFoldLists(paperLabels, rowLabels, valSize)
    toReturn

  }

  private def paperFoldLists(rowPaperLabels:Iterable[String], rows:Iterable[String], valSize: Int): mutable.HashSet[(Array[Int], Array[Int], Array[Int])] = {
    val toReturn = new mutable.HashSet[(Array[Int], Array[Int], Array[Int])]
    val testingSets = collection.mutable.Map[String, collection.mutable.HashSet[Int]]()
    rowPaperLabels.foreach({x => testingSets += (x -> new collection.mutable.HashSet[Int])})
    rows.zipWithIndex.foreach {
      case(ele, index) => testingSets(ele) += index
    }

    rowPaperLabels.foreach {
      case currentId => {
        val testingSet = testingSets(currentId)
        val otherIDs = rowPaperLabels.toSet -- Seq(currentId)
        val shuffled = scala.util.Random.shuffle(otherIDs)
        val validationIds = shuffled.take(valSize)
        val trainingIds = shuffled.drop(valSize)
        val validationSet = new mutable.HashSet[Int]()
        val trainingSet = new mutable.HashSet[Int]()
        rows.zipWithIndex.foreach {
          case (paperID, subIndex) => {
            if (validationIds.contains(paperID))
              validationSet += subIndex
            else if (trainingIds.contains(paperID))
              trainingSet += subIndex
          }
        }
        val tup = (trainingSet.toArray, validationSet.toArray, testingSet.toArray)
        toReturn += tup

      }
    }
    toReturn
  }

  def getPossibleNumericFeatures(rows:Iterable[InputRow]): Seq[String] = {
    val tempo = mutable.HashSet("closesCtxOfClass", "context_frequency",
      "evtNegationInTail", "evtSentenceFirstPerson", "evtSentencePastTense", "evtSentencePresentTense", "sentenceDistance", "dependencyDistance")
    val construct = new mutable.HashSet[String]
    rows.foreach(x => {
      x.evt_dependencyTails.foreach(y => construct += y)
      x.ctx_dependencyTails.foreach(z => construct += z)
    })
    tempo.map{x => construct += x}
    val toReturn = new mutable.HashSet[String]
    construct.map{ x => {
      val min = x + "_min"
      toReturn += min
      val max = x+"_max"
      toReturn += max
      val mean = x + "_mean"
      toReturn += mean
    }}
    toReturn.toSeq
  }

  def createData(features:Seq[String], aggRows:Iterable[AggregatedRow]):Array[Array[Double]] = {
    val row = new mutable.HashSet[Double]
    val dataFrame = new mutable.HashSet[Array[Double]]
    aggRows.map(x => {
      val currentAggFeatures = x.featureGroups
      val featureNames = currentAggFeatures.map(_.name).toSet
      val absentFeatures = features.toSet -- (featureNames ++ Seq(""))
      currentAggFeatures.map(c => {
        row += c.mean
        row += c.min
        row += c.max
      })
      var temp = List.fill(absentFeatures.size)(0.0)
      row ++= temp
      val array = row.toArray
      dataFrame += array
      row.clear()
    })
    dataFrame.toArray
  }
}
