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
        val validationIds = shuffled.slice(0,valSize)
        val trainingIds = shuffled.slice(valSize, shuffled.size)
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
}
