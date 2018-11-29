package org.ml4ai.data
import scala.collection.mutable


import scala.io.{BufferedSource, Source}
case class FoldMaker(groupedFeatures: Map[(String, String, String), AggregatedRow]) extends Iterable[(Array[Int], Array[Int], Array[Int])]{
  def toFolds:Iterable[(Array[Int], Array[Int], Array[Int])] = new mutable.HashSet[(Array[Int], Array[Int], Array[Int])]()
  override def iterator:Iterator[(Array[Int], Array[Int], Array[Int])] = this.toFolds.iterator
}

object FoldMaker {

  def createData(features:Seq[String], aggRows:Iterable[AggregatedRow]):Array[Array[Double]] = {
    val row = new mutable.ListBuffer[Double]
    val dataFrame = new mutable.HashSet[Array[Double]]
    aggRows.map(x => {
      val currentAggFeatures = x.featureGroups
      val featureNames = currentAggFeatures.map(_.name)
      val appendedFeat = new mutable.ListBuffer[String]
      featureNames.map(x => {
        appendedFeat += x + "_mean"
        appendedFeat += x + "_max"
        appendedFeat += x + "_min"
      })
      val absentFeatures = features.toSet -- (appendedFeat ++ Seq(""))
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

  def createSentenceDistData(aggRows:Iterable[((String, String, String), AggregatedRow)]):Iterable[((String, String, String), AggregatedRow)] = {
    val toReturn = new mutable.ListBuffer[((String, String, String), AggregatedRow)]
    aggRows.map(x => {
      val keys = x._1
      val row = x._2
      val feat = row.featureGroups.filter(_.name == "sentenceDistance")
      val label = row.label
      val aggRow = AggregatedRow(feat, label)
      val entry = (keys, aggRow)
      toReturn += entry
    })
    toReturn
  }

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

}
