package org.ml4ai.data.utils
import com.typesafe.config.ConfigFactory
import java.io.InputStream

import scala.collection.mutable
import scala.io.Source
import scala.io.BufferedSource

case class AggregatedRow(
                             sentenceIndex:Int,
                             PMCID:String,
                             EvtID: String,
                             CtxID: String,
                             label: Option[Boolean],
                             featureGroups: Array[Double],
                             featureGroupNames:Array[String])


object AggregatedRow {
  val config = ConfigFactory.load()
  val hardCodedFeaturePath = config.getString("features.hardCodedFeatures")
  private val listOfSpecificFeatures = CodeUtils.readHardcodedFeaturesFromFile(hardCodedFeaturePath)
  private def allOtherFeatures(headers:Seq[String]): Set[String] = headers.toSet -- (listOfSpecificFeatures ++ Seq(""))
  private def indices(headers:Seq[String]): Map[String, Int] = headers.zipWithIndex.toMap
  def apply(str:String, headers: Seq[String], allOtherFeatures:Set[String], indices:Map[String, Int]):AggregatedRow = {
    val rowData = str.split(",")
    val sentencePos = rowData(0).toInt
    var evt_dependencyTails = new mutable.ListBuffer[Double]
    var ctx_dependencyTails = new mutable.ListBuffer[Double]
    var evt_dependencyFeatures = new mutable.ListBuffer[String]
    var ctx_dependencyFeatures = new mutable.ListBuffer[String]
    val featureGroups = new mutable.ListBuffer[Double]
    val featureNames = new mutable.ListBuffer[String]
    allOtherFeatures foreach {
      case evt:String if evt.startsWith("evtDepTail") =>
        if(rowData(indices(evt)) != "0.0")
          {evt_dependencyTails += (rowData(indices(evt))).toDouble
            evt_dependencyFeatures += evt
          }
      case ctx:String if ctx.startsWith("ctxDepTail") =>
        if(rowData(indices(ctx)) != "0.0")
        {
          ctx_dependencyTails += (rowData(indices(ctx))).toDouble
          ctx_dependencyFeatures += ctx
        }
      case _ => 0.0
    }



    val pmcid = rowData(indices("PMCID"))

    val evt = rowData(indices("EvtID"))
    val ctx = rowData(indices("CtxID"))
    val label = rowData(indices("label"))

    val listOfNumericFeatures = listOfSpecificFeatures.drop(4)
    featureNames ++= listOfNumericFeatures
    listOfNumericFeatures.map(l => {
      val tempVal = rowData(indices(l))
      featureGroups += tempVal.toDouble
    })

    featureGroups ++= evt_dependencyTails
    featureGroups ++= ctx_dependencyTails
    featureNames ++= evt_dependencyFeatures
    featureNames ++= ctx_dependencyFeatures
    AggregatedRow(sentencePos, pmcid, evt, ctx, Some(label.toBoolean), featureGroups.toArray, featureNames.toArray)
  }

  def fromFile(groupedFeaturesFileName: String):(Seq[String], Seq[AggregatedRow]) = {

    val source = Source.fromFile(groupedFeaturesFileName)
    val lines = source.getLines()
    val headers = lines.next() split ","
    val rectifiedHeaders = rectifyWrongFeatures(headers)
    val features = allOtherFeatures(rectifiedHeaders)
    val ixs = indices(rectifiedHeaders)
    val ret = lines.map(l => AggregatedRow(l, rectifiedHeaders, features, ixs)).toList
    source.close()
    (rectifiedHeaders, ret)
  }

  def rectifyWrongFeatures(headers:Seq[String]): Seq[String] = {
    val result = collection.mutable.ListBuffer[String]()
    headers.map(h => if(headers.indexOf(h) == 1) result += "PMCID" else result += h)
    result
  }
}