package org.ml4ai.data
import scala.collection.mutable
case class AggregatedRow(featureGroups:Iterable[AggregatedFeature], label:Option[Boolean]) extends Iterable[(String, Double)]{
  def toFeatures: Iterable[(String, Double)] = featureGroups.flatten

  override def iterator: Iterator[(String, Double)] = this.toFeatures.iterator
}

object AggregatedRow {
  // TODO: Code this with groupBy and mapValues
  // The key should be: PMCID, EvtID, CtxID
  // rows.groupBy(r => (r.PMCID, r.EvtID, r.CtxID))
  def fromRows(rows: Iterable[InputRow]): Map[(String, String, String), AggregatedRow] = {
    val groups = rows.groupBy(l => (l.PMCID, l.EvtID, l.CtxID))
    val result = groups mapValues createAggRow
    result
  }

  def createAggRow(rows: Iterable[InputRow]): AggregatedRow = {

    val sentenceColl = rows map { r => r.sentenceDistance }
    val List(sent_min, sent_max, sent_avg) = createStats(sentenceColl)
    val aggregatedSent = AggregatedFeature("sentenceDistance", sent_avg, sent_min, sent_max)

    val dependencyColl = rows map { r => r.dependencyDistance }
    val List(d_min, d_max, d_avg) = createStats(dependencyColl)
    val aggregatedDep = AggregatedFeature("dependencyDistance", d_avg, d_min, d_max)

    val evtSentencePresentTenseColl = rows map { r => r.evtSentencePresentTense }
    val List(e_min, e_max, e_avg) = createStats(evtSentencePresentTenseColl)
    val aggregatedPresent = AggregatedFeature("evtSentencePresentTense", e_avg, e_min, e_max)

    val evtSentencePastTenseColl = rows map { r => r.evtSentencePastTense }
    val List(p_min, p_max, p_avg) = createStats(evtSentencePastTenseColl)
    val aggPast = AggregatedFeature("evtSentencePastTense", p_avg, p_min, p_max)

    val evtSentenceFirstPerson = rows map { r => r.evtSentenceFirstPerson }
    val List(f_min, f_max, f_avg) = createStats(evtSentenceFirstPerson)
    val aggFirst = AggregatedFeature("evtSentenceFirstPerson", f_avg, f_min, f_max)

    val evtNegationInTail = rows map { r => r.evtNegationInTail }
    val List(n_min, n_max, n_avg) = createStats(evtNegationInTail)
    val aggNeg = AggregatedFeature("evtNegationInTail", n_avg, n_min, n_max)

    val context_frequency = rows map { r => r.context_frequency }
    val List(c_min, c_max, c_avg) = createStats(context_frequency)
    val aggCon = AggregatedFeature("context_frequency", c_avg, c_min, c_max)

    val closesCtxOfClass = rows map { r => r.closesCtxOfClass }
    val List(ct_min, ct_max, ct_avg) = createStats(closesCtxOfClass)
    val aggCtx = AggregatedFeature("closesCtxOfClass", ct_avg, ct_min, ct_max)

    val label = rows map { r => r.label }
    val labelForGroup = oneHitAll(label)

    val allAggregatedFeatures = new mutable.HashSet[AggregatedFeature]
    allAggregatedFeatures += aggregatedSent
    allAggregatedFeatures += aggregatedDep
    allAggregatedFeatures += aggregatedPresent
    allAggregatedFeatures += aggPast
    allAggregatedFeatures += aggFirst
    allAggregatedFeatures += aggNeg
    allAggregatedFeatures += aggCon
    allAggregatedFeatures += aggCtx

    for (r <- rows) {
      val ctxDep = r.ctx_dependencyTails
      val evtDep = r.evt_dependencyTails
      val contextFeatures = makeAggrFeatureSet(ctxDep)
      val eventFeatures = makeAggrFeatureSet(evtDep)
      allAggregatedFeatures ++ contextFeatures
      allAggregatedFeatures ++ eventFeatures
    }


    AggregatedRow(allAggregatedFeatures, Some(labelForGroup))
  }


  def makeAggrFeatureSet(set: Set[String]): mutable.HashSet[AggregatedFeature] = {
    val toReturn = new mutable.HashSet[AggregatedFeature]
    set foreach { feature_name => {
      val agg = AggregatedFeature(feature_name, 1.0, 1.0, 1.0)
      toReturn += agg
    }
    }
    toReturn
  }


  def createStats(nums: Iterable[Double]): List[Double] = {
    val min = nums.min
    val max = nums.max
    val avg = nums.sum / nums.size
    List(min, max, avg)
  }

  def oneHitAll(bools: Iterable[Option[Boolean]]): Boolean = {

    //bools.exists { case Some(v) => v }
    val bList = bools map {b => b match {
      case Some(v) => v
      case None => false
    }}
    bList.foldLeft(false) (_ || _)
  }

}

// TODO: pass the correct parameters to the constructor
case class AggregatedFeature(name:String, mean:Double, min:Double, max:Double) extends Iterable[(String, Double)] {
  def toFeatures: Map[String, Double] = {
    Map(s"${name}_mean" -> mean, s"${name}_min" -> min, s"${name}_max" -> mean)
  }

  override def iterator: Iterator[(String, Double)] = this.toFeatures.iterator
}



