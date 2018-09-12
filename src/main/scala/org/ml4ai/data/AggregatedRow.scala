package org.ml4ai.data

case class AggregatedRow(featureGroups:Iterable[AggregatedFeature], label:Option[Boolean]) extends Iterable[(String, Double)]{
  def toFeatures: Iterable[(String, Double)] = featureGroups.flatten

  override def iterator: Iterator[(String, Double)] = this.toFeatures.iterator
}

object AggregatedRow {
  // TODO: Code this with groupBy and mapValues
  // The key should be: PMCID, EvtID, CtxID
  // rows.groupBy(r => (r.PMCID, r.EvtID, r.CtxID))
  def fromRows(rows:Iterable[InputRow]):Map[(String, String, String), AggregatedRow] = {
    val groups = rows.groupBy(l => (l.PMCID, l.EvtID, l.CtxID))
    groups mapValues createAggRow

  }

 def createAggRow(rows: Seq[InputRow]):AggregatedRow = {
   val sentenceColl = rows map {r => r.sentenceDistance}
   val List(sent_min, sent_max, sent_avg) = createStats(sentenceColl)
   val aggregatedSent = AggregatedFeature("sentenceDistance", sent_avg, sent_min, sent_max)

   val dependencyColl = rows map {r => r.dependencyDistance}
   val List(d_min,d_max,d_avg) = createStats(dependencyColl)
   val aggregatedDep = AggregatedFeature("dependencyDistance", d_avg, d_min, d_max)

   val evtSentencePresentTenseColl = rows map {r => r.evtSentencePresentTense}
   val List(e_min,e_max,e_avg) = createStats(evtSentencePresentTenseColl)
   val aggregatedPresent  = AggregatedFeature("evtSentencePresentTense", e_avg, e_min, e_max)

   val evtSentencePastTenseColl = rows map {r => r.evtSentencePastTense}
   val List(p_min,p_max,p_avg) = createStats(evtSentencePastTenseColl)
   val aggPast = AggregatedFeature("evtSentencePastTense", p_avg, p_min, p_max)

   val evtSentenceFirstPerson = rows map { r => r.evtSentenceFirstPerson}
   val List(f_min,f_max,f_avg) = createStats(evtSentenceFirstPerson)
   val aggFirst = AggregatedFeature("evtSentenceFirstPerson", f_avg, f_min, f_max)

   val evtNegationInTail = rows map { r => r.evtNegationInTail}
   val List(n_min, n_max, n_avg) = createStats(evtNegationInTail)
   val aggNeg = AggregatedFeature("evtNegationInTail", n_avg, n_min, n_max)

   val context_frequency = rows map { r => r.context_frequency}
   val List(c_min,c_max, c_avg) = createStats(context_frequency)
   val aggCon = AggregatedFeature("context_frequency", c_avg, c_min, c_max)

   val closesCtxOfClass = rows map { r => r.closesCtxOfClass}
   val List(ct_min,ct_max,ct_avg) = createStats(closesCtxOfClass)
   val aggCtx = AggregatedFeature("closesCtxOfClass", ct_avg, ct_min, ct_max)

   val label = rows map {r => r.label}
   val labelForGroup = oneHitAll(label)

   val allAggregatedFeatures: Option[Iterable[AggregatedFeature]] = None
// The basic idea behind the following code is:
   // We initialize an empty Iterable of Aggregated features, call it allAggregatedFeatures
   // Each InputRow (rows) has a ctxDepTail list, and evtDepTail list. Each list has strings, i.e. the names of the features.
   // We need to create aggregated features for each of these features, using values 1.0 for min, max and avg.
   // for each of the aggregated feature thus obtained, we have to append it to the running iterable.
   for(r <- rows) {
     val setOfCtxDep = r.ctx_dependencyTails
     val setOfEvtDep = r.evt_dependencyTails
     setOfCtxDep foreach {s => {
       val agg = AggregatedFeature(s,1.0,1.0,1.0)
       allAggregatedFeatures match {
         case None => Some(agg)
         case Some(s) => {
          // **** Please refer to Balancer.scala for reference. The similar logic had worked just fine yesterday without issues
           // Should be *** allAggregatedFeatures = Some(s ++ agg) ***  according to the correctness of code in Balancer.scala
           // check with Enrique
           //allAggregatedFeatures = Some(s ++ agg)
           Some(s ++ agg)
         }
       }
     }}
  // same logic as above
     setOfEvtDep foreach { e =>
       val aggr = AggregatedFeature(e,1.0,1.0,1.0)
       allAggregatedFeatures match {
         case None => Some(aggr)
         case Some(evt) => {
           Some(evt ++ aggr)
         }
       }
     }
   }

   // Continuing the logic, Now we need to examine the allFeatures iterable. We are yet to add the features above, i.e. sentenceDistance, closestContext, etc.
   // That is what we need to do in the next step.
   // Suppose that for some row, both ctxDepTail and evtDepTail are empty. Then the aggregated row will only contain those 10 features that were specifically created.
   // If they are not empty, then we need to append them to the list of above 10 particular features.
   // After this step, we should have the correct state of allAggregatedFeatures, and are ready to return.
   allAggregatedFeatures foreach println
   allAggregatedFeatures match {
     case None => Some(allAggregatedFeatures ++ aggregatedSent ++
       aggregatedDep ++ aggregatedPresent ++ aggPast ++ aggFirst ++
       aggNeg ++ aggCon ++ aggCtx)
     case Some(s) => Some(s ::: List(aggregatedSent, aggregatedDep, aggregatedPresent, aggPast, aggFirst, aggNeg, aggCon, aggCtx))

   }
   // ToReturn: AggregatedRow
   // For the return part, we should emphasize that allAggregatedFeatures was actually an optional value. It may be none or it may have a valid iterable.
   // We need to examine the cases and return the proper instance of AggregatedRow

   def toBeReturned(s:Option[Iterable[AggregatedFeature]]):AggregatedRow = {
     case Some(a) => AggregatedRow(a, labelForGroup)
     case None => AggregatedRow(None, labelForGroup)
   }

   toBeReturned(allAggregatedFeatures)
 }



  def createStats(nums:Seq[Double]): List[Double] = {
    val min = nums.min
    val max = nums.max
    val avg = nums.sum/nums.size
    List(min,max,avg)
  }

  def oneHitAll(bools:Seq[Option[Boolean]]):Option[Boolean] = {
    bools foreach {l => if(l == Some(true)) Some(true)}
    Some(false)
  }
}

// TODO: pass the correct parameters to the constructor
case class AggregatedFeature(name:String, mean:Double, min:Double, max:Double) extends Iterable[(String, Double)] {
  def toFeatures: Map[String, Double] = {
    Map(s"${name}_mean" -> mean, s"${name}_min" -> min, s"${name}_max" -> mean)
  }

  override def iterator: Iterator[(String, Double)] = this.toFeatures.iterator
}



