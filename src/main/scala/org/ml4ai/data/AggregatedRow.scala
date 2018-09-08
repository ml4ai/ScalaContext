package org.ml4ai.data

case class AggregatedRow(featureGroups:Iterable[AggregatedFeature], label:Option[Boolean]) extends Iterable[(String, Double)]{
  def toFeatures: Iterable[(String, Double)] = featureGroups.flatten

  override def iterator: Iterator[(String, Double)] = this.toFeatures.iterator
}

object AggregatedRow {
  // TODO: Code this with groupBy and mapValues
  // The key sould be: PMCID, EvtID, CtxID
  def fromRows(rows:TraversableOnce[InputRow]):Map[(String, String, String), AggregatedRow] = ???
}

// TODO: pass the correct parameters to the constructor
case class AggregatedFeature(name:String, mean:Double, min:Double, max:Double) extends Iterable[(String, Double)] {
  def toFeatures: Map[String, Double] = {
    Map(s"${name}_mean" -> mean, s"${name}_min" -> min, s"${name}_max" -> mean)
  }

  override def iterator: Iterator[(String, Double)] = this.toFeatures.iterator
}
