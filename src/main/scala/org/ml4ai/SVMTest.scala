package org.ml4ai

import org.clulab.learning.{LibSVMClassifier, LinearKernel, RVFDataset, RVFDatum}
import org.clulab.struct.Counter

object SVMTest extends App{

  def mkRVFDatum[L](label:L, features:List[String]):RVFDatum[L, String] = {
    val c = new Counter[String]
    for(f <- features) c.incrementCount(f)
    new RVFDatum[L, String](label, c)
  }


  val classifier = new LibSVMClassifier[String, String](LinearKernel)

  val dataset = new RVFDataset[String, String]()

  val d1 = mkRVFDatum("+", List("good", "great", "good"))
  val d2 = mkRVFDatum("-", List("bad", "awful"))
  val d3 = mkRVFDatum("~", List("meh", "soso"))

  dataset += d1
  dataset += d2
  dataset += d3
  classifier.train(dataset)

  val dn = mkRVFDatum("+", List("good", "great", "bad", "new"))
  println(classifier.classOf(d1))
  println(classifier.classOf(d2))
  println(classifier.classOf(dn))
}
