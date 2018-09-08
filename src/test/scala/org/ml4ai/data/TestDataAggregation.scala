package org.ml4ai.data

import org.scalatest.{FlatSpec, Matchers}

class TestDataAggregation extends FlatSpec with Matchers{

  import TestDataAggregation._


  "The raw data" should "be balanced correctly" in {
    val balanced = Balancer.balanceByPaper(data, 1)

    val positives = balanced count {
      row =>
        row.label match {
          case Some(true) => true
          case _ => false
        }
    }

    positives should equal (34468)

    val negatives = balanced count {
      row =>
        row.label match {
          case Some(false) => true
          case _ => false
        }
    }

    negatives should equal (34468)
  }

  it should "be balanced correctly again with another ratio" ignore {
    val balanced = Balancer.balanceByPaper(data, 3)

    val positives = balanced count {
      row =>
        row.label match {
          case Some(true) => true
          case _ => false
        }
    }

    positives should equal (34468)

    val negatives = balanced count {
      row =>
        row.label match {
          case Some(false) => true
          case _ => false
        }
    }

    negatives should equal (34468)
  }

  "The aggregated data" should "have the right number of rows" in {
    val aggregated = AggregatedRow.fromRows(Balancer.balanceByPaper(data, 1))
    aggregated should have size 22761
  }

  it should "have the correct number of instances by label" in {
    val aggregated = AggregatedRow.fromRows(Balancer.balanceByPaper(data, 1))

    val positives = aggregated count {
      case (_, row) =>
        row.label match {
          case Some(true) => true
          case _ => false
        }
    }

    positives should be (2556)

    val negatives = aggregated count {
      case (_, row) =>
        row.label match {
          case Some(false) => true
          case _ => false
        }
    }

    negatives should be (20205)
  }



}

object TestDataAggregation {
  val data: Seq[InputRow] = InputRow.fromStream(TestInputCSV.getGzipedStream)
}
