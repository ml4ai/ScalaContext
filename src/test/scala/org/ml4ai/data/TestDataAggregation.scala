package org.ml4ai.data

// need to include tests for AggegatedRowNew class. I removed the existing once because old usage of AggegatedRow was deprecated
// There will be only one instance of AggregatedRow, so please import that for testing like so:
// import org.ml4ai.data.utils.AggregatedRow
import org.ml4ai.data.utils.{Balancer, InputRow}
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



}

object TestDataAggregation {
  val data: Seq[InputRow] = InputRow.fromStream(TestInputCSV.getGzipedStream)
}
