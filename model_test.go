package fgsdmm

import (
	"fmt"
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func shouldAllAlmostEqual(actual interface{}, expected ...interface{}) string {
	del := 0.0001
	if len(expected) == 2 {
		del = expected[1].(float64)
	}
	actualF64 := actual.([]float64)
	expectedF64 := expected[0].([]float64)

	for i := range actualF64 {
		if math.Abs(actualF64[i]-expectedF64[i]) >= del {
			return fmt.Sprintf("Expected element %v to be almost equal to %v, got %v", i, actualF64[i], expectedF64[i])
		}
	}
	return ""
}

func TestCluster(t *testing.T) {
	Convey("Given a cluster and a document", t, func() {
		c := &Cluster{
			TknCts: make(map[int]int),
			NDoc:   0,
			NTkn:   0,
		}
		d1 := &Document{
			TknIDs: []int{0, 1, 2},
			TknCts: []int{1, 2, 1},
			NTkn:   4,
		}

		d2 := &Document{
			TknIDs: []int{1, 2, 3},
			TknCts: []int{3, 2, 1},
			NTkn:   6,
		}
		s := &State{
			Clusters: []*Cluster{c},
			Labels:   []int{0, 0},
		}
		Convey("When the document is added to the cluster", func() {
			s.clusterAdd(0, 0, d1)

			Convey("The number of documents in the cluster should be 1.", func() {
				So(c.NDoc, ShouldEqual, 1)
			})

			Convey("The cluster's number of tokens should equal the document's.", func() {
				So(c.NTkn, ShouldEqual, d1.NTkn)
			})

			Convey("The cluster token counts should equal the document's.", func() {
				for i, tknID := range d1.TknIDs {
					So(c.TknCts[tknID], ShouldEqual, d1.TknCts[i])
				}
			})
		})
		Convey("When two documents are added to the cluster", func() {
			s.clusterAdd(0, 0, d1)
			s.clusterAdd(0, 1, d2)

			Convey("The number of documents in the cluster should be two.", func() {
				So(c.NDoc, ShouldEqual, 2)
			})

			Convey("The number of tokens in the cluster should equal the document sum", func() {
				So(c.NTkn, ShouldEqual, d1.NTkn+d2.NTkn)
			})

			Convey("and when the second document is removed", func() {
				clusterRemove(c, d2)
				Convey("The number of documents in the cluster should be 1.", func() {
					So(c.NDoc, ShouldEqual, 1)
				})

				Convey("The cluster's number of tokens should equal the first document's.", func() {
					So(c.NTkn, ShouldEqual, d1.NTkn)
				})

				Convey("The cluster token counts should equal the first document's.", func() {
					for i, tknID := range d1.TknIDs {
						So(c.TknCts[tknID], ShouldEqual, d1.TknCts[i])
					}
				})
			})
		})
	})
}

func TestExpNormalize(t *testing.T) {
	Convey("Given a vector of log weights", t, func() {
		logW := []float64{100, 102, 99}

		Convey("The normalization of the vector should be correct.", func() {
			W := expNormalize(logW)
			So(W, shouldAllAlmostEqual, []float64{.135335, 1.0, .049787068})
		})
	})
}

func TestScoreNonEmpty(t *testing.T) {
	Convey("Given a model, cluster and a document", t, func() {
		model := NewFGSDMM(&HyperParams{
			KMax:  5,
			Alpha: 0.1,
			Beta:  0.1,
		})
		model.KNon = 2
		model.Corpus = &Corpus{
			V: 5,
		}

		c := &Cluster{
			TknCts: map[int]int{0: 2, 1: 2, 2: 1},
			NDoc:   2,
			NTkn:   5,
		}
		d := &Document{
			TknIDs: []int{0, 1, 2},
			TknCts: []int{1, 1, 1},
			NTkn:   3,
		}
		Convey("The score for the document in the cluster should be correct.", func() {
			score := model.scoreNonEmpty(c, d)
			So(score, ShouldAlmostEqual, -3.270331, 0.0001)
		})

		Convey("The score for the document in an empty cluster should be correct", func() {
			score := model.scoreEmpty(d)
			So(score, ShouldAlmostEqual, -8.7403367427)
		})
	})
}
