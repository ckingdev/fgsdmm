package fgsdmm

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

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

func TestScoreNonEmpty(t *testing.T) {
	Convey("Given a model, cluster and a document", t, func() {
		model := NewFGSDMM(&HyperParams{
			KMax:  5,
			Alpha: 0.1,
			Beta:  0.1,
		})
		model.KNon = 2
		model.Corpus = Corpus{
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
			So(score, ShouldAlmostEqual, 0.03799384615384616, 0.0001)
		})

		Convey("The score for the document in an empty cluster should be correct", func() {
			score := model.scoreEmpty(d)
			So(score, ShouldAlmostEqual, 0.00016000000000000007)
		})
	})
}
