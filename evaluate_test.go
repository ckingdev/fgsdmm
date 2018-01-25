package fgsdmm

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestAdjustedRandIndex(t *testing.T) {
	Convey("Given sample true/predicted labels x and y", t, func() {
		c := &Corpus{
			Docs: []Document{
				{Label: 0},
				{Label: 0},
				{Label: 0},
				{Label: 1},
				{Label: 1},
				{Label: 1},
			},
		}
		y := []int{0, 0, 1, 1, 2, 2}

		Convey("ARI(x, y) should be correct.", func() {
			//ari := AdjustedRandIndex(x, y)
			ari := c.AdjustedRandIndex(y)
			So(ari, ShouldAlmostEqual, 0.242424242424)
		})
	})
	Convey("Given identical label sets", t, func() {
		c := &Corpus{
			Docs: []Document{
				{Label: 0},
				{Label: 0},
				{Label: 0},
				{Label: 1},
				{Label: 1},
				{Label: 1},
			},
		}
		x := []int{0, 0, 0, 1, 1, 1}
		Convey("ARI(x, x) should equal 1.", func() {
			ari := c.AdjustedRandIndex(x)
			So(ari, ShouldAlmostEqual, 1.0)
		})
	})
	Convey("Given disjoint label sets", t, func() {
		y := []int{0, 1, 2, 3}
		c := &Corpus{
			Docs: []Document{
				{Label: 0},
				{Label: 0},
				{Label: 1},
				{Label: 1},
			},
		}
		ari := c.AdjustedRandIndex(y)
		Convey("ARI(x, y) should equal 0.", func() {
			So(ari, ShouldAlmostEqual, 0.0)
		})
	})
}
