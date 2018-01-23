package fgsdmm

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestAdjustedRandIndex(t *testing.T) {
	Convey("Given sample true/predicted labels x and y", t, func() {
		x := []int{0, 0, 0, 1, 1, 1}
		y := []int{0, 0, 1, 1, 2, 2}

		Convey("ARI(x, y) should be correct.", func() {
			ari := AdjustedRandIndex(x, y)
			So(ari, ShouldAlmostEqual, 0.242424242424)
		})

		Convey("ARI(y, x) should be correct.", func() {
			ari := AdjustedRandIndex(y, x)
			So(ari, ShouldAlmostEqual, 0.242424242424)
		})

	})
	Convey("Given identical label sets", t, func() {
		x := []int{0, 0, 0, 1, 1, 1}
		Convey("ARI(x, x) should equal 1.", func() {
			ari := AdjustedRandIndex(x, x)
			So(ari, ShouldAlmostEqual, 1.0)
		})
	})
	Convey("Given disjoint label sets", t, func() {
		x := []int{0, 0, 1, 1}
		y := []int{0, 1, 2, 3}
		ari := AdjustedRandIndex(x, y)
		Convey("ARI(x, y) should equal 0.", func() {
			So(ari, ShouldAlmostEqual, 0.0)
		})
	})
}
