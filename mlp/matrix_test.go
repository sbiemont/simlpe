package mlp

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestVector(t *testing.T) {
	Convey("vector", t, func() {
		Convey("zeros", func() {
			v := newVector(3).zeros()
			So(v, ShouldResemble, vector{0, 0, 0})
		})

		Convey("iter", func() {
			Convey("simple iter", func() {
				v := newVector(3)
				v.iter(func(i int) { v[i] = float64(i + 1) })
				So(v, ShouldResemble, vector{1, 2, 3})
			})

			Convey("multi iter", func() {
				v := newVector(3)
				v.iter(func(i int) { v[i] = 1 }).iter(func(i int) { v[i] += 2 })
				So(v, ShouldResemble, vector{3, 3, 3})
			})
		})
	})
}

func TestMatrix(t *testing.T) {
	Convey("matrix", t, func() {
		Convey("zeros", func() {
			m := newMatrix(2, 3).zeros()
			So(m, ShouldResemble, matrix{
				{0, 0, 0},
				{0, 0, 0},
			})
		})

		Convey("iter", func() {
			Convey("simple iter", func() {
				m := newMatrix(2, 3)
				m.iter(func(i, j int) { m[i][j] = float64(10*(i+1) + j + 1) })
				So(m, ShouldResemble, matrix{
					{11, 12, 13},
					{21, 22, 23},
				})
			})

			Convey("multi iter", func() {
				m := newMatrix(2, 3)
				m.iter(func(i, j int) { m[i][j] = 1 }).iter(func(i, j int) { m[i][j] += 2 })
				So(m, ShouldResemble, matrix{
					{3, 3, 3},
					{3, 3, 3},
				})
			})
		})

		Convey("matrix x vector", func() {
			m := matrix{
				{1, 2, 3},
				{4, 5, 6},
			}
			v := vector{1, 2, 3}

			res := vector{0, 0}
			m.iter(func(i, j int) { res[i] += m[i][j] * v[j] })
			So(res, ShouldResemble, vector{1 + 4 + 9, 4 + 10 + 18})
		})

		Convey("vector x matrix", func() {
			m := matrix{
				{1, 2, 3},
				{4, 5, 6},
			}
			v := vector{1, 2}

			res := vector{0, 0, 0}
			m.iter(func(i, j int) { res[j] += v[i] * m[i][j] })
			So(res, ShouldResemble, vector{
				1 + 8, 2 + 10, 3 + 12,
			})
		})

		Convey("vector x vector => scalar", func() {
			v1 := vector{1, 2, 3}
			v2 := vector{1, 2, 3}

			res := 0.0
			v1.iter(func(i int) { res += v1[i] * v2[i] })
			So(res, ShouldEqual, 1+4+9)
		})

		Convey("vector x vector => matrix", func() {
			v1 := vector{1, 2}
			v2 := vector{1, 2}

			res := newMatrix(2, 2).zeros()
			v1.iter(func(i int) {
				v2.iter(func(j int) {
					res[i][j] += v1[i] * v2[j]
				})
			})
			So(res, ShouldResemble, matrix{
				{1, 2},
				{2, 4},
			})
		})
	})
}
