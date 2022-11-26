package xor

import (
	"context"
	"math/rand"
	"simlpe/mlp"
	"testing"
	"time"

	. "github.com/smartystreets/goconvey/convey"
)

func TestMain(t *testing.T) {
	ctx := context.Background()

	Convey("main", t, func() {
		rand.Seed(42)
		net := mlp.NewNetwork(0.3, 2)                       // input layer (2 neurons)
		net.AddLayer(mlp.LinearBuilder{}, 3, mlp.Sigmoid{}) // hidden layer (3 neurons)
		net.AddLayer(mlp.LinearBuilder{}, 1, mlp.Sigmoid{}) // output layer (1 neuron)

		xData := [][]float64{
			{0, 0},
			{1, 0},
			{0, 1},
			{1, 1},
		}
		yData := [][]float64{{0}, {1}, {1}, {0}}

		net.Stop.OnEpoch(10000)
		net.Stop.OnDuration(time.Second)
		net.Stop.OnMeanSquaredError(0.0001)

		_, err := net.Train(ctx, xData, yData)
		So(err, ShouldBeNil)

		result := [][]float64{
			net.Predict([]float64{0, 0}),
			net.Predict([]float64{1, 0}),
			net.Predict([]float64{0, 1}),
			net.Predict([]float64{1, 1}),
		}
		So(result, ShouldHaveLength, 4)
		So(result[0][0], ShouldAlmostEqual, 0, 0.03)
		So(result[1][0], ShouldAlmostEqual, 1, 0.03)
		So(result[2][0], ShouldAlmostEqual, 1, 0.03)
		So(result[3][0], ShouldAlmostEqual, 0, 0.03)
	})
}
