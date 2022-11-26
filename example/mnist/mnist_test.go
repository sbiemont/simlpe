package mnist

import (
	"context"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"simlpe/mlp"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

// convert images to input for the network network
func imageToInput(images [][]byte) [][]float64 {
	inputs := make([][]float64, len(images))
	for i, image := range images {
		fltImage := make([]float64, len(image))
		for j, px := range image {
			fltImage[j] = float64(px) / 255.0
		}
		inputs[i] = fltImage
	}
	return inputs
}

// convert labels to output for the network network
func labelsToOutput(labels []byte) [][]float64 {
	outputs := make([][]float64, len(labels))
	for i, label := range labels {
		output := make([]float64, 10) // fill with 0
		output[label] = 1
		outputs[i] = output
	}
	return outputs
}

// re-convert output to label
func outputToLabel(output []float64) byte {
	result := 0
	prev := output[0]
	for i, out := range output[1:] {
		if out > prev {
			result = i + 1
			prev = out
		}
	}
	return byte(result)
}

// Test train + write output
func TestTrain(t *testing.T) {
	ctx := context.Background()

	Convey("nnet", t, func() {
		rand.Seed(42)

		// Read train database
		db, err := newDatabase(TrainLabelsFile, TrainImagesFile)
		So(err, ShouldBeNil)
		inputs := imageToInput(db.images)
		outputs := labelsToOutput(db.labels)

		// Train
		net := mlp.NewNetwork(0.25, db.h*db.w)
		net.AddLayer(mlp.LinearBuilder{}, 40, mlp.Sigmoid{})
		net.AddLayer(mlp.LinearBuilder{}, 10, mlp.Sigmoid{})
		net.Stop.OnEpoch(2)

		stop, err := net.Train(ctx, inputs, outputs)
		So(err, ShouldBeNil)
		fmt.Println(stop)

		str, err := net.MarshalJSON()
		So(err, ShouldBeNil)
		f, err := os.Create("nnet.json") // creates a file at current directory
		So(err, ShouldBeNil)
		_, errWrite := f.Write(str)
		So(errWrite, ShouldBeNil)
	})
}

func TestCheck(t *testing.T) {
	Convey("output", t, func() {
		So(outputToLabel([]float64{0, 0.1, 0.2, 0.3, 0.4}), ShouldEqual, 4)
		So(outputToLabel([]float64{0.4, 0.3, 0.2, 0.1, 0}), ShouldEqual, 0)
		So(outputToLabel([]float64{0.4, 0.3, 0.9, 0.5, 0.6}), ShouldEqual, 2)
	})

	Convey("check", t, func() {
		// Read file
		content, err := ioutil.ReadFile("nnet.json")
		So(err, ShouldBeNil)

		// Read net
		net := mlp.Network{}
		errJSON := net.UnmarshalJSON(content)
		So(errJSON, ShouldBeNil)

		// Read check database
		dbCheck, errCheck := newDatabase(TestLabelsFile, TestImagesFile)
		So(errCheck, ShouldBeNil)
		inputsCheck := imageToInput(dbCheck.images)

		// Check
		var ok int
		for i, input := range inputsCheck {
			output := net.Predict(input)
			if outputToLabel(output) == dbCheck.labels[i] {
				ok++
			}
		}

		So(len(dbCheck.labels), ShouldEqual, 10000)
		So(ok, ShouldBeGreaterThan, 9300) // > 93%
		fmt.Println(ok)
	})
}
