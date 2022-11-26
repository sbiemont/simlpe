package mlp

import (
	"encoding/json"
	"fmt"
)

// Layer is the required interface for building a layer
type Layer interface {
	FeedForward(x []float64) []float64
	BackPropagation(x, yGrad []float64) []float64
	Update(learningRate float64)
	Type() string
}

// Linear layer applies a linear transformation
// y = x.w + b
type Linear struct {
	biaises     vector // d = out
	biaisesGrad vector // d = out

	weights     matrix // d = in x out
	weightsGrad matrix // d = in x out
}

// NewLinear allocates the linear layer
func NewLinear(in, out int) Linear {
	weights := newMatrix(in, out)
	return Linear{
		weights:     weights.iter(func(i, j int) { weights[i][j] = normRandom(1, 0) }),
		weightsGrad: newMatrix(in, out).zeros(),
		biaises:     newVector(out).zeros(),
		biaisesGrad: newVector(out).zeros(),
	}
}

// FeedForward applies the linear transformation
// y = x.w + b
func (ln Linear) FeedForward(x []float64) []float64 {
	y := newVector(len(ln.biaises)).zeros()
	ln.weights.iter(func(i, j int) {
		y[j] += x[i] * ln.weights[i][j]
	})
	return y.iter(func(j int) {
		y[j] += ln.biaises[j]
	})
}

// BackPropagation computes the x gradient
func (ln Linear) BackPropagation(x, yGrad []float64) []float64 {
	// gradient bias += grad
	ln.biaisesGrad.iter(func(j int) {
		ln.biaisesGrad[j] += yGrad[j]
	})

	// gradient weight += x * yGrad
	ln.weightsGrad.iter(func(i, j int) {
		ln.weightsGrad[i][j] += x[i] * yGrad[j]
	})

	// gradient x = weights * yGrad
	xGrad := newVector(len(x)).zeros()
	ln.weights.iter(func(i, j int) {
		xGrad[i] += ln.weights[i][j] * yGrad[j]
	})
	return xGrad
}

// Update weights and biases
func (ln Linear) Update(learningRate float64) {
	// biases -= rate * gradient biases
	ln.biaises.iter(func(j int) {
		ln.biaises[j] -= learningRate * ln.biaisesGrad[j]
	})

	// weights -= rate * gradient weights
	ln.weights.iter(func(i, j int) {
		ln.weights[i][j] -= learningRate * ln.weightsGrad[i][j]
	})

	// Clear gradients
	ln.biaisesGrad.zeros()
	ln.weightsGrad.zeros()
}

func (ln Linear) Type() string {
	return "linear"
}

type exportLayer struct {
	Weights matrix `json:"weights"`
	Biaises vector `json:"biaises"`
}

func (ln Linear) MarshalJSON() ([]byte, error) {
	return json.Marshal(exportLayer{
		Weights: ln.weights,
		Biaises: ln.biaises,
	})
}

func (ln *Linear) UnmarshalJSON(data []byte) error {
	// Load data
	exp := exportLayer{}
	err := json.Unmarshal(data, &exp)
	if err != nil {
		return err
	}

	// Check dimensions
	if len(exp.Biaises) == 0 || len(exp.Weights) == 0 || len(exp.Weights[0]) == 0 {
		return fmt.Errorf("cannot load 0 lentgh matrix")
	}

	// Copy or init
	ln.biaises = exp.Biaises
	ln.weights = exp.Weights
	ln.biaisesGrad = newVector(len(exp.Biaises)).zeros()
	ln.weightsGrad = newMatrix(len(exp.Weights), len(exp.Weights[0])).zeros()
	return nil
}
