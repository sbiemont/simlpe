package mlp

import (
	"encoding/json"
	"fmt"
)

// activatorLayer is a layer-like build from an activator
type activatorLayer struct {
	act Activator
}

// newActivatorLayer builds a new layer from an activator
func newActivatorLayer(act Activator) activatorLayer {
	return activatorLayer{
		act: act,
	}
}

// FeedForward activates all input values one by one
// yi = activ(xi)
func (al activatorLayer) FeedForward(x []float64) []float64 {
	y := newVector(len(x))
	return y.iter(func(i int) {
		y[i] = al.act.Activ(x[i])
	})
}

// BackPropagation derivates all values one by one and multiplies par the y gradient
// yi = yGrad_i * deriv(xi)
func (al activatorLayer) BackPropagation(x, yGrad []float64) []float64 {
	xGrad := newVector(len(x))
	return xGrad.iter(func(i int) {
		xGrad[i] = yGrad[i] * al.act.Deriv(x[i])
	})
}

// Update does nothing
func (al activatorLayer) Update(learninRate float64) {
	// No processing
}

func (al activatorLayer) Type() string {
	return "activator"
}

// for marshal/unmarshal an activation layer
type exportActivationLayer struct {
	Fct string `json:"fct"`
}

func (al activatorLayer) MarshalJSON() ([]byte, error) {
	return json.Marshal(exportActivationLayer{
		Fct: al.act.String(),
	})
}

func (al *activatorLayer) UnmarshalJSON(data []byte) error {
	var exp exportActivationLayer
	err := json.Unmarshal(data, &exp)
	if err != nil {
		return err
	}

	// Convert activator
	var act Activator
	switch exp.Fct {
	case "sigmoid":
		act = Sigmoid{}
	case "htan":
		act = Htan{}
	case "relu":
		act = ReLU{}
	default:
		return fmt.Errorf("unknown activator %q", exp.Fct)
	}

	// Fill layer
	*al = newActivatorLayer(act)
	return nil
}
