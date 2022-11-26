package mlp

type LayerBuilder interface {
	New(in, out int) Layer
}

type LinearBuilder struct{}

func (bld LinearBuilder) New(in, out int) Layer {
	return NewLinear(in, out)
}
