package mlp

import (
	"math/rand"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNetwork(t *testing.T) {
	Convey("network", t, func() {
		Convey("marshal, unmarshal", func() {
			rand.Seed(42)

			// Export net1
			net1 := NewNetwork(0.42, 2)
			net1.AddLayer(LinearBuilder{}, 5, Htan{})
			net1.AddLayer(LinearBuilder{}, 6, ReLU{})
			net1.AddLayer(LinearBuilder{}, 1, Sigmoid{})

			js1, err1 := net1.MarshalJSON()
			So(err1, ShouldBeNil)

			// Reload into net2
			net2 := Network{}
			err2 := net2.UnmarshalJSON(js1)
			So(err2, ShouldBeNil)

			// Compare both
			So(net2, ShouldResemble, net1)
		})

		Convey("train and cancel context", func() {
			net1 := NewNetwork(0.42, 2)
			net1.AddLayer(LinearBuilder{}, 5, Htan{})
			net1.AddLayer(LinearBuilder{}, 6, ReLU{})
			net1.AddLayer(LinearBuilder{}, 1, Sigmoid{})
		})
	})
}
