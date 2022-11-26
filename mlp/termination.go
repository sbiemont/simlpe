package mlp

import (
	"fmt"
	"strings"
	"time"
)

type Termination struct {
	epoch            *int
	duration         *time.Duration
	meanSquaredError *float64
}

func (tn *Termination) OnEpoch(epoch int) {
	tn.epoch = &epoch
}

func (tn *Termination) OnDuration(duration time.Duration) {
	tn.duration = &duration
}

func (tn *Termination) OnMeanSquaredError(mse float64) {
	tn.meanSquaredError = &mse
}

func (tn Termination) hasReached(current Termination) bool {
	empty := tn.epoch == nil && tn.duration == nil && tn.meanSquaredError == nil

	switch {
	case empty:
		return true // stop if all empty
	case tn.epoch != nil && current.epoch != nil &&
		*current.epoch >= *tn.epoch:
		return true // stop if epoch reached
	case tn.duration != nil && current.duration != nil &&
		*current.duration >= *tn.duration:
		return true // stop if duration reached
	case tn.meanSquaredError != nil && current.meanSquaredError != nil &&
		*current.meanSquaredError <= *tn.meanSquaredError:
		return true // stop if mse reached
	default:
		return false
	}
}

func (tn Termination) String() string {
	var str []string
	if tn.epoch != nil {
		str = append(str, fmt.Sprintf("epoch: %d", *tn.epoch))
	}
	if tn.duration != nil {
		str = append(str, fmt.Sprintf("duration: %s", *tn.duration))
	}
	if tn.meanSquaredError != nil {
		str = append(str, fmt.Sprintf("error: %f", *tn.meanSquaredError))
	}
	return strings.Join(str, ", ")
}
