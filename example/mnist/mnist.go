package mnist

import (
	"encoding/binary"
	"os"
)

const (
	TrainImagesFile = "train-images.idx3-ubyte"
	TrainLabelsFile = "train-labels.idx1-ubyte"
	TestImagesFile  = "t10k-images.idx3-ubyte"
	TestLabelsFile  = "t10k-labels.idx1-ubyte"
)

// database of labels and images
type database struct {
	w, h   int
	images [][]byte
	labels []byte
}

// init database using labels + images path files
func newDatabase(pathLabels, pathImages string) (*database, error) {
	labels, err := readLabels(pathLabels)
	if err != nil {
		return nil, err
	}
	images, w, h, err := readImages(pathImages)
	if err != nil {
		return nil, err
	}
	return &database{
		w:      w,
		h:      h,
		images: images,
		labels: labels,
	}, nil
}

// read labels from mnist file
func readLabels(path string) ([]byte, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	header := [2]int32{}
	binary.Read(file, binary.BigEndian, &header)
	labels := make([]byte, header[1])
	_, err = file.Read(labels)
	return labels, err
}

// read images from mnist file
func readImages(path string) ([][]byte, int, int, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, err
	}
	header := [4]int32{}
	binary.Read(file, binary.BigEndian, &header)
	images := make([][]byte, header[1])
	width, height := int(header[2]), int(header[3])
	for i := 0; i < len(images); i++ {
		images[i] = make([]byte, width*height)
		file.Read(images[i])
	}
	return images, width, height, nil
}
