package mlp

// vector representation of a list of values
type vector []float64

// allocate a new vector of size m
func newVector(m int) vector {
	return make(vector, m)
}

// iter over each item using i and j coordinates
func (vec vector) iter(f func(int)) vector {
	for i := 0; i < len(vec); i++ {
		f(i)
	}
	return vec
}

// zeros fills the vector with zeros
func (vec vector) zeros() vector {
	return vec.iter(func(i int) { vec[i] = 0 })
}

// matrix representation of a 2D list of values
type matrix [][]float64

// allocate a new matrix of size m x n
func newMatrix(m, n int) matrix {
	mat := make(matrix, m)
	for i := 0; i < m; i++ {
		mat[i] = make([]float64, n)
	}
	return mat
}

// iter over each item using i and j coordinates
func (mat matrix) iter(f func(int, int)) matrix {
	for i := 0; i < len(mat); i++ {
		for j := 0; j < len(mat[i]); j++ {
			f(i, j)
		}
	}
	return mat
}

// zeros fills the matrix with zeros
func (mat matrix) zeros() matrix {
	return mat.iter(func(i, j int) { mat[i][j] = 0 })
}
