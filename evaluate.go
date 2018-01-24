package fgsdmm

func nC2(n int) int {
	return n * (n - 1) / 2
}

func (c *Corpus) AdjustedRandIndex(labels []int) float64 {
	// C is the ground truth class assignment. K is the clustering.
	// a is the number of pairs of elements that are in the same set in C and
	// in the same set in K.
	// b is the number of pairs of elements that are in different sets in C and
	// in different sets in K.
	// Raw Rand index = (a+b)/(n_docs(n_docs-1)/2)

	//contingency matrix
	sumsX := make(map[int]int)
	sumsY := make(map[int]int)
	cont := make(map[[2]int]int)

	for i := 0; i < len(labels); i++ {
		xi := c.Docs[i].Label
		yi := labels[i]
		sumsX[xi]++
		sumsY[yi]++
		cont[[2]int{xi, yi}]++
	}
	index := 0
	for _, nij := range cont {
		index += nC2(nij)
	}
	sumAC2 := 0
	sumBC2 := 0
	for _, ai := range sumsX {
		sumAC2 += nC2(ai)
	}
	for _, bi := range sumsY {
		sumBC2 += nC2(bi)
	}
	Nc2 := nC2(len(labels))

	// prod_comb
	expectedIdx := float64(sumAC2*sumBC2) / float64(Nc2)

	// mean_comb
	maxIdx := float64(sumAC2+sumBC2) / 2

	return (float64(index) - expectedIdx) / (maxIdx - expectedIdx)
}
