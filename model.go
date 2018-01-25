package fgsdmm

import (
	"gonum.org/v1/gonum/stat/distuv"
)

// HyperParams stores the hyperparameters for FGSDMM.
type HyperParams struct {
	// KMax is the maximum number of clusters to be used.
	KMax int

	// Alpha controls the pressure to assign documents to bigger clusters.
	// If Alpha is 0, empty tables will never be joined. (0 < Alpha < 1)
	Alpha float64

	// Beta controls the pressure to assign documents to clusters with similar interests.
	// Low beta => documents are more likely to go to clusters with similar interests.
	// High beta => documents are more likely to go to larger clusters.
	Beta float64

	// MaxIters is a hard upper bound on the number of iterations of Gibbs sampling.
	MaxIters int
}

var DefaultHyperParams = &HyperParams{
	KMax:     100,
	Alpha:    0.1,
	Beta:     0.1,
	MaxIters: 50,
}

type FGSDMM struct {
	*HyperParams
	*State

	// Corpus holds the documents the model is training on.
	Corpus Corpus
}

// NewFGSDMM creates a new model with the given parameters.
func NewFGSDMM(hp *HyperParams) *FGSDMM {
	if hp == nil {
		hp = DefaultHyperParams
	}
	model := &FGSDMM{
		HyperParams: hp,
		State: &State{
			Clusters: make([]*Cluster, 0, hp.KMax),
			Labels:   make([]int, 0),
		},
	}
	return model
}

// Fit trains the model on the given corpus.
func (m *FGSDMM) Fit(c Corpus) {
	m.Corpus = c

	// calculate V as the number of unique terms in the corpus.
	vocab := make(map[int]struct{})
	for _, doc := range c.Docs {
		for _, tknID := range doc.TknIDs {
			vocab[tknID] = struct{}{}
		}
	}
	m.Corpus.V = len(vocab)

	// Initialize the cluster assignments randomly from a uniform distribution.
	// TODO: sample the initial assignments a la fitting procedure
	weights := make([]float64, m.KMax)
	for i := range weights {
		weights[i] = 1.0
	}

	m.State.Labels = make([]int, c.NDocs)
	// TODO: make the random src a parameter for reproducible runs
	sampler := distuv.NewCategorical(weights, nil)
	for i, doc := range c.Docs {
		z := int(sampler.Rand())
		m.clusterAdd(z, i, &doc)
	}
	swaps := make([]int, 0, m.MaxIters)
	for iter := 0; iter < m.MaxIters; iter++ {
		swaps = append(swaps, 0)

		Logger.Debugf("Beginning iteration %v.", iter+1)

		for d, doc := range m.Corpus.Docs {
			z := m.Labels[d]

			m.clusterRemove(z, &doc)

			// sample a new nonempty cluster or the KNon+1'th empty cluster
			w := make([]float64, m.KNon)

			// TODO: use a pool of workers to calculate this in parallel
			for i := 0; i < m.KNon; i++ {
				w[i] = m.scoreNonEmpty(m.Clusters[i], &doc)
			}

			// Only calculate the weight for a new cluster if we're not already using all clusters
			if m.KNon < m.KMax {
				w = append(w, m.scoreEmpty(&doc))
			}

			sampler := distuv.NewCategorical(w, nil)
			zNew := int(sampler.Rand())

			m.clusterAdd(zNew, d, &doc)
			if z != zNew {
				swaps[iter]++
			}
		}
		if swaps[iter] == 0 && swaps[iter-1] == 0 {
			Logger.Infof("Last two iterations had no cluster assignments change. Stopping Gibbs sampling.")
			break
		}
		if swaps[iter] == 0 {
			Logger.Debugf("Iteration %v of %v: %v documents changed clusters.", iter+1, m.MaxIters, swaps[iter])
		}
		Logger.Debugf("%v non-empty clusters.", m.KNon)
	}
}
