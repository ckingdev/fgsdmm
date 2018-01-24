package fgsdmm

import (
	"math"

	"gonum.org/v1/gonum/stat/distuv"
)

type State struct {
	KNon     int
	Clusters []*Cluster
	Labels   []int
}

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

	// Clusters is an array of all of the non-empty clusters.
	Clusters []*Cluster

	// Labels holds the current assignment of documents to clusters.
	Labels []int

	// KNon stores the number of non-empty clusters.
	KNon int

	// Corpus holds the documents the model is training on.
	Corpus *Corpus

	// V is the vocabulary size- the number of unique words in the entire corpus.
	V int
}

// NewFGSDMM creates a new model with the given parameters.
func NewFGSDMM(hp *HyperParams) *FGSDMM {
	if hp == nil {
		hp = DefaultHyperParams
	}
	model := &FGSDMM{
		HyperParams: hp,
		Clusters:    make([]*Cluster, 0, hp.KMax),
		Labels:      make([]int, 0),
	}
	return model
}

// clusterAdd updates the cluster to include the counts from the document.
func clusterAdd(c *Cluster, d *Document) {
	for i := 0; i < len(d.TknIDs); i++ {
		c.TknCts[d.TknIDs[i]] += d.TknCts[i]
		c.NTkn += d.TknCts[i]
	}
	c.NDoc++
}

// clusterRemove updates the cluster to remove the counts from the document.
func clusterRemove(c *Cluster, d *Document) {
	for i := 0; i < len(d.TknIDs); i++ {
		c.TknCts[d.TknIDs[i]] -= d.TknCts[i]
		c.NTkn -= d.TknCts[i]
	}
	c.NDoc--
}

// moveCluster moves the last nonempty cluster into cluster z.
func (m *FGSDMM) moveCluster(z int) {
	idx := len(m.Clusters) - 1

	// Move last cluster
	m.Clusters[z] = m.Clusters[idx]

	// Update the labels for docs in the last cluster
	for i := range m.Labels {
		if m.Labels[i] == idx {
			m.Labels[i] = z
		}
	}
	// Remove the last cluster.
	m.Clusters = m.Clusters[:idx]

	m.KNon--
}

// scoreNonEmpty gives the log weight associated with the given cluster and document.
func (m *FGSDMM) scoreNonEmpty(z *Cluster, d *Document) float64 {
	score := math.Log(float64(z.NDoc) + m.Alpha)

	for i := 0; i < len(d.TknIDs); i++ {
		for j := 1; j <= d.TknCts[i]; j++ {
			score += math.Log(float64(z.TknCts[d.TknIDs[i]]+j-1) + m.Beta)
		}
	}

	for i := 1; i <= d.NTkn; i++ {
		score -= math.Log(float64(z.NTkn+i-1) + float64(m.V)*m.Beta)
	}
	return score
}

// scoreEmpty gives the log weight associated with the given document being assigned to one of the empty clusters
func (m *FGSDMM) scoreEmpty(d *Document) float64 {
	score := math.Log(float64(m.KMax-m.KNon) * m.Alpha)
	for i := 0; i < len(d.TknIDs); i++ {
		for j := 1; j <= d.TknCts[i]; j++ {
			score += math.Log(float64(j-1) + m.Beta)
		}
	}
	for i := 1; i <= d.NTkn; i++ {
		score -= math.Log(float64(m.V)*m.Beta + float64(i-1))
	}
	return score
}

// expNormalize takes a set of log weights and calculates the exponent in a way that avoids overflow.
// the result is proportional to the raw probability, i.e., P(i) = expNormalize(logW)[i]/sum(expNormalize(logW))
func expNormalize(logW []float64) []float64 {
	b := logW[0]
	for i := 1; i < len(logW); i++ {
		if logW[i] > b {
			b = logW[i]
		}
	}
	W := make([]float64, len(logW))
	for i := 0; i < len(logW); i++ {
		W[i] = math.Exp(logW[i] - b)
	}
	return W
}

// Fit trains the model on the given corpus.
func (m *FGSDMM) Fit(c *Corpus) {
	m.Corpus = c

	// calculate V as the number of unique terms in the corpus.
	vocab := make(map[int]struct{})
	for _, doc := range c.Docs {
		for _, tknID := range doc.TknIDs {
			vocab[tknID] = struct{}{}
		}
	}
	m.V = len(vocab)

	// Initialize the cluster assignments randomly from a uniform distribution.
	// TODO: sample the initial assignments a la fitting procedure
	weights := make([]float64, m.KMax)
	for i := range weights {
		weights[i] = 1.0
	}

	// TODO: make the random src a parameter for reproducible runs
	sampler := distuv.NewCategorical(weights, nil)
	for _, doc := range c.Docs {
		z := int(sampler.Rand())

		// If assigned to an empty cluster, assign it to the *first* empty cluster.
		if z >= m.KNon {
			z = m.KNon
			m.Clusters = append(m.Clusters, &Cluster{
				TknCts: make(map[int]int),
				NDoc:   0,
			})
			m.KNon++
		}
		m.Labels = append(m.Labels, z)
		clusterAdd(m.Clusters[z], doc)

	}

	swaps := make([]int, 0, m.MaxIters)

	for iter := 0; iter < m.MaxIters; iter++ {
		swaps = append(swaps, 0)

		Logger.Debugf("Beginning iteration %v.", iter+1)

		for d, doc := range m.Corpus.Docs {
			z := m.Labels[d]

			// remove doc d from cluster z and reorder the clusters if empty
			// i.e., if the cluster is empty after removing the document, move the last non-empty cluster into its place.
			clusterRemove(m.Clusters[z], doc)
			if m.Clusters[z].NDoc == 0 {
				m.moveCluster(z)
			}

			// sample a new nonempty cluster or the KNon+1'th empty cluster
			w := make([]float64, m.KNon)

			// TODO: use a pool of workers to calculate this in parallel
			for i := 0; i < m.KNon; i++ {
				w[i] = m.scoreNonEmpty(m.Clusters[i], doc)

			}

			// Only calculate the weight for a new cluster if we're not already using all clusters
			if m.KNon < m.KMax {
				w = append(w, m.scoreEmpty(doc))
			}

			w = expNormalize(w)
			sampler := distuv.NewCategorical(w, nil)
			zNew := int(sampler.Rand())

			// If assigned to an empty cluster, create it first
			if zNew == m.KNon {
				m.Clusters = append(m.Clusters, &Cluster{
					TknCts: make(map[int]int, len(doc.TknIDs)),
					NTkn:   0,
					NDoc:   0,
				})
				m.KNon++
			}

			clusterAdd(m.Clusters[zNew], doc)
			m.Labels[d] = zNew
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
