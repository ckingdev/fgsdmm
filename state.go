package fgsdmm

import (
	"math"
)

type State struct {
	// Clusters is an array of all of the non-empty clusters.
	Clusters []*Cluster

	// Labels holds the current assignment of documents to clusters.
	Labels []int

	// KNon stores the number of non-empty clusters.
	KNon int
}

func (s *State) clusterAdd(z int, docID int, d *Document) {
	if z >= len(s.Clusters) {
		z = len(s.Clusters)
		s.Clusters = append(s.Clusters, &Cluster{
			TknCts: make(map[int]int),
			NDoc:   0,
		})

		s.KNon++
	}
	c := s.Clusters[z]
	for i := 0; i < len(d.TknIDs); i++ {
		c.TknCts[d.TknIDs[i]] += d.TknCts[i]
		c.NTkn += d.TknCts[i]
	}
	c.NDoc++
	s.Labels[docID] = z
}

// clusterRemove updates the cluster to remove the counts from the document.
func clusterRemove(c *Cluster, d *Document) {
	for i := 0; i < len(d.TknIDs); i++ {
		c.TknCts[d.TknIDs[i]] -= d.TknCts[i]
		c.NTkn -= d.TknCts[i]
	}
	c.NDoc--
}

func (s *State) clusterRemove(z int, d *Document) {
	c := s.Clusters[z]
	for i := 0; i < len(d.TknIDs); i++ {
		c.TknCts[d.TknIDs[i]] -= d.TknCts[i]
		c.NTkn -= d.TknCts[i]
	}
	c.NDoc--

	if c.NDoc == 0 {
		// move the cluster
		s.moveCluster(z)
	}
}

// moveCluster moves the last nonempty cluster into cluster z.
func (s *State) moveCluster(z int) {
	idx := len(s.Clusters) - 1

	// Move last cluster
	s.Clusters[z] = s.Clusters[idx]

	// Update the labels for docs in the last cluster
	for i := range s.Labels {
		if s.Labels[i] == idx {
			s.Labels[i] = z
		}
	}
	// Remove the last cluster.
	s.Clusters = s.Clusters[:idx]

	s.KNon--
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
		score -= math.Log(float64(z.NTkn+i-1) + float64(m.Corpus.V)*m.Beta)
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
		score -= math.Log(float64(m.Corpus.V)*m.Beta + float64(i-1))
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
