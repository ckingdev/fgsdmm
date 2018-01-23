package fgsdmm

import (
	"bufio"
	"encoding/json"
	"os"
	"strings"
)

type LabelledDoc struct {
	*Document
	Cluster int
}

type RawDoc struct {
	Text    string `json:"text"`
	Cluster int    `json:"cluster"`
}

func LoadJSONL(path string) ([]LabelledDoc, map[string]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	docs := make([]LabelledDoc, 0)
	tokenMap := make(map[string]int)
	for scanner.Scan() {
		d := &RawDoc{}
		if err := json.Unmarshal(scanner.Bytes(), d); err != nil {
			return nil, nil, err
		}
		lDoc := LabelledDoc{
			Cluster: d.Cluster,
			Document: &Document{
				TknIDs: make([]int, 0),
				TknCts: make([]int, 0),
			},
		}
		cts := make(map[int]int)
		tokens := strings.Split(d.Text, " ")
		for _, tkn := range tokens {
			tknID, ok := tokenMap[tkn]
			if !ok {
				tknID = len(tokenMap)
				tokenMap[tkn] = tknID
			}
			cts[tknID]++
		}
		for tknID, ct := range cts {
			lDoc.TknIDs = append(lDoc.TknIDs, tknID)
			lDoc.TknCts = append(lDoc.TknCts, ct)
			lDoc.NTkn += ct
		}
		docs = append(docs, lDoc)
	}
	if err := scanner.Err(); err != nil {
		return nil, nil, err
	}
	return docs, tokenMap, nil
}

func nC2(n int) int {
	return n * (n - 1) / 2
}

func AdjustedRandIndex(truth []int, labels []int) float64 {
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

	for i := 0; i < len(truth); i++ {
		xi := truth[i]
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
	Nc2 := nC2(len(truth))

	// prod_comb
	expectedIdx := float64(sumAC2*sumBC2) / float64(Nc2)

	// mean_comb
	maxIdx := float64(sumAC2+sumBC2) / 2

	return (float64(index) - expectedIdx) / (maxIdx - expectedIdx)
}

func EvalTweets(model *FGSDMM) {
	// 89 real clusters in tweets
	tweets, tknMap, err := LoadJSONL("data/Tweet.txt")
	if err != nil {
		panic(err)
	}
	Logger.Infof("Loaded %v tweets with %v unique tokens.", len(tweets), len(tknMap))
	corpus := &Corpus{
		NDocs: len(tweets),
		Docs:  make([]*Document, 0, len(tweets)),
	}

	for _, d := range tweets {
		corpus.Docs = append(corpus.Docs, d.Document)
	}
	model.Fit(corpus)

	Logger.Infof("Document labels: %v", model.Labels)
}
