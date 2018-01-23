package fgsdmm

import (
	"os"

	"bufio"
	"bytes"
	"fmt"
	"github.com/apex/log"
	"github.com/apex/log/handlers/text"
	"strconv"
)

var Logger = &log.Logger{
	Handler: text.New(os.Stderr),
	Level:   log.InfoLevel,
}

// Document represents a text document as a bag of words.
type Document struct {
	TknIDs []int
	TknCts []int
	NTkn   int
}

// Cluster represents the current token and document counts assigned to the cluster.
type Cluster struct {
	TknCts map[int]int
	NDoc   int
	NTkn   int
}

// Corpus holds the Documents associated with a training set.
type Corpus struct {
	NDocs int
	Docs  []*Document
}

// LoadLDACFile loads a file in the LDA-C format. See the readme at
// https://github.com/blei-lab/lda-c for details.
func LoadLDACFile(fp string) (*Corpus, error) {
	f, err := os.Open(fp)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	c := &Corpus{}

	for scanner.Scan() {
		fields := bytes.Split(scanner.Bytes(), []byte(" "))
		nTkns, err := strconv.Atoi(string(fields[0]))
		if err != nil {
			return nil, err
		}
		if len(fields)-1 != nTkns {
			return nil, fmt.Errorf("incorrect # of tokens in doc %v. Expected %v, got %v", len(c.Docs), nTkns, len(fields)-1)
		}
		d := &Document{
			TknIDs: make([]int, 0, nTkns),
			TknCts: make([]int, 0, nTkns),
		}
		for i := 1; i < len(fields); i++ {
			idx := bytes.Index(fields[i], []byte(":"))
			if idx == -1 {
				return nil, fmt.Errorf("bad format in doc %v. Token IDs and counts must be separated by a colon", len(c.Docs))
			}
			tknID, err := strconv.Atoi(string(fields[i][:idx]))
			if err != nil {
				return nil, err
			}
			tknCt, err := strconv.Atoi(string(fields[i][idx+1:]))
			if err != nil {
				return nil, err
			}
			d.TknIDs = append(d.TknIDs, tknID)
			d.TknCts = append(d.TknCts, tknCt)
		}
		c.Docs = append(c.Docs, d)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	c.NDocs = len(c.Docs)

	return c, nil
}
