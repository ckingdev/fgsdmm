package fgsdmm

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path"
	"strconv"
	"strings"

	"github.com/apex/log"
	"github.com/apex/log/handlers/text"
)

type CorpusType int

const (
	UnknownCorpus CorpusType = iota
	LDACCorpus
	JSONLCorpus
	TxtCorpus
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
	Label  int
}

// Cluster represents the current token and document counts assigned to the cluster.
type Cluster struct {
	TknCts map[int]int
	NDoc   int
	NTkn   int
}

// Corpus holds the Documents associated with a training set and a map of
// token => numerical ID if needed.
type Corpus struct {
	NDocs  int
	Docs   []*Document
	TknMap map[string]int

	// TODO: set this in loading functions
	V int
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

type RawDoc struct {
	Text    string `json:"text"`
	Cluster int    `json:"cluster"`
}

func LoadJSONL(path string) (*Corpus, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	c := &Corpus{
		NDocs:  0,
		Docs:   make([]*Document, 0),
		TknMap: make(map[string]int),
	}

	for scanner.Scan() {
		rawDoc := &RawDoc{}
		if err := json.Unmarshal(scanner.Bytes(), rawDoc); err != nil {
			return nil, err
		}

		doc := &Document{
			Label:  rawDoc.Cluster,
			TknIDs: make([]int, 0),
			TknCts: make([]int, 0),
		}
		cts := make(map[int]int)
		tokens := strings.Split(rawDoc.Text, " ")
		for _, tkn := range tokens {
			tknID, ok := c.TknMap[tkn]
			if !ok {
				tknID = len(c.TknMap)
				c.TknMap[tkn] = tknID
			}
			cts[tknID]++
		}
		for tknID, ct := range cts {
			doc.TknIDs = append(doc.TknIDs, tknID)
			doc.TknCts = append(doc.TknCts, ct)
			doc.NTkn += ct
		}
		c.Docs = append(c.Docs, doc)
	}
	c.NDocs = len(c.Docs)
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return c, nil
}

func LoadCorpus(fp string, cType CorpusType) (*Corpus, error) {
	if cType == UnknownCorpus {
		Logger.Debugf("No corpus type specified. Attempting to infer from path: %v", fp)
		ext := path.Ext(fp)
		switch ext {
		case "jsonl":
			cType = JSONLCorpus
		case "txt":
			cType = TxtCorpus
		case "dat":
			cType = LDACCorpus
		default:
			return nil, fmt.Errorf("cannot infer corpus type from path: %v", fp)
		}
	}
	switch cType {
	case JSONLCorpus:
		return LoadJSONL(fp)
	case LDACCorpus:
		return LoadLDACFile(fp)
	default:
		return nil, fmt.Errorf("Unknown corpus type.")
	}
}
