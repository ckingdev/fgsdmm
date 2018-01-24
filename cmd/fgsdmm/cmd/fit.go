package cmd

import (
	"fmt"
	"time"

	"github.com/spf13/cobra"

	"github.com/ckingdev/fgsdmm"
)

var (
	output     string
	corpusPath string
	corpusType string
)

var fitCmd = &cobra.Command{
	Use:   "fit [data path]",
	Short: "fits the model to the data.",
	Run: func(cmd *cobra.Command, args []string) {
		if corpusPath == "" {
			logger.Fatal("Must supply an input path.")
		}
		logger.Infof("Loading corpus from %s", corpusPath)
		cType := fgsdmm.UnknownCorpus
		switch corpusType {
		case "jsonl":
			cType = fgsdmm.JSONLCorpus
		case "ldac":
			cType = fgsdmm.LDACCorpus
		case "txt":
			cType = fgsdmm.TxtCorpus
		}

		corpus, err := fgsdmm.LoadCorpus(corpusPath, cType)
		if err != nil {
			logger.WithError(err).Fatalf("Error loading corpus from %s", corpusPath)
		}
		logger.Info("Fitting model to corpus.")
		model := fgsdmm.NewFGSDMM(&fgsdmm.HyperParams{
			KMax:     178,
			Alpha:    0.1,
			Beta:     0.1,
			MaxIters: 50,
		})
		model.Fit(corpus)
		logger.Info("Complete.")
		ari := corpus.AdjustedRandIndex(model.Labels)
		logger.Infof("Adjusted Rand Index: %v", ari)
	},
}

func init() {
	fitCmd.Flags().StringVarP(&corpusPath, "input", "i",
		"", "path to input file")
	fitCmd.Flags().StringVarP(&output, "output", "o",
		fmt.Sprintf("model-%v.json", time.Now().Unix()), "path to save the fitted model")
	fitCmd.Flags().StringVarP(&corpusType, "type", "t", "txt", "File format of the corpus. One of jsonl, txt, ldac.")
	RootCmd.AddCommand(fitCmd)
}
