package cmd

import (
	"github.com/apex/log"
	"github.com/spf13/cobra"

	"github.com/ckingdev/fgsdmm"
)

var (
	logger  = fgsdmm.Logger
	verbose = false
)

func init() {
	RootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false, "Use debug logging")
}

var RootCmd = &cobra.Command{
	Use:   "fgsdmm",
	Short: "command line tool for working with fgsdmm",
	PersistentPreRun: func(cmd *cobra.Command, args []string) {
		if verbose {
			logger.Level = log.DebugLevel
			logger.Debug("Using verbose mode.")
		}
	},
}
