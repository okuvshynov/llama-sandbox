// Reference oracle for the hcl-2 spec.
//
// Reads the file given as argv[1] and exits 0 if its contents are a
// syntactically valid HCL2 document per HashiCorp's reference parser
// (github.com/hashicorp/hcl/v2/hclparse), non-zero otherwise.
//
// Used by verify_corpus.py to re-derive labels at corpus materialization
// time, so a checked-in test in valid/ that fails this check (or vice
// versa) flags a label drift.
//
// "Syntactically valid" here means hclparse.ParseHCL returns without
// any diagnostic errors. This matches what `hclparse.NewParser` does
// when you feed it raw `.hcl` content — it accepts whatever the HCL2
// grammar accepts, with no schema applied on top. Terraform and other
// downstream tools layer additional validation (resource schemas,
// variable types, etc.), and those rejections are NOT failures here:
// a file that's syntactically valid HCL but semantically nonsensical
// to Terraform is still "valid HCL" for our purposes.
package main

import (
	"fmt"
	"os"

	"github.com/hashicorp/hcl/v2/hclparse"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintf(os.Stderr, "usage: %s <file>\n", os.Args[0])
		os.Exit(2)
	}
	data, err := os.ReadFile(os.Args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "read: %v\n", err)
		os.Exit(2)
	}
	parser := hclparse.NewParser()
	_, diags := parser.ParseHCL(data, os.Args[1])
	if diags.HasErrors() {
		os.Exit(1)
	}
	os.Exit(0)
}
