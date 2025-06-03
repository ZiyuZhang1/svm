#!/bin/bash

# Filter the lines based on the 7th column being one of the specified evidence codes.
grep -v '^!' goa_human.gaf | awk -F'\t' '($7 == "EXP" || $7 == "IDA" || $7 == "IPI" || $7 == "IMP" || $7 == "IGI" || $7 == "IEP" || $7 == "HTP" || $7 == "HDA" || $7 == "HGI" || $7 == "HEP")' > exp_go.txt

echo "Filtered lines saved to exo_go.txt"
