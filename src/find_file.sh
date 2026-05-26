#!/bin/bash

# Usage:
# ./find_keywords.sh [directory]

DIR=${1:-.}

grep -rliZ "plot" "$DIR" | \
xargs -0 grep -liZ "category" | \
xargs -0 grep -liZ "fuse" | \
xargs -0 grep -li "ppi"