#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# Takes in the sentances to embed and the embeddings.json
searchsem() {
	"$VENV_PYTHON" "$PYS/semantic_search.py" \
	"$1" \
	$2 \
	All_Contract_Opportunities_1998_2030.csv \
	0.88 \
	top.csv
}
