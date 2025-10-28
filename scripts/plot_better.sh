#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# Takes in 1 arguement for csv to pull from
# Runs entire pipeline
plot_from_scratch() {
	if [ ! -d ""$INTR"" ]; then 
		mkdir -p "$INTR"
	fi

	"$VENV_PYTHON" "$PYS/sbert_filter_embed.py" \
	"$1" \
	"$CONTEXT_ROOT/cache.json" \
	"$INTR/filtered_embeddings.json" \
	"$CONTEXT_ROOT/capabilities.txt" 

	"$VENV_PYTHON" "pys/umap_reduce.py" \
	"$INTR/filtered_embeddings.json" \
	768 \
	"$INTR/50d_embeddings.json" \
	50 \
	"$INTR/model/umap.pkl" 
	
	"$VENV_PYTHON" "$PYS/cluster.py" \
	"$INTR/50d_embeddings.json" \
	50 \
	"$INTR/cluster_embeddings.json" 
	
	plot_json "$INTR/50d_embeddings.json" "$1"
	#"$VENV_PYTHON" "$PYS/plotting.py" \
	#"$INTR/50d_embeddings.json" \
	#"$INTR/cluster_embeddings.json" \
	#"$1" 
	#"$CONTEXT_ROOT/plot.html" 
}

# Takes in json to plot and the csv it was derived from
plot_json() {
	"$VENV_PYTHON" "$PYS/plotting.py" \
	"$1" \
	"$INTR/cluster_embeddings.json" \
	"$2" \
	"$CONTEXT_ROOT/plot.html" 
}

