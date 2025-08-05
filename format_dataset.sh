#!/bin/bash

# Check if any arguments were provided
if [ $# -eq 0 ]; then
    echo "Usage: ./format_dataset.sh [DATASET_SOURCE] [OPTIONS]"
    echo ""
    echo "Examples:"
    echo "  # Format Hugging Face dataset"
    echo "  ./format_dataset.sh squad --prompt-column question --reference-column answers.text"
    echo ""
    echo "  # Format local CSV file"
    echo "  ./format_dataset.sh data.csv --prompt-column question --reference-column answer"
    echo ""
    echo "  # Format local JSONL file with custom tags"
    echo "  ./format_dataset.sh data.jsonl --prompt-column input --reference-column output --intermediate-tag reasoning --final-tag solution"
    echo ""
    echo "  # Using different tags"
    echo "  ./format_dataset.sh data.csv --intermediate-tag steps --final-tag result"
    echo ""
    echo "For more options, run: ./format_dataset.py --help"
    exit 1
fi

# Run the Python script with all arguments passed to this shell script
python3 src/format_dataset.py "$@" 