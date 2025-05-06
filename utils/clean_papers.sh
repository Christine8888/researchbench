#!/bin/bash

# Define input and output directories
INPUT_DIR="papers/full_texts"
OUTPUT_DIR="papers/cleaned_texts"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Process each file in the input directory
for INPUT_FILE in "$INPUT_DIR"/*; do
    # Skip if not a file
    if [ ! -f "$INPUT_FILE" ]; then
        continue
    fi
    
    # Get just the filename without the path
    FILENAME=$(basename "$INPUT_FILE")
    OUTPUT_FILE="$OUTPUT_DIR/$FILENAME"
    
    echo "Processing $FILENAME..."
    
    # Run the clean_paper.py script
    python utils/clean_paper.py "$INPUT_FILE" "$OUTPUT_FILE"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully cleaned $FILENAME"
    else
        echo "Error processing $FILENAME"
    fi
done

echo "All files processed. Cleaned papers saved to $OUTPUT_DIR"