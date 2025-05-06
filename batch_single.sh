#!/bin/bash

# check if config file was provided
if [ -z "$1" ]; then
    echo "Error: Config file not provided."
    echo "Usage: $0 CONFIG_FILE [PAPER_ID1 PAPER_ID2 ...]"
    exit 1
fi

# Store the config file path
CONFIG_FILE="$1"
shift  # Remove the first argument (config file) from the arguments list

# Check if at least one PAPER_ID was provided
if [ $# -eq 0 ]; then
    echo "Error: No PAPER_IDs provided."
    echo "Usage: $0 CONFIG_FILE [PAPER_ID1 PAPER_ID2 ...]"
    exit 1
fi

# Check if the config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

# Check if single.sh exists and is executable
if [ ! -x "./single.sh" ]; then
    echo "Error: single.sh not found or not executable."
    echo "Make sure single.sh exists in the current directory and has execute permissions."
    exit 1
fi

echo "Using config file: $CONFIG_FILE"
echo "Processing ${#} paper IDs..."

# Loop through all remaining arguments (PAPER_IDs)
for PAPER_ID in "$@"; do
    echo ""
    echo "Processing PAPER_ID: $PAPER_ID"
    echo "----------------"
    
    # Call the single.sh script with the current PAPER_ID and the config file
    ./single.sh "$PAPER_ID" "$CONFIG_FILE"
    
    # Check if the previous command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully processed PAPER_ID: $PAPER_ID"
    else
        echo "Error processing PAPER_ID: $PAPER_ID"
    fi
    
    echo "----------------"
done

echo ""
echo "All paper IDs have been processed."
