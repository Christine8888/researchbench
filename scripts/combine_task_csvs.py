#!/usr/bin/env python3
import pandas as pd
import sys
import argparse


def combine_csvs(input_file, category_file, output_file=None):
    """
    Combine a task list CSV with a categorization CSV and output the result.

    Args:
        input_file: Path to the paper_list_filtered.csv
        category_file: Path to the paper_categorized.csv containing categories
        output_file: Path to save the output (if None, prints to stdout)
    """
    # Load the two CSV files
    try:
        paper_list = pd.read_csv(input_file)
        categories = pd.read_csv(category_file)

        # Ensure the category is an integer
        categories["category"] = categories["category"].astype(
            "Int64"
        )  # Pandas integer type
    except Exception as e:
        print(f"Error loading input files: {e}", file=sys.stderr)
        sys.exit(1)

    # Merge the dataframes on both paper_id and task_name
    combined = pd.merge(
        paper_list, categories, on=["paper_id", "task_name"], how="left"
    )

    # Check if any tasks didn't get a category
    missing_categories = combined[combined["category"].isna()]
    if not missing_categories.empty:
        print(
            f"Warning: {len(missing_categories)} tasks don't have categories assigned:",
            file=sys.stderr,
        )
        print(
            missing_categories[["paper_id", "task_name"]].to_string(), file=sys.stderr
        )

    # Ensure category is an integer column
    if "category" in combined.columns:
        combined["category"] = combined["category"].astype(
            "Int64"
        )  # Pandas integer type

    # Output the combined dataframe
    if output_file:
        # Write to file without float formatting for integers
        combined.to_csv(output_file, index=False, float_format="%.0f")
        print(
            f"Combined CSV created with {len(combined)} rows at {output_file}",
            file=sys.stderr,
        )
    else:
        # Print to stdout without float formatting for integers
        combined.to_csv(sys.stdout, index=False, float_format="%.0f")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Combine task list CSV with categorization CSV"
    )
    parser.add_argument("input_file", help="Path to the paper_list_filtered.csv")
    parser.add_argument(
        "category_file", help="Path to the paper_categorized.csv containing categories"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to save the output (if not specified, prints to stdout)",
    )

    args = parser.parse_args()

    # Call the function with provided arguments
    combine_csvs(args.input_file, args.category_file, args.output)


if __name__ == "__main__":
    """
    python combine_task_csvs.py paper_tasks_filtered.csv paper_tasks_categorized.csv > final_task_list.csv
    """
    main()
