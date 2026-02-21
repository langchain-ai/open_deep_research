#!/usr/bin/env python3
"""
Script to create a DeepConsult responses CSV file combining:
- Questions from the original queries.csv
- Baseline answers from the existing responses CSV
- Candidate answers from the generated edr_reports_gemini JSON files
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from datetime import datetime


def read_queries_csv(queries_file):
    """Read the queries from the CSV file."""
    queries = []
    with open(queries_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            queries.append({"id": i, "query": row["query"].strip()})
    return queries


def read_baseline_csv(baseline_file):
    """Read the baseline answers from the existing responses CSV."""
    baseline_answers = {}
    with open(baseline_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            baseline_answers[i] = row["baseline_answer"]
    return baseline_answers


def read_generated_reports(reports_dir):
    """Read all generated deepconsult JSON reports."""
    reports = {}

    # Get all deepconsult_*.json files
    for filename in os.listdir(reports_dir):
        if filename.startswith("deepconsult_") and filename.endswith(".json"):
            # Extract the ID from filename (e.g., deepconsult_0.json -> 0)
            try:
                task_id = int(filename.replace("deepconsult_", "").replace(".json", ""))

                with open(
                    os.path.join(reports_dir, filename), "r", encoding="utf-8"
                ) as f:
                    report_data = json.load(f)

                # Extract the article content as candidate answer
                candidate_answer = report_data.get("article", "")
                reports[task_id] = candidate_answer

            except (ValueError, json.JSONDecodeError) as e:
                print(f"Error processing {filename}: {e}")
                continue

    return reports


def create_responses_csv(queries, baseline_answers, generated_reports, output_file):
    """Create the new responses CSV file."""
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(["question", "baseline_answer", "candidate_answer"])

        # Write data rows
        for query_data in queries:
            task_id = query_data["id"]
            question = query_data["query"]

            # Get baseline answer (if available)
            baseline_answer = baseline_answers.get(task_id, "")

            # Get generated candidate answer (if available)
            candidate_answer = generated_reports.get(task_id, "")

            writer.writerow([question, baseline_answer, candidate_answer])

            # Print progress
            if task_id % 10 == 0:
                print(f"Processed query {task_id}: {question[:50]}...")


def get_default_paths():
    """Get default file paths based on the script location."""
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent.parent

    return {
        "queries_file": base_dir
        / "benchmarks"
        / "ydc-deep-research-evals"
        / "datasets"
        / "DeepConsult"
        / "queries.csv",
        "baseline_file": base_dir
        / "benchmarks"
        / "ydc-deep-research-evals"
        / "datasets"
        / "DeepConsult"
        / "responses_OpenAI-DeepResearch_vs_ARI_2025-05-15.csv",
        "reports_dir": base_dir
        / "benchmarks"
        / "ydc-deep-research-evals"
        / "results"
        / "edr_reports_gemini",
        "output_dir": base_dir
        / "benchmarks"
        / "ydc-deep-research-evals"
        / "datasets"
        / "DeepConsult",
    }


def main():
    """Main function to handle command line arguments and process DeepConsult responses."""

    parser = argparse.ArgumentParser(
        description="Create DeepConsult responses CSV file for evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths
  python process_deepconsult.py
  
  # Custom paths
  python process_deepconsult.py \\
    --queries-file /path/to/queries.csv \\
    --baseline-file /path/to/baseline_responses.csv \\
    --reports-dir /path/to/generated_reports \\
    --output-file /path/to/output_responses.csv
  
  # Custom output filename with timestamp
  python process_deepconsult.py --output-name "responses_EDR_gemini_10loops"
        """,
    )

    # Get default paths
    defaults = get_default_paths()

    # Input file arguments
    parser.add_argument(
        "--queries-file",
        type=str,
        default=str(defaults["queries_file"]),
        help="Path to queries CSV file (default: auto-detected)",
    )

    parser.add_argument(
        "--baseline-file",
        type=str,
        default=str(defaults["baseline_file"]),
        help="Path to baseline responses CSV file (default: auto-detected)",
    )

    parser.add_argument(
        "--reports-dir",
        type=str,
        default=str(defaults["reports_dir"]),
        help="Directory containing generated deepconsult_*.json files (default: auto-detected)",
    )

    # Output file arguments
    parser.add_argument(
        "--output-file",
        type=str,
        help="Full path to output CSV file (overrides --output-dir and --output-name)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(defaults["output_dir"]),
        help="Output directory for the responses CSV file (default: auto-detected)",
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="responses_EDR_vs_ARI",
        help="Base name for output file (timestamp will be added) (default: responses_EDR_vs_ARI)",
    )

    # Processing options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with detailed progress",
    )

    args = parser.parse_args()

    # Validate input files exist
    if not os.path.exists(args.queries_file):
        print(f"[ERROR] Queries file not found: {args.queries_file}")
        return 1

    if not os.path.exists(args.baseline_file):
        print(f"[ERROR] Baseline file not found: {args.baseline_file}")
        return 1

    if not os.path.exists(args.reports_dir):
        print(f"[ERROR] Reports directory not found: {args.reports_dir}")
        return 1

    # Determine output file path
    if args.output_file:
        output_file = args.output_file
    else:
        # Create output filename with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d")
        output_filename = f"{args.output_name}_{timestamp}.csv"
        output_file = os.path.join(args.output_dir, output_filename)

        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)

    # Print configuration
    print("Configuration:")
    print(f"   Queries file: {args.queries_file}")
    print(f"   Baseline file: {args.baseline_file}")
    print(f"   Reports directory: {args.reports_dir}")
    print(f"   Output file: {output_file}")
    print()

    try:
        # Process files
        print("Reading queries from CSV...")
        queries = read_queries_csv(args.queries_file)
        print(f"Loaded {len(queries)} queries")

        print("Reading baseline answers...")
        baseline_answers = read_baseline_csv(args.baseline_file)
        print(f"Loaded {len(baseline_answers)} baseline answers")

        print("Reading generated reports...")
        generated_reports = read_generated_reports(args.reports_dir)
        print(f"Loaded {len(generated_reports)} generated reports")

        print("Creating new responses CSV...")
        create_responses_csv(queries, baseline_answers, generated_reports, output_file)

        print(f"[OK] Successfully created: {output_file}")

        # Print summary statistics
        print("\nSummary:")
        print(f"   Total queries: {len(queries)}")
        print(f"   Baseline answers: {len(baseline_answers)}")
        print(f"   Generated reports: {len(generated_reports)}")
        print(f"   Output file: {output_file}")

        # Check for any missing reports
        missing_reports = []
        for query_data in queries:
            if query_data["id"] not in generated_reports:
                missing_reports.append(query_data["id"])

        if missing_reports:
            print(
                f"[WARN] Missing reports for {len(missing_reports)} queries: {missing_reports[:10]}{'...' if len(missing_reports) > 10 else ''}"
            )
            if args.verbose:
                print(f"   Missing IDs: {missing_reports}")
        else:
            print("[OK] All queries have corresponding generated reports!")

        return 0

    except (FileNotFoundError, json.JSONDecodeError, csv.Error, IOError) as e:
        print(f"[ERROR] {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1
    except Exception as e:
        print(f"[ERROR] UNEXPECTED ERROR: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
