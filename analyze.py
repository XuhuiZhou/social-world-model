import json
import os
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, DefaultDict


def get_question_type(question: str) -> str:
    question = question.lower().strip()
    if question.startswith("where will"):
        return "first order"
    elif question.startswith("where does"):
        return "second order"
    elif question.startswith("where was"):
        return "reality"
    elif question.startswith("where is"):
        return "reality"
    return "unknown"


def analyze_simulation_files(directory: str) -> None:
    # Get the directory path
    dir_path = Path(directory).resolve()  # Get absolute path

    # Statistics counters
    stats: DefaultDict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "incorrect": 0}
    )
    mismatched_files: List[Dict[str, str]] = []

    # Iterate through all JSON files in the directory
    for file_path in dir_path.glob("*.json"):
        try:
            with open(file_path, "r") as f:
                data: Dict[str, Any] = json.load(f)

            if "question" in data:
                question_type = get_question_type(data["question"])
                stats[question_type]["total"] += 1

                # Check if correct_answer exists and is different from answer
                if "correct_answer" in data and "answer" in data:
                    if data["correct_answer"] not in data["answer"]:
                        stats[question_type]["incorrect"] += 1
                        mismatched_files.append(
                            {
                                "filepath": str(file_path),
                                "filename": file_path.name,
                                "question": data["question"],
                                "question_type": question_type,
                                "correct_answer": data["correct_answer"],
                                "given_answer": data["answer"],
                            }
                        )

        except json.JSONDecodeError:
            print(f"Error reading JSON file: {file_path}")
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")

    # Create CSV file
    csv_path = "mismatched_answers.csv"
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "filename",
            "question_type",
            "question",
            "correct_answer",
            "given_answer",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for file in mismatched_files:
            writer.writerow(
                {
                    "filename": file["filename"],
                    "question_type": file["question_type"],
                    "question": file["question"],
                    "correct_answer": file["correct_answer"],
                    "given_answer": file["given_answer"],
                }
            )

    # Print statistics
    print("\nQuestion Type Statistics:")
    print("-----------------------")
    for qtype, counts in stats.items():
        total = counts["total"]
        incorrect = counts["incorrect"]
        accuracy = ((total - incorrect) / total * 100) if total > 0 else 0
        print(f"\nType: {qtype}")
        print(f"Total questions: {total}")
        print(f"Incorrect answers: {incorrect}")
        print(f"Accuracy: {accuracy:.2f}%")

    # Print detailed results for incorrect answers
    if mismatched_files:
        print("\n\nDetailed Incorrect Answers:")
        print("---------------------------")
        memory_path = "/Users/xuhuizhou/Projects/social-world-model/data/tomi_results/simulation_o1-2024-12-17_rephrased_tomi_test_600.csv"
        for file in mismatched_files:
            print(f"\nFile: {file['filepath']}")
            print(f"Memory: {os.path.join(memory_path, file['filename'])}")
            print(f"Question Type: {file['question_type']}")
            print(f"Question: {file['question']}")
            print(f"Correct answer: {file['correct_answer']}")
            print(f"Given answer: {file['given_answer']}")

        # Count mismatches by question type
        type_counts: DefaultDict[str, int] = defaultdict(int)
        for file in mismatched_files:
            type_counts[file["question_type"]] += 1

        print(f"\nTotal files with mismatched answers: {len(mismatched_files)}")
        print("\nBreakdown by question type:")
        for qtype, count in type_counts.items():
            print(f"- {qtype}: {count} files")

        print(f"\nCSV file created: {csv_path}")
    else:
        print("\nNo files found with mismatched answers.")


if __name__ == "__main__":
    simulation_dir = "data/simulations_tomi/o1-2024-12-17"
    analyze_simulation_files(simulation_dir)
