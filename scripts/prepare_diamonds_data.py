import json
import os
from typing import Any, Tuple

import pandas as pd


def load_jsonl(file_path: str) -> list[dict[str, Any]]:
    """Load a jsonl file into a list of dictionaries."""
    data = []
    with open(file_path, "r") as f:
        content = f.read()
        # Check if the content is a JSON array
        if content.strip().startswith("["):
            data = json.loads(content)
        else:
            # Process as JSONL (one JSON object per line)
            for line in content.splitlines():
                if line.strip():
                    data.append(json.loads(line))
    return data


def prepare_diamonds_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare DIAMONDs dataset for benchmarking."""
    # Create the data directory if it doesn't exist
    os.makedirs("data/diamonds", exist_ok=True)

    # Load the DIAMONDs datasets
    base_data = load_jsonl("DIAMONDs/data/base-conv-QA.jsonl")
    distr_data = load_jsonl("DIAMONDs/data/distr-conv-QA.jsonl")
    underspec_data = load_jsonl("DIAMONDs/data/underspec-conv-QA.jsonl")

    # Process and combine the datasets
    all_data = []

    # Process base data
    for i, item in enumerate(base_data):
        item_dict = dict(item)
        item_dict["index"] = f"base_{i}"
        item_dict["set_id"] = item_dict["id"]
        all_data.append(item_dict)

    # Process distr data
    for i, item in enumerate(distr_data):
        item_dict = dict(item)
        item_dict["index"] = f"distr_{i}"
        item_dict["set_id"] = item_dict["id"]
        all_data.append(item_dict)

    # Process underspec data
    for i, item in enumerate(underspec_data):
        item_dict = dict(item)
        item_dict["index"] = f"underspec_{i}"
        item_dict["set_id"] = item_dict["id"]
        all_data.append(item_dict)

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Save to CSV
    df.to_csv("data/diamonds/diamonds_data.csv", index=False)
    print(f"Saved {len(df)} examples to data/diamonds/diamonds_data.csv")

    # Also save a small sample for testing
    sample_df = df.sample(min(10, len(df)))
    sample_df.to_csv("data/diamonds/diamonds_sample.csv", index=False)
    print(f"Saved {len(sample_df)} examples to data/diamonds/diamonds_sample.csv")

    return df, sample_df


if __name__ == "__main__":
    prepare_diamonds_data()
