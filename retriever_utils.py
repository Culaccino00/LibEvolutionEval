import os
import json
from typing import Dict, List, Tuple

import numpy as np
import re
import logging
np.random.seed(0)



def augment_prompt(task: Dict, retrieved: List[Tuple[Dict, float]]) -> Dict:
    """
    Helper to augment prompt with retrieved docs (Torch-style schema).
    Works for both torch and matplotlib once the latter is converted.
    """
    augmentation = "# Use the following API information as reference:\n"

    for ret_pair in retrieved:
        ret = ret_pair[0]  # ignore score

        # Torch-style schema access (works for converted Matplotlib too)
        api_name = ret.get("API_Call", "Unknown API")
        signature = (ret.get("Signature", "") or "").replace("[source]\u00b6", "").replace("[source]", "").replace("\u00b6", "")
        detailed_desc = ret.get("Detailed_Description", "")

        block = f"{api_name} : {signature}\n"

        # Parameters (dict of {name: details/desc})
        params = ret.get("Parameters", {})
        if isinstance(params, dict) and params:
            # Keep concise, name-only list for readability
            param_names = ", ".join(params.keys())
            block += f"Input Parameters : {param_names}\n"

        # Optional short description (first non-empty line)
        if isinstance(detailed_desc, str) and detailed_desc.strip():
            first_line = detailed_desc.strip().split("\n", 1)[0].strip()
            if first_line:
                block += f"Description : {first_line}\n"

        block += "\n" + ("=" * 20) + "\n"

        # Comment for readability (keeps augmentation from polluting runtime code)
        commented = "\n".join("# " + line for line in block.splitlines())
        augmentation += commented + "\n"

    # Prepend augmentation to prompt area
    task["prompt"] = f"\n{augmentation}\n\n{task['prompt']}"
    return task


def read_documents_from_disk(source: str, version: str, debug: bool, base_path: str) -> List[Dict]:
    """Read documents from disk."""
    apis = []
    # Construct the path to include source and version as separate directories
    if "torch" in source:
        source_ = "torch"
    elif "matplotlib" in source:
        source_ = "matplotlib"
    path = os.path.join(base_path, source_, f"{version}.jsonl")
    print("Loading APIs from:", path)
    logging.info(f"Loading APIs from: {path}")
    # Load the data from the specified path
    if debug:
        this_apis = [json.loads(l) for l in open(path, "r")][:100]
    else:
        this_apis = [json.loads(l) for l in open(path, "r")]

    # Annotate each entry with its package name based on the source
    for d in this_apis:
        if "torch" in source:
            d["package"] = "torch"
        if "matplotlib" in source:
            d["package"] = "matplotlib"
        else:
            d["package"] = source
    apis.extend(this_apis)

    return apis


def setup_retrieval(use_retrieval: bool, retrieval_config: Dict, documents: List[Dict]) -> Tuple:
    """Helper function to set up the retrieval settings for RAG."""
    if use_retrieval:
        # instantiate and set up the retriever
        retrieval_type = retrieval_config["retrieval_type"]

        if retrieval_type == "no_augmentation":
            # No augmentation mode - no actual retrieval
            retriever = None
            decision_function = lambda _: False
        else:
            raise NotImplementedError(f"Retrieval type '{retrieval_type}' is not supported")
        
    else:
        # set up the conditions to avoid retrieval
        retriever = None
        decision_function = lambda _: False

    return retriever, decision_function
