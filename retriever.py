from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import re

np.random.seed(0)


def _clean_signature(sig: str) -> str:
    if not isinstance(sig, str):
        return "No signature available."
    # Strip Matplotlib doc artifacts and normalize arrow
    return (sig
            .replace("[source]\u00b6", "")
            .replace("[source]", "")
            .replace("\u00b6", "")
            .replace("\u2192", "->")
            .replace(")#:", "):")
            .strip())

def prepare_key(api_dict: Dict, normalize: bool = False) -> str:
    """Prepare a unified key string for indexing (Torch-style for both libs)."""

    # --- Map to a unified schema (works even if matplotlib wasn't pre-converted) ---
    api_call = api_dict.get("API_Call") or api_dict.get("API_Name") or "Unknown API Call"
    signature = _clean_signature(api_dict.get("Signature", "No signature available."))
    description = api_dict.get("Detailed_Description", "No description available.")
    params = api_dict.get("Parameters") or {}

    # Ensure types we expect
    if not isinstance(description, str):
        description = str(description)
    if not isinstance(params, dict):
        params = {}

    # --- Build a consistent key body (same for torch & matplotlib) ---
    # 1) Signature line
    # 2) Inputs (names only, concise)
    # 3) Commented, line-by-line description (keeps it readable in-ctx)
    commented_description = "\n".join(f"# {line}" for line in description.split("\n"))
    inputs_section = "Inputs: " + ", ".join(params.keys()) if params else "Inputs:"

    # Keep format very close to your Torch branch, but include api_call context
    key = f"{api_call} : {signature}\n{inputs_section}\n{commented_description}"

    # Optional normalization step (if you have a normalize_text helper)
    if normalize:
        try:
            return normalize_text(key)  # assumes your normalize_text exists in scope
        except NameError:
            # Fall back gracefully if normalize_text isn't available
            return key

    return key


def process_api_full(api_full):
    """
    Process the 'api_full' string to keep only the method and attribute access,
    removing any method arguments.
    """
    # Remove parentheses and their contents, including nested ones
    # This pattern matches parentheses and their contents
    pattern = r'\([^()]*\)'
    while re.search(pattern, api_full):  # Keep removing until all parentheses and their contents are removed
        api_full = re.sub(pattern, '', api_full)
    return api_full


def prepare_query(
        generation_dict: Dict, 
        normalize: bool = False, 
        tree_style_normalize: bool = False, 
        use_hypothesis: bool = False,
        use_prompt: bool = False,
        use_comment_str: bool = False,
) -> str:
    # last 5 lines from the prompt may be used in the query
    prompt = "\n".join(generation_dict["prompt"].split("\n")[-5:])
    # the model prediction may be used in the query
    hypothesis = generation_dict.get("hypothesis", "")

    comment_str = generation_dict["comment_str"]
    code_str = generation_dict["code_str"]
    if normalize:
        func = normalize_text
    elif tree_style_normalize:
        func = process_api_full
    else:
        func = lambda x: x  # Identity function
    query = ""
    if use_prompt:
        query += prompt
    if use_hypothesis:
        query += hypothesis
    if use_comment_str:
        query += comment_str

    return func(query)


#     # Replace these characters with an empty string
def normalize_text(text: str) -> str:
    text = text.lower()
    import re
    # Pattern to match characters that are not letters, numbers, or spaces
    pattern_non_alphanumeric = r"[^a-zA-Z0-9\s]"
    # Replace these characters with a space
    cleaned_text = re.sub(pattern_non_alphanumeric, " ", text)
    # Pattern to match any numbers
    pattern_numbers = r"\d+"
    # Remove numbers from the cleaned text
    cleaned_text_without_numbers = re.sub(pattern_numbers, "", cleaned_text)
    return cleaned_text_without_numbers

class BaseRetriever(ABC):
    """Base class to define all types of retrievers"""
    def __init__(self):
        self.index = None

    @abstractmethod
    def prepare_index(self, documents: List[Dict]):
        pass
    
    @abstractmethod
    def retrieve(self, query: str, num_results: int = 5) -> List[Tuple[Dict, float]]:
        pass