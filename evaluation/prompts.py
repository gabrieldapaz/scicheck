"""Prompt construction.

Prompt templates are plain-text files in `prompts/` (see `prompts/README.md`).
`--prompt_variant` is the template file name without the `.txt` extension.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict

from evaluation.config import PROMPTS_DIR

# Cache for prompt templates to avoid re-reading files
_prompt_cache: Dict[str, str] = {}


def get_prompt(prompt_name: str) -> str:
    """Return the prompt template stored in `prompts/<prompt_name>.txt`."""
    if prompt_name in _prompt_cache:
        return _prompt_cache[prompt_name]

    path = PROMPTS_DIR / f"{prompt_name}.txt"
    if not path.is_file():
        available = sorted(p.stem for p in PROMPTS_DIR.glob("*.txt"))
        raise ValueError(
            f"Prompt template '{prompt_name}' not found in {PROMPTS_DIR}. "
            f"Available templates: {available}"
        )
    # newline="" keeps the file content byte for byte
    with open(path, "r", encoding="utf-8", newline="") as f:
        prompt_template = f.read()
    _prompt_cache[prompt_name] = prompt_template
    return prompt_template


class PromptStrategyBase(ABC):
    """Base class for prompt strategies."""

    def __init__(self, classification_type=None, prompt_variant=None):
        """
        Args:
            classification_type (str): Only "binary" is supported
            prompt_variant (str): Name of the prompt template in `prompts/`
        """
        self.classification_type = classification_type
        self.prompt_variant = prompt_variant

    @abstractmethod
    async def create_prompt(self, claim: str, row: Dict = None) -> str:
        """Create a prompt based on the claim and optional row data."""
        pass


class ClaimOnlyPromptStrategy(PromptStrategyBase):
    """Strategy that uses only the claim without evidence."""

    async def create_prompt(self, claim: str, row: Dict = None) -> str:
        """Create a prompt with just the claim."""
        if self.classification_type == "binary":
            prompt_template = get_prompt(self.prompt_variant)

            # Format with claim
            return prompt_template.format(claim=claim)
        else:
            raise NotImplementedError(
                "Ternary classification with claim only is not implemented yet."
            )


class ClaimWithStaticEvidenceStrategy(PromptStrategyBase):
    """Strategy that uses static evidence from a specific column."""

    def __init__(
        self,
        evidence_column="evidence_tavily_20_results",
        classification_type=None,
        prompt_variant=None,
    ):
        """Initialize with evidence column name, classification type, and prompt variant."""
        super().__init__(classification_type, prompt_variant)
        self.evidence_column = evidence_column

    async def create_prompt(self, claim: str, row: Dict = None) -> str:
        """Create a prompt with the claim and static evidence from the specified column."""
        # Extract evidence from the specified column
        evidence = ""
        if row is not None and self.evidence_column in row:
            evidence = row[self.evidence_column]
        else:
            evidence = "No evidence found for this claim."
            logging.warning(
                f"Evidence column '{self.evidence_column}' not found in row. Using default message"
            )
        if self.classification_type == "binary":
            prompt_template = get_prompt(self.prompt_variant)
            return prompt_template.format(claim=claim, evidence=evidence)
        else:  # ternary classification
            raise NotImplementedError(
                "Ternary classification with static evidence is not implemented yet."
            )
