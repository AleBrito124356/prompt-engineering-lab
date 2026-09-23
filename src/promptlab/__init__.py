"""promptlab -- a tested prompt-engineering library and template engine.

The pure, offline core (template engine, front-matter, versioned prompts) is
imported here. The NIM-calling parts (``client``, ``runner``, ``compare``,
``patterns``) are imported on demand so that importing ``promptlab`` never
requires the ``openai`` package or a network connection.
"""

from __future__ import annotations

from .frontmatter import load_file, split_frontmatter
from .prompt import (
    BUNDLED_LIBRARY,
    ContractError,
    InputSpec,
    Prompt,
    PromptError,
    PromptLibrary,
)
from .template import (
    MissingVariableError,
    PartialNotFoundError,
    Template,
    TemplateError,
    TemplateSyntaxError,
    few_shot,
    find_variables,
    missing_variables,
    render,
)

__version__ = "0.1.0"

__all__ = [
    "Template",
    "render",
    "few_shot",
    "find_variables",
    "missing_variables",
    "TemplateError",
    "TemplateSyntaxError",
    "MissingVariableError",
    "PartialNotFoundError",
    "split_frontmatter",
    "load_file",
    "Prompt",
    "PromptLibrary",
    "PromptError",
    "ContractError",
    "InputSpec",
    "BUNDLED_LIBRARY",
    "__version__",
]
