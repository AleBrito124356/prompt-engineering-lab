"""Render a library prompt with variables and run it on a model backend.

The library prompts are *system* prompts: they define behaviour, and you then
send an actual user turn. ``run_prompt`` renders the chosen version with your
variables (defaults from its input contract filled in), uses it as the system
message, and sends ``user_input`` as the user message.

The model comes from ``client`` when given, else from ``backend`` (``"nim"``,
``"mock"``, ``"record:FILE"``, ``"replay:FILE"``), else from
``$PROMPTLAB_BACKEND``, else live NVIDIA NIM.
"""

from __future__ import annotations

from .backends import make_client
from .prompt import PromptLibrary, default_library_root

__all__ = ["run_prompt", "render_prompt", "default_library"]


def default_library(root=None):
    """Return a :class:`PromptLibrary` for ``root``.

    Defaults to ``$PROMPTLAB_LIBRARY`` when set, else the library bundled
    inside the package (``promptlab/library``).
    """
    return PromptLibrary(root or default_library_root())


def render_prompt(name, variables=None, *, version=None, root=None, strict=True):
    """Render a library prompt to text without calling the network."""
    return default_library(root).render(
        name, variables or {}, version=version, strict=strict
    )


def run_prompt(
    name,
    variables=None,
    *,
    user_input=None,
    version=None,
    root=None,
    client=None,
    model=None,
    backend=None,
    temperature=0.3,
    max_tokens=1024,
):
    """Render a library prompt and run it.

    Parameters
    ----------
    name:
        Prompt folder name, e.g. ``"sql-expert"``.
    variables:
        Values for the template variables in the system prompt.
    user_input:
        The user turn. Optional -- some prompts embed the input via variables.
    version:
        Version spec (``"v2"``, ``"2"``, ``"latest"`` or ``None`` for latest).
    client / backend / model:
        An explicit client, or a backend spec and model for ``make_client``.
    """
    system = render_prompt(name, variables or {}, version=version, root=root)
    llm = client or make_client(backend, model=model)
    user = user_input if user_input is not None else "Begin."
    return llm.complete(
        user, system=system, temperature=temperature, max_tokens=max_tokens
    )
