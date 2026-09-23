"""A thin NVIDIA NIM client built on the OpenAI-compatible SDK.

NIM exposes an OpenAI-compatible endpoint, so we reuse the ``openai`` package.
The API key is read from ``NVIDIA_API_KEY`` and the model from ``NIM_MODEL``
(defaulting to ``meta/llama-3.3-70b-instruct``). If the key is missing we raise
``MissingKeyError`` carrying a short signup walkthrough; the CLI prints it as a
one-line ``error:`` plus the walkthrough, never a stack trace. Failures reported
by the ``openai`` SDK (network, auth, rate limits) surface as ``BackendError``.

``ChatClient`` is the interface every backend shares -- ``chat`` plus the
``complete`` and ``sample`` helpers built on it -- so the technique modules,
the runner and ``compare`` work unchanged against NIM or an offline backend.
"""

from __future__ import annotations

import os

__all__ = [
    "ChatClient",
    "NIMClient",
    "MissingKeyError",
    "BackendError",
    "get_openai_client",
    "DEFAULT_MODEL",
    "BASE_URL",
]

DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"
BASE_URL = "https://integrate.api.nvidia.com/v1"

_FRIENDLY_KEY_MESSAGE = (
    "NVIDIA_API_KEY is not set.\n"
    "\n"
    "promptlab runs prompts on NVIDIA NIM, which has a free tier:\n"
    "  1. Create a free account at https://build.nvidia.com\n"
    "  2. Open any model and click 'Get API Key' -- it starts with 'nvapi-'\n"
    "  3. Put it in your environment or .env file:\n"
    "       NVIDIA_API_KEY=nvapi-XXXXXXXXXXXXXXXXXXXXXXXX\n"
    "\n"
    "The whole thing takes about two minutes, then re-run the command."
)


class MissingKeyError(RuntimeError):
    """Raised when NVIDIA_API_KEY is not configured."""

    def __init__(self, message=_FRIENDLY_KEY_MESSAGE):
        super().__init__(message)


class BackendError(RuntimeError):
    """A model call failed (network, authentication, rate limit, bad model id...)."""


def _is_sdk_error(exc):
    """True for exceptions raised by the ``openai`` SDK (or httpx underneath it)."""
    module = type(exc).__module__ or ""
    return module.split(".")[0] in ("openai", "httpx", "httpcore")


def _load_dotenv():
    """Best-effort load of a local .env; a no-op if python-dotenv is absent."""
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()


def get_openai_client():
    """Return an ``openai.OpenAI`` client pointed at NIM, or raise MissingKeyError."""
    _load_dotenv()
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        raise MissingKeyError()
    from openai import OpenAI

    return OpenAI(base_url=BASE_URL, api_key=key)


class ChatClient:
    """Shared interface: subclasses implement ``chat``; ``complete``/``sample`` come free."""

    model = DEFAULT_MODEL

    def chat(self, messages, *, temperature=0.7, max_tokens=1024, n=1, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def complete(self, user, *, system=None, **kwargs):
        """Single-turn helper: optional ``system`` prompt plus a ``user`` message."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return self.chat(messages, **kwargs)

    def sample(self, messages, n, *, temperature=0.7, max_tokens=1024, **kwargs):
        """Draw ``n`` independent completions with one call per sample.

        Sampling one at a time (rather than relying on the ``n`` parameter) keeps
        self-consistency working across NIM models that do not return multiple
        choices in a single response.
        """
        return [
            self.chat(messages, temperature=temperature, max_tokens=max_tokens, n=1, **kwargs)
            for _ in range(n)
        ]


class NIMClient(ChatClient):
    """Convenience wrapper around chat completions on NIM."""

    def __init__(self, model=None, client=None):
        self.model = model or os.environ.get("NIM_MODEL", DEFAULT_MODEL)
        self._client = client

    @property
    def client(self):
        if self._client is None:
            self._client = get_openai_client()
        return self._client

    def chat(self, messages, *, temperature=0.7, max_tokens=1024, n=1, **kwargs):
        """Send chat ``messages``. Returns a string when ``n == 1``, else a list."""
        client = self.client  # MissingKeyError propagates unchanged
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                **kwargs,
            )
        except Exception as exc:
            if _is_sdk_error(exc):
                raise BackendError(
                    "NIM request to model {!r} failed: {}: {}".format(
                        self.model, type(exc).__name__, exc
                    )
                ) from exc
            raise
        contents = [choice.message.content or "" for choice in resp.choices]
        return contents[0] if n == 1 else contents
