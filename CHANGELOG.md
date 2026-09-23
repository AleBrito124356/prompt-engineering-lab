# Changelog

All notable changes to this project are documented here.

## 0.2.0

An audit of 0.1.0 found that several technique modules failed on realistic model output, that `compare` could declare a winner from judge position bias alone, and that nothing could run without an API key. This release fixes those problems and adds the tooling the README promised.

### Added

- **Offline mode.** `promptlab.backends` adds `ScriptedClient` (a deterministic fake model), `RecordingClient` / `ReplayClient` (JSONL cassettes keyed by request hash, where a miss shows a diff against the closest recorded request) and a generic `mock_model`. `run` and `compare` take `--offline`, `--record FILE` and `--replay FILE`, and `PROMPTLAB_BACKEND=mock|record:FILE|replay:FILE` does the same for everything.
- **Runnable demos without a key.** Each of the 12 technique modules has `demo_client()` and `main(argv)`. `python -m promptlab.patterns.<name> --offline` runs the full loop on scripted output. The demos also take `--record`, `--replay` and `--model`.
- **Input contracts.** Front-matter `inputs` accept `required` (default `true`), `default` and `enum`. `Prompt.render` fills defaults, rejects values outside an enum and refuses to render with a missing required input. `show` prints the contract. New public API: `ContractError`, `InputSpec`, `parse_inputs`, `apply_contract`, `Prompt.inputs`, `Prompt.missing`, `Prompt.resolve_context`.
- **`promptlab lint`**, which checks each prompt's contract against its body: syntax, unknown partials, undeclared or unused inputs, unguarded optional inputs, bad `latest`, missing changelog entries, missing purpose, malformed contracts, "optional" prose on a required input, inputs used inside `#each`, and a strict smoke render. It exits 1 on errors (or on warnings with `--strict`), with `--format json` for CI.
- **`compare` statistics and reports.** Pairs are judged in both orders by default. `tally` adds a Wilson interval, an exact sign-test p-value, a significance-aware `verdict` and a position-bias rate. New options: `--judge-model`, `--criteria`, `--repeats`, `--single-order`, `--alpha`, `--temperature` and `--out report.json|report.md`. `load_report()` reloads the JSON report.
- `rag_prompt.check_citations()` reports cited, out-of-range and missing citations and detects abstention.
- `guardrail_prompt.screen_input_detailed()` and `leak_scores()`. `tree_of_thought.run(verbose=True)` prints the beam search.
- `template.variable_references()` returns each reference with its loop and guard context.
- `CHANGELOG.md`, `pytest-cov` in the dev extras, and tests: 84 in 0.1.0, 359 now, with line coverage up from 59% to 96%.

### Fixed

- `compare.parse_verdict` turned `Winner: Response 2`, `Winner: **2**` and `**Winner:** 2` into a tie. It now accepts those and `A`/`B`/`first`/`second`/`draw`/`equal`, and it can report "unparseable" with `default=None`.
- `chain_of_thought.parse_final_answer` returned `** 42` for `**Final answer:** 42`. The new `clean_answer()` strips markdown, backticks, `\boxed{}`, math delimiters and trailing punctuation.
- `self_consistency` split votes between `23`, `23 apples` and `**23**`, so `24` could win the vote. Vote keys now normalise numbers, including money, thousands separators and percents, with no float rounding.
- `structured_json.extract_json` failed when the prose contained braces or when the output had two objects. It now scans with `raw_decode` and prefers `` ```json `` fences.
- `react_mini.safe_calculator('9**9**9')` hung the process. The calculator now limits expression length, exponent size and integer size, and reports division by zero and `^` as `ValueError`.
- `guardrail_prompt.check_output` only looked for the first sentence of the system prompt. It now measures word 5-gram overlap against every sentence. `screen_input` flagged harmless text such as "you are now logged out". Its patterns now target instructions to the model.
- `missing_variables` missed outer variables used inside `{{#each}}` and dotted paths with a missing leaf, so the render then raised. It now dry-runs the real render. Variables in an `{{#each}}...{{else}}` branch are no longer treated as loop-local.
- The CLI printed tracebacks for expected errors: a missing key, missing variables in `run`/`compare`, a bad `--vars-json`, a missing inputs file, and template or YAML errors. It now prints `error: ...` and exits 2. SDK failures surface as `BackendError` with exit 1.
- A non-editable install shipped without the prompt library, so `promptlab list` failed. The library moved into the package and is declared as package data.
- `interviewer`, `summarizer`, `translator` and `api-designer` documented inputs as optional but required them. `interviewer` gets a `v2` with a guarded `level`. `translator.target_language` is now an enum.
- The `compare` table used a Unicode ellipsis that a Windows cp1252 console cannot print.

### Changed (behaviour)

- The library lives in `src/promptlab/library/` (`promptlab.BUNDLED_LIBRARY`). `--library` and the new `PROMPTLAB_LIBRARY` still point at any other folder.
- `compare` judges each pair twice by default, so it makes twice as many judge calls. A pair is a win only when both orders agree. The printed verdict names a winner only when the sign test is significant. `tally()["overall"]` keeps its 0.1 meaning (the raw leader) for compatibility, and `tally()["verdict"]` is the significance-aware result.
- CLI exit codes: `2` for usage and configuration errors (this includes a missing key, which used to exit 1 with a traceback), and `1` for lint findings and failed model calls.
- Technique demos return exit code 2 when the key is missing (0.1 printed the message and exited 0).
- `guardrail_prompt.run` withholds a response that the leak check flags (`block_on_leak=True`) and keeps the original in `raw_response`.
- `Prompt.render` enforces the contract: a missing required input raises `ContractError` in strict mode.
- `NIMClient` now subclasses `ChatClient`. `runner.run_prompt` accepts `backend=` and defaults to `$PROMPTLAB_BACKEND`.

## 0.1.0

Initial release: template engine, versioned prompt library, 12 technique modules, A/B compare on NVIDIA NIM.
