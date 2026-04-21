# NewtonBench Fork for Consistency and Open-Weight Experiments

This repository is a research fork of the original NewtonBench benchmark. It keeps the core task family and evaluation philosophy of NewtonBench, but extends the benchmark and the experiment stack in several directions that are central to our current project:

- physically modified benchmark laws beyond the upstream release
- consistency-preserving cross-law transformations
- support for open-weight models through OpenAI-compatible endpoints
- a web-based dashboard that can be used either as a passive monitor or as part of the modified-prompt workflow

The repository is therefore best understood as a NewtonBench-derived experimental platform rather than as a clean mirror of the upstream codebase.

## What Changed Relative to Upstream NewtonBench

### 1. Physics

The benchmark laws in this fork are no longer restricted to the original upstream metaphysical shifts. We added and revised law variants, including explicit `v_unchanged` control variants and additional edited law families used in our own experiments.

Operational consequences:

- the benchmark can be run on the original shifted laws (`v0`, `v1`, `v2`)
- the control variant `v_unchanged` can be included explicitly with `--include_unchanged`
- the exact task count depends on whether that control set is included

### 2. Consistency

This fork adds a consistency-preserving mode controlled by `--consistency`.

The conceptual relations behind consistency are externalized in:

- `consistency_groups.yml`

The YAML separates:

- strict consistency axes
- prompt-time relevance axes

This means that some law families are synchronized structurally when `--consistency` is enabled, while broader cross-task relations are used only for prompt-time retrieval.

Related paper fragments in the repo:

- `consistency_policy.tex`
- `consistency_test.tex`
- `consistency_prompt_results.tex`

### 3. Open-Weight Models via OpenAI-Compatible APIs

This fork supports three explicit API sources:

- `oa`: direct OpenAI
- `or`: OpenRouter
- `g4s`: GenAI4Science OpenAI-compatible endpoint

The OpenAI-compatible routing is handled centrally in:

- `utils/call_llm_api.py`

Important design choice:

- provider selection is explicit
- there is no silent provider fallback once `--api_source` is specified

This keeps runs reproducible and prevents accidental mixing of providers inside one benchmark result set.

### 4. Web Interface and Dashboard

This fork includes a lightweight web dashboard under:

- `mini_scientist/dashboard/index.html`

The local server is:

- `mini_scientist/server.py`

The dashboard reads and visualizes:

- `accumulation/global_kg.json`

The dashboard serves two distinct roles:

1. Passive monitoring
   - you may enable it simply to watch current best laws and graph updates during a run
2. Prompt-time memory for the modified prompt
   - when `prompt_set=modified`, the accumulated discovered laws are used as prompt context

That second role matters because it changes when the dashboard should be on.

## Repository Layout

Public entrypoint at the repository root:

- `run_pipeline.py`

Internal runners:

- `scripts/internal/run_experiments.py`
- `scripts/internal/run_all_evaluations.py`

Legacy wrappers moved out of the root:

- `scripts/legacy/quick_start.py`
- `scripts/legacy/run_master.py`
- `scripts/legacy/run_benchmark.py`
- `scripts/legacy/comprehensive_benchmark.py`
- `scripts/legacy/run_system.py`

Result analysis:

- `result_analysis/summarize_results.py`
- `result_analysis/compare_consistency.py`
- `result_analysis/compare_prompt_consistency.py`

Prompt and dashboard utilities:

- `utils/prompt_utils.py`
- `utils/kg_utils.py`
- `mini_scientist/server.py`

## Installation

### 1. Environment

```bash
conda create --name newtonbench python=3.10.18
conda activate newtonbench
```

### 2. Dependencies

```bash
pip install -r requirements.txt
```

For a more reproducible local environment:

```bash
pip install -r requirements.lock.txt
```

Dependency policy in this fork:

- `requirements.txt` uses bounded version ranges
- `requirements.lock.txt` pins a concrete snapshot
- `networkx` and `PyYAML` are required by the current codebase
- `pysr` is intentionally not part of the default install because it is only needed in the `mini_scientist` symbolic-regression path

### 3. API Keys

Copy `.env.example` to `.env`, then fill only the providers you need.

Supported variables:

- `OPENAI_API_KEY`
- `OPENROUTER_API_KEY`
- `GENAI4SCIENCE_API_KEY`
- optional: `OPENAI_BASE_URL`
- optional: `GENAI4SCIENCE_BASE_URL`

GenAI4Science currently uses:

```text
https://genai.science-cloud.hu/api/
```

## Official Entry Point

Use `run_pipeline.py` unless you are debugging the internals.

Two presets exist:

- `quick`: bounded smoke test
- `benchmark`: full sweep or filtered benchmark sweep

### Quick Smoke Test

OpenAI example:

```bash
python run_pipeline.py --preset quick --model_name gpt41mini --api_source oa
```

GenAI4Science example:

```bash
python run_pipeline.py --preset quick --model_name llama3.1:8b --api_source g4s
```

### Benchmark Sweep

Single provider:

```bash
python run_pipeline.py --preset benchmark --model_name gpt41mini --api_source oa
```

Multiple providers in one orchestrated run:

```bash
python run_pipeline.py --preset benchmark --model_name gpt41mini --api_source oa,or
```

Single law version only:

```bash
python run_pipeline.py --preset benchmark --model_name gpt41mini --api_source oa --law_version v0
```

Include `v_unchanged` control tasks:

```bash
python run_pipeline.py --preset benchmark --model_name gpt41mini --api_source oa --include_unchanged
```

Use all models from:

- `configs/models.txt`

```bash
python run_pipeline.py --preset benchmark --api_source oa
```

## Prompt Sets

Two prompt modes exist:

- `original`
- `modified`

`original`:

- only the task prompt is shown
- no retrieved prior laws are prepended

`modified`:

- prepends a relevance-filtered set of previously discovered laws from the same conceptual family
- uses `consistency_groups.yml`
- uses the accumulated dashboard state in `accumulation/global_kg.json`

This means `modified` is not just a formatting tweak. It depends on a live accumulation state.

## Web Interface: When It Runs Automatically, and When It Should Not

### When it runs automatically

If you use:

```bash
--prompt_set modified
```

then the benchmark sweep runner will automatically:

- enable dashboard accumulation
- reset `accumulation/global_kg.json` once at sweep start
- keep the accumulated history across child experiment runs inside the same sweep
- start the local dashboard server during the sweep

This is intentional, because the modified prompt uses prior discovered laws as context. Without accumulation, the prompt would be empty or stale.

### When it does not need to run

If you use:

```bash
--prompt_set original
```

then the dashboard is optional.

Use no dashboard when:

- you only care about final CSV outputs
- you want the cleanest, cheapest baseline runs
- you do not need live visualization

Use `--dashboard` with `original` when:

- you want to visually inspect what the agent is discovering
- you want the graph and best-law display while the run is in progress

### Direct single-run caveat

If you call the low-level runner directly:

- `scripts/internal/run_experiments.py`

and you use `--prompt_set modified`, then you should usually also enable:

```bash
--dashboard
```

Otherwise the modified prompt may read an empty or stale `accumulation/global_kg.json`.

Also note:

- `scripts/internal/run_experiments.py --dashboard` resets the accumulation by default
- `--keep_history` disables that reset
- this is useful for manually staged, seed-then-target experiments

## Judge Model Behavior

The benchmark uses an LLM judge for exact symbolic equivalence checks.

You can control it explicitly:

```bash
--judge_model_name ...
--judge_api_source ...
```

If you do not specify a judge and you run with `--api_source g4s`, the code falls back to using the evaluated model itself as judge, so that a G4S-only environment remains runnable.

This is convenient operationally, but methodologically you should be aware that:

- changing the judge model changes the exact symbolic equivalence pipeline

## Output Structure

Each pipeline run writes:

- `outputs/pipeline_runs/<run_tag>/pipeline.log`
- `outputs/pipeline_runs/<run_tag>/manifest.json`
- `outputs/pipeline_runs/<run_tag>/RESULTS_INDEX.md`
- `outputs/pipeline_runs/<run_tag>/report/...`

The fastest file to inspect law-level accuracy is:

- `outputs/pipeline_runs/<run_tag>/report/law_accuracy_summary.csv`

The most useful files are:

- `law_accuracy_summary.csv`
- `config_summary.csv`
- `aggregated_trial_summary.csv`
- `results_by_trial.csv`
- `summary_report.md`

Provider-aware result storage now lives under:

```text
evaluation_results/<api_source>/<model_name>/<module>/<backend>/<difficulty>/<law_version>/...
```

This prevents OpenAI, OpenRouter, and GenAI4Science runs from overwriting each other.

## Comparison Reports

Consistency comparison:

```bash
python result_analysis/compare_consistency.py --result_dir evaluation_results --output_dir outputs/pipeline_runs/<compare_tag>/report --inconsistent_run_tag <run_a> --consistent_run_tag <run_b>
```

Prompt × consistency comparison:

```bash
python result_analysis/compare_prompt_consistency.py --result_dir evaluation_results --output_dir outputs/pipeline_runs/<compare_tag>/report --original_inconsistent_run_tag <run_a> --original_consistent_run_tag <run_b> --modified_inconsistent_run_tag <run_c> --modified_consistent_run_tag <run_d>
```

The comparison tables now also retain:

- `api_source`

so multi-provider studies can be analyzed without ambiguity.

## Guidance for New Collaborators

If you are joining development on this fork, start in this order:

1. Read this README.
2. Run a quick smoke test through `run_pipeline.py`.
3. Inspect `utils/call_llm_api.py` for provider routing.
4. Inspect `scripts/internal/run_experiments.py` and `scripts/internal/run_all_evaluations.py` for experiment orchestration.
5. Inspect `utils/prompt_utils.py` and `consistency_groups.yml` for the modified prompt and consistency logic.

## Notes

- This fork intentionally favors explicitness and reportability over hidden convenience behavior.
- If a provider, prompt set, or consistency regime differs, it should remain visible in the filesystem layout and the generated tables.
- For current recommended commands, see also `futtatas.md`.
