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

Minipaper/reviewer hypothesis layer:

- `docs/minipaper_reviewer_architecture.md`
- `scripts/internal/run_minipaper_experiment.py`
- `scripts/hypotheses/run_h1_reviewer_experiments.py`
- `scripts/hypotheses/run_h2_cross_provider_review.py`
- `utils/minipaper_protocol.py`
- `utils/minipaper_engine.py`

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
- optional: `GENAI4SCIENCE_FALLBACK_BASE_URL`

GenAI4Science currently uses:

```text
https://genai.science-cloud.hu/performance-api/

Recommended G4S setup:

```env
GENAI4SCIENCE_BASE_URL=https://genai.science-cloud.hu/performance-api/
GENAI4SCIENCE_FALLBACK_BASE_URL=https://genai.science-cloud.hu/api/
```

The client uses the performance endpoint first and automatically falls back to the standard endpoint if the performance endpoint fails.
```

## Official Entry Point

Use `run_pipeline.py` unless you are debugging the internals.

Two presets exist:

- `quick`: bounded smoke test
- `benchmark`: full sweep or filtered benchmark sweep

### Quick Smoke Test

OpenAI example:

```bash
python run_pipeline.py --preset quick --model_name gpt5mini --api_source oa
```

GenAI4Science example:

```bash
python run_pipeline.py --preset quick --model_name gemma4:31b --api_source g4s
```

### Benchmark Sweep

Single provider:

```bash
python run_pipeline.py --preset benchmark --model_name gpt5mini --api_source oa
```

Multiple providers in one orchestrated run:

```bash
python run_pipeline.py --preset benchmark --model_name gpt5mini --api_source oa,or
```

Single law version only:

```bash
python run_pipeline.py --preset benchmark --model_name gpt5mini --api_source oa --law_version v0
```

Include `v_unchanged` control tasks:

```bash
python run_pipeline.py --preset benchmark --model_name gpt5mini --api_source oa --include_unchanged
```

Use all models from:

- `configs/models.txt`

```bash
python run_pipeline.py --preset benchmark --api_source oa
```

Use a provider-specific model list without touching the default model file:

```bash
python run_pipeline.py --preset benchmark --api_source g4s --models_file configs/models_g4s.txt
```

This is the recommended way to run GenAI4Science-only sweeps. In other words:

- if you pass `--model_name`, you do **not** need to edit any model list
- if you want a multi-model G4S sweep, prefer a separate file such as `configs/models_g4s.txt`
- do **not** overwrite `configs/models.txt` unless you intentionally want to change the default benchmark model pool

The repository already includes:

- `configs/models_g4s.txt`

with the current recommended GenAI4Science model set:

- `gemma4:31b`
- `llama3.3:70b`
- `gpt-oss:120b`

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

In addition, `run_pipeline.py` now prints a short terminal summary automatically after report generation. For each provider/model/backend/prompt/consistency group in the run, it prints:

- aggregated exact accuracy
- aggregated RMSLE
- aggregated trial success rate

So you no longer need to call a second evaluation command just to see whether the run discovered anything correctly.

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

## Minipaper + Reviewer Hypothesis Layer

This fork now also contains a separate experimental foundation for a new protocol in which scientist agents no longer submit only a bare final law. Instead, each scientist produces a short **minipaper** containing:

- the proposed law
- a short justification paragraph

A separate reviewer agent then decides whether that minipaper should enter the shared knowledge base. Only accepted papers become reusable context for future agents in the same scenario.

This layer intentionally freezes several assumptions relative to the legacy benchmark path:

- it always uses consistent law modifications
- it always uses the new minipaper-oriented prompt logic
- it uses accepted minipapers, rather than raw discoveries, as shared context

The full design is documented in:

- `docs/minipaper_reviewer_architecture.md`

The current implementation also supports bounded revision fallback:

- if a minipaper is rejected, the scientist can revise and resubmit it
- the fallback depth is controlled by `--max_review_rounds`
- the default is `2`, meaning at most one revision after the first rejection

### Generic single-scenario run

```bash
python scripts/internal/run_minipaper_experiment.py --run_tag minipaper-demo --scientist_model_name gpt5mini --scientist_api_source oa --reviewer_model_name gemma4:31b --reviewer_api_source g4s --modules m0_gravity,m1_coulomb_force --equation_difficulties easy --model_systems vanilla_equation --law_versions v0,v1 --reviewer_can_run_experiments --max_review_rounds 2
```

### H1 runner

Hypothesis H1 tests whether reviewer-side experimentation improves aggregate outcomes.

```bash
python scripts/hypotheses/run_h1_reviewer_experiments.py --scientist_model_name gpt5mini --scientist_api_source oa
```

For a full multi-model GenAI4Science batch:

```bash
bash scripts/hypotheses/H1_runner.sh
```

This runner compares:

- reviewer without experiment access
- reviewer with experiment access

and writes run-specific outputs under:

```text
outputs/hypothesis_runs/<run_tag>/
```

including:

- `paper_rounds.csv`
- `paper_results.csv`
- `scenario_summary.csv`
- `h1_summary.csv`
- scenario-specific `knowledge_base.json`

The batch shell runner also writes aggregated outputs under:

- `outputs/hypothesis_runs/<run_group_tag>/h1_summary_all.csv`
- `outputs/hypothesis_runs/<run_group_tag>/scenario_summary_all.csv`
- `outputs/hypothesis_runs/<run_group_tag>/paper_results_all.csv`
- `outputs/hypothesis_runs/<run_group_tag>/paper_rounds_all.csv`

### H2 runner

Hypothesis H2 tests whether cross-provider review is stricter than same-provider review.

```bash
python scripts/hypotheses/run_h2_cross_provider_review.py --openai_model_name gpt5mini --open_model_name gemma4:31b --open_api_source g4s --max_review_rounds 2
```

For a full multi-model batch:

```bash
bash scripts/hypotheses/H2_runner.sh
```

The non-OpenAI side is provider-parametric. Use `model@provider` entries in `MODELS_CSV` or `MODELS_FILE`, for example `gemma4:31b@g4s` or `gem25p@or`.

This runner compares four scenarios:

- OpenAI scientist / OpenAI reviewer
- non-OpenAI scientist / non-OpenAI reviewer
- OpenAI scientist / non-OpenAI reviewer
- non-OpenAI scientist / OpenAI reviewer

and writes:

- `paper_rounds.csv`
- `paper_results.csv`
- `scenario_summary.csv`
- `h2_summary.csv`

The batch shell runner also writes aggregated outputs under:

- `outputs/hypothesis_runs/<run_group_tag>/h2_summary_all.csv`
- `outputs/hypothesis_runs/<run_group_tag>/scenario_summary_all.csv`
- `outputs/hypothesis_runs/<run_group_tag>/paper_results_all.csv`
- `outputs/hypothesis_runs/<run_group_tag>/paper_rounds_all.csv`

### H3 runner

Hypothesis H3 tests whether incorrect accepted knowledge-base entries degrade later related discoveries.

```bash
python scripts/hypotheses/run_h3_poisoned_kb.py --scientist_model_name gemma4:31b --scientist_api_source g4s --reviewer_model_name gemma4:31b --reviewer_api_source g4s --poison_rate 0.1 --poison_edit_distance 1
```

H3 has two analysis paths:

- direct intervention: compare clean seeded related minipapers with intentionally perturbed related minipapers
- observational log analysis: scan previous `paper_results.csv` files for accepted wrong minipapers followed by later related failures

Direct H3 writes:

- `h3_summary.csv`
- `h3_paper_results.csv`
- `poison_manifest.csv`

The observational analysis is:

```bash
bash scripts/hypotheses/H3_log_analysis_runner.sh
```

### H4 runner

Hypothesis H4 tests the effect of thinking-mode on the scientist and reviewer sides in a 2x2 design.

```bash
python scripts/hypotheses/run_h4_thinking_mode.py --scientist_model_name gem25p --scientist_api_source or --reviewer_model_name gem25p --reviewer_api_source or
```

The four H4 conditions are:

- scientist thinking off / reviewer thinking off
- scientist thinking on / reviewer thinking off
- scientist thinking off / reviewer thinking on
- scientist thinking on / reviewer thinking on

Thinking-mode is implemented as a private system-level instruction. For OpenRouter models, the API call also requests provider-side reasoning when supported. The visible minipaper and review output formats remain unchanged.

### Run all hypotheses

To run H1-H4 across all configured models:

```bash
bash scripts/hypotheses/run_all_hypotheses_all_models.sh
```

Useful controls:

- `MAX_PARALLEL_RUNS=4` limits concurrently launched hypothesis jobs
- `HYPOTHESES=H1,H3,H4` runs a subset
- `MODELS_CSV=gemma4:31b@g4s,gem25p@or` overrides the model file and includes provider tags
- `REVIEWER_CAN_RUN_EXPERIMENTS=1` enables reviewer-side experiments for H2-H4
- `DRY_RUN=1` validates dispatch and manifests without API calls

On Windows Git Bash, use `PYTHON_BIN=python.exe` if `python` is not visible inside Bash.

### Operational note

The minipaper protocol is inherently API-expensive. Each episode can include:

- multiple scientist-side experiment rounds
- a minipaper generation step
- an optional reviewer-side experiment loop
- a review decision
- evaluator calls on the submitted law

So the new hypothesis runners should be treated as high-call-count experimental workflows rather than as cheap benchmark sweeps.

## End-to-End Recipes for Provider Comparison

If you want to compare closed-source OpenAI models and open-weight GenAI4Science models on the same benchmark protocol, use separate run tags and keep the provider explicit in every run.

### Example A: full four-way prompt × consistency study for one OpenAI model

```bash
python run_pipeline.py --preset benchmark --model_name gpt5mini --api_source oa --include_unchanged --prompt_set original --run_tag oa-gpt5mini-original-inconsistent
python run_pipeline.py --preset benchmark --model_name gpt5mini --api_source oa --include_unchanged --prompt_set original --consistency --run_tag oa-gpt5mini-original-consistent
python run_pipeline.py --preset benchmark --model_name gpt5mini --api_source oa --include_unchanged --prompt_set modified --run_tag oa-gpt5mini-modified-inconsistent
python run_pipeline.py --preset benchmark --model_name gpt5mini --api_source oa --include_unchanged --prompt_set modified --consistency --run_tag oa-gpt5mini-modified-consistent
```

Then aggregate:

```bash
python result_analysis/compare_prompt_consistency.py --result_dir evaluation_results --output_dir outputs/pipeline_runs/oa-gpt5mini-prompt-consistency/report --original_inconsistent_run_tag oa-gpt5mini-original-inconsistent --original_consistent_run_tag oa-gpt5mini-original-consistent --modified_inconsistent_run_tag oa-gpt5mini-modified-inconsistent --modified_consistent_run_tag oa-gpt5mini-modified-consistent
```

### Example B: full four-way prompt × consistency study for all GenAI4Science models in a dedicated file

First create a G4S-only model file such as:

```text
configs/models_g4s.txt
```

with entries like:

```text
gemma4:31b
llama3.3:70b
gpt-oss:120b
```

Then run:

```bash
python run_pipeline.py --preset benchmark --api_source g4s --models_file configs/models_g4s.txt --include_unchanged --prompt_set original --run_tag g4s-original-inconsistent
python run_pipeline.py --preset benchmark --api_source g4s --models_file configs/models_g4s.txt --include_unchanged --prompt_set original --consistency --run_tag g4s-original-consistent
python run_pipeline.py --preset benchmark --api_source g4s --models_file configs/models_g4s.txt --include_unchanged --prompt_set modified --run_tag g4s-modified-inconsistent
python run_pipeline.py --preset benchmark --api_source g4s --models_file configs/models_g4s.txt --include_unchanged --prompt_set modified --consistency --run_tag g4s-modified-consistent
```

Then aggregate:

```bash
python result_analysis/compare_prompt_consistency.py --result_dir evaluation_results --output_dir outputs/pipeline_runs/g4s-prompt-consistency/report --original_inconsistent_run_tag g4s-original-inconsistent --original_consistent_run_tag g4s-original-consistent --modified_inconsistent_run_tag g4s-modified-inconsistent --modified_consistent_run_tag g4s-modified-consistent
```

### Example C: compare only consistency at fixed prompt set

For a fixed prompt set such as `original`:

```bash
python result_analysis/compare_consistency.py --result_dir evaluation_results --output_dir outputs/pipeline_runs/g4s-original-consistency-only/report --inconsistent_run_tag g4s-original-inconsistent --consistent_run_tag g4s-original-consistent
```

and analogously for OpenAI:

```bash
python result_analysis/compare_consistency.py --result_dir evaluation_results --output_dir outputs/pipeline_runs/oa-gpt5mini-original-consistency-only/report --inconsistent_run_tag oa-gpt5mini-original-inconsistent --consistent_run_tag oa-gpt5mini-original-consistent
```

### Practical note

At the moment, provider-level comparison is done by running separate studies with different `run_tag`s and then comparing the resulting CSV files side by side. The generated comparison tables already keep `api_source`, so downstream plotting and merged analysis can remain provider-aware.

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
