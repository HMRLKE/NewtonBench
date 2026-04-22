# Minipaper + Reviewer Architecture

## Goal

This experimental layer introduces a new protocol on top of the existing NewtonBench-style task modules. Instead of treating the final answer as a bare function, scientist agents communicate through a structured **minipaper**. A minipaper contains:

- a discovered law in executable Python form
- a short justification paragraph explaining why the agent believes the law is correct
- optional references to previously accepted regularities from the shared knowledge base

The central design change is that a scientist proposal no longer enters the shared memory automatically. Every minipaper must be reviewed by a separate reviewer agent, which either accepts or rejects the proposal. Only accepted minipapers become part of the run-specific shared knowledge base.

## Frozen assumptions for this layer

This architecture intentionally freezes several choices that used to be free flags in the legacy benchmark pipeline:

- physics is always evaluated under **consistent** law modifications
- the scientist prompt always uses the new **minipaper-oriented modified prompt**
- the knowledge shared between agents consists only of **accepted** minipapers, never raw proposals

These assumptions keep the hypothesis runners interpretable and prevent the new protocol from becoming another large flag matrix.

## Core entities

### Scientist agent

The scientist agent explores the task with the existing experiment interface. Its final output is a `<mini_paper>` block rather than a bare `<final_law>` block. The scientist still proposes experiments through `<run_experiment>` tags, but the terminal deliverable is now a structured artifact with:

- an equation block containing `def discovered_law(...)`
- a short justification block

### Reviewer agent

The reviewer agent receives:

- the task setup
- the scientist minipaper
- the currently accepted shared context

The reviewer returns a `<review_decision>` block that contains:

- `accept` or `reject`
- a short rationale
- a lightweight confidence field

The reviewer can optionally run its own experiments. This is the main manipulated variable in hypothesis H1.

### Shared knowledge base

The shared knowledge base is run-specific. It stores only accepted minipapers and is written to the hypothesis run directory. It is therefore:

- persistent across tasks inside a scenario
- isolated between scenarios
- safe to use for provider and hypothesis comparisons

The scientist prompt retrieves only conceptually related accepted minipapers, using the existing conceptual grouping logic.

## Execution model

The new protocol is implemented as a separate experimental layer rather than as another branch of the legacy benchmark runner. Each scenario is defined by:

- scientist model name and provider
- reviewer model name and provider
- whether the reviewer may run experiments

For a given task, the engine runs a small population of scientist-reviewer episodes. Each episode proceeds as follows:

1. the scientist receives the task prompt plus accepted contextual minipapers
2. the scientist interacts with the simulator and submits a minipaper
3. the discovered law is evaluated with the existing module-specific evaluator
4. the reviewer inspects the minipaper and optionally runs experiments
5. the reviewer accepts or rejects the proposal
6. if the proposal is rejected and review rounds remain, the scientist receives the rejected draft plus reviewer feedback and submits a revised minipaper
7. only the final accepted paper, if any, is inserted into the shared knowledge base

This preserves the implicit-memory design while keeping the evaluation path anchored in the existing module interface.

## Metrics

Each generated minipaper yields two classes of signals:

- **scientific correctness**
  - exact accuracy
  - RMSLE
- **review-layer behavior**
  - acceptance decision
  - acceptance rate
  - accepted-paper accuracy
  - false acceptance behavior

The most important aggregate metric for the new layer is the percentage of **correctly accepted** papers, i.e. accepted proposals that are also exact matches under the existing evaluator.

## Hypothesis runners

Each major experimental question gets its own dedicated runner.

### H1

`H1`: reviewer-side experimentation improves aggregate outcomes.

Operationalized as a controlled comparison between:

- reviewer cannot run experiments
- reviewer can run experiments

The default primary metric is `accepted_correct_rate_pct`, and the runner reports whether the reviewer-with-experiments condition improves this metric by at least 10% relative.

### H2

`H2`: cross-provider review is less lenient than same-provider review.

Operationalized as comparisons between:

- OpenAI scientist / OpenAI reviewer
- G4S scientist / G4S reviewer
- OpenAI scientist / G4S reviewer
- G4S scientist / OpenAI reviewer

The primary reported signals are:

- accuracy among accepted papers
- false acceptance rate

This makes it possible to test whether heterogeneous review pairs filter erroneous proposals more aggressively than homogeneous pairs.

## Revision fallback

The current implementation supports bounded fallback through a `max_review_rounds` parameter. With `max_review_rounds=2`, an episode may proceed as:

1. scientist draft 1
2. reviewer decision 1
3. revised scientist draft 2
4. reviewer decision 2

If the first review is a rejection, the next scientist draft receives:

- the previous minipaper
- the reviewer rationale
- the reviewer confidence

This keeps the protocol iterative without allowing unbounded re-submission loops.

## Why this is a separate subsystem

The legacy benchmark path is still useful for baseline NewtonBench-style experiments. However, the minipaper protocol changes:

- the final artifact
- the shared-memory semantics
- the role of the second agent
- the cost profile of a run

For that reason, the minipaper/reviewer system is implemented as a new foundation layer with dedicated hypothesis runners, logs, summaries, and knowledge-base files.
