# Atomic results.jsonl + error taxonomy + harness fixes

Captured from 2026-05-06 session. Defer; revisit when ready to land.

## Goal

Refactor `results.jsonl` from an event-log (write-on-each-turn) into a
**committed-only canonical scored store**, with a separate `events.jsonl`
as the live audit log. Distinguish model failures (which count toward score)
from provider/harness failures (which don't).

## Architecture: WAL + committed store

| file | write pattern | purpose | reader semantics |
|---|---|---|---|
| `events.jsonl` | append + flush after every turn | live monitoring, forensics | may contain partial attempts; tolerate torn last line |
| `results.jsonl` | atomic batch append at attempt teardown | canonical scored data | every row belongs to a fully-realized attempt |

`results.jsonl` ⊆ `events.jsonl` keyed by `attempt_id`. Reconciliation: any
`attempt_id` in events but not in results = incomplete/failed attempt.

Migration shape: rename current `results.jsonl` → `events.jsonl` (it already
has the events shape), populate a fresh `results.jsonl` going forward via the
clean commit path.

## attempt_status enum (new field, written at attempt teardown)

| value | meaning | scored? |
|---|---|---|
| `completed` | ran to budget OR early-perfect | yes — all rows |
| `model_no_submit` | ran to budget cleanly, model never produced any valid submission | yes — counts as failure (MCC=0 / sentinel) |
| `provider_error` | API call(s) failed unrecoverably | no — drop, retry-eligible |
| `harness_error` | bug/exception in our code | no — fix and re-run |
| `aborted` | operator-initiated SIGINT | no — ignore |

Critical distinction: `model_no_submit` is a **model failure**, not an
aborted attempt. It belongs in `results.jsonl` because pretending it didn't
happen would inflate scores by removing a model's failures from the
denominator.

## Error taxonomy (cause × scoring)

### Model errors — count

Whatever the model emits, the model owns the consequences:

- Refused / didn't engage with task → 0
- Thinking-mode hits max_tokens mid-thought without submitting → 0
- Submits malformed tool JSON args → 0 for that turn (don't abort attempt)
- Compile failures, runtime crashes (segfault, abort, signal kill) → tests
  scored as failures
- Submitted code hangs (per-test timeout) → those tests fail
- Calls a wrong tool (not `submit`) → harness ignores, model failure
- Multi-turn: never submits across N turns → 0

### Provider errors — don't count, retry

- HTTP 5xx, 429, connection reset
- API timeout with zero or near-zero tokens received
- Streaming connection drops mid-response
- Provider returns malformed API JSON
- Empty assistant response (after a tool call)

**Ambiguity**: API timeout *after* tokens received could be either model
hanging server-side or provider lost-connection. Heuristic: tokens arrived
AND timeout matches typical thinking-mode budget → lean model. Zero/few
tokens → lean provider.

### Harness errors — don't count, fix

- Python exception in our code
- OOM in harness process
- Disk full mid-write
- SIGKILL by external

These leave the same on-disk shape regardless of cause — partial attempt with
no committed status row. Atomic-write semantics naturally drop them.

### Sandbox / test runner errors — split

Anything *in the model's code's execution* = model failure.
Anything in *the harness around it* = our failure.

| condition | classification |
|---|---|
| Submitted binary segfaults / hangs / OOMs | model failure |
| Docker daemon crashes / image gone | harness/infra |
| Disk full inside container during compile | harness/infra (cf. yaml-1.2-go tmpfs TODO) |
| Test driver itself crashes | harness failure |

## Migration of existing data

Per analysis of 165 attempts in current `results.jsonl`:

- **151 (91.5%) complete cleanly** under model-centric scoring (asst_count >= 5
  OR max_mcc >= 1.0)
- **14 (8.5%) incomplete**:
  - **13 provider_error**: next API call after a tool result silently failed.
    Different providers, different times, same shape: tool result returned,
    no further model response, attempt terminated without status marker.
  - **1 borderline model_failure**: kimi yaml, "Invalid tool arguments:
    Unterminated string". Model produced malformed JSON. Borderline because
    harness's "abort the attempt" response is a policy choice — under the
    proposed taxonomy this should be "treat as failed turn, continue loop."

**Plan**: drop all 14, re-run under fixed harness. None of them produce data
reliably attributable to model performance.

### Cells losing data

| slug | task | n_lost | n_after |
|---|---|---|---|
| kimi-k2.6-thinking | yaml-1.2-cpp17 | 4 | 6 (was 10) |
| kimi-k2.6-thinking | toml-1.0-nospec-cpp17 | 2 | 2 (was 4) |
| kimi-k2.6-thinking | toml-1.0-cpp17 | 1 | **0** (cell empty) |
| gpt-5.5-xhigh | yaml-1.2-nospec-cpp17 | 1 (mcc=0.993) | 4 |
| gpt-5.5-xhigh | toml-1.0-nospec-cpp17 | 1 (the stub outlier) | 4 |
| opus-4-7-adaptive | yaml-1.2-cpp17 | 1 (mcc=0.964) | 4 |
| deepseek-v4-pro-thinking | toml-1.0-cpp17 | 2 | 3 |
| deepseek-v4-pro-thinking | yaml-1.2-cpp17 | 1 | 10 |
| qwen3.6-27b-q6_k_xl | yaml-1.2-cpp17 | 1 | 13 |

Notable signal lost: opus 0.964, gpt-5.5 0.993, kimi 0.937, kimi 0.944.
Real scores but truncated; we don't know what the next turns would have
done.

## Harness fixes (priority order)

1. **Wrap post-tool-result model call in retry loop** (3 attempts, exponential
   backoff). This is the dominant `tool→silence` failure mode — addresses
   13/14 of our incompletes.
2. **Write explicit `attempt_status` row at attempt teardown** in the
   harness's `finally` block, so even crashes leave a marker.
3. **For "Invalid tool arguments"**: change policy to "treat as failed turn,
   continue loop." Currently aborts whole attempt unnecessarily.
4. **For empty assistant response**: treat as failed turn, continue with
   nudge ("you returned an empty response, try again"). Currently the loop
   doesn't recover.
5. **Buffer-then-commit semantics for `results.jsonl`**: hold attempt's rows
   in memory; flush as a batch at clean teardown only.
6. **Rename current `results.jsonl` → `events.jsonl`** as part of the cutover.

## Open questions

- For `model_no_submit`: write per-iteration rows with `error: model_no_submit, mcc: null`
  for each silent turn, OR a single attempt-summary row at teardown? Per-iteration
  preserves the iteration-vs-submission distinction the data already has.
- For `Invalid tool arguments`: always a model failure, or can a provider's
  tool-call layer also produce malformed args? (rare but possible)
- Should `events.jsonl` and `results.jsonl` share a row schema, or should
  `results.jsonl` add an `attempt_status` field that `events.jsonl` doesn't
  need? Probably former — same schema, different commit semantics.
- Migration order: do harness fixes first (so re-runs of the 14 are clean), or
  migrate existing data first (rename + filter) and re-run during/after?

## Related project context

- CLAUDE.md mistake "Scoring-contract bugs ship without scorer unit tests" —
  any change to attempt termination semantics should land with tests covering
  (right/wrong verdict) × (clean/dirty exit) × (timeout/crash/no-output).
  Now extended: also × (provider-error / harness-error / model-no-submit /
  early-perfect).
- vb_version bump expected when this lands (will be ≥ 0.0.12). Old data in
  `events.jsonl` (renamed from current `results.jsonl`) is pre-0.0.12 and
  pre-status-tracking; new data has `attempt_status` and lives in fresh
  `results.jsonl`.
