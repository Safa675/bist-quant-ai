# Quant AI Web Reimagination - Phase 0 + Phase 1 (Foundation)

## Goal
Rebuild the current script-first BIST research library as a web-native platform where research, screening, signal construction, backtesting, publishing, and agent workflows run from one consistent application surface.

This phase set focuses on:
- Phase 0: parity baseline and contract hardening
- Phase 1: execution and data foundation for scalable web workflows

## Current Baseline (as of 2026-02-13)
- Next.js routes call Python scripts directly via `spawn`:
  - `src/lib/server/factorLabPython.ts`
  - `src/lib/server/signalConstructionPython.ts`
  - `src/lib/server/stockFilterPython.ts`
- Factor/signal engines live in:
  - `dashboard/factor_lab_api.py`
  - `dashboard/signal_construction_api.py`
  - `dashboard/stock_filter_api.py`
  - `Models/portfolio_engine.py`
- Dashboard snapshots are generated to JSON via:
  - `dashboard/generate_dashboard_data.py`
  - `src/lib/server/dashboardData.ts`
- User-published signals/presets are JSON-file backed:
  - `src/lib/server/signalStore.ts`

## Phase 0 - Parity Baseline

### Objective
Guarantee deterministic parity between script outputs and web outputs, with explicit API/data contracts, before deeper architecture changes.

### Deliverables
1. Canonical capability inventory with parity matrix
- Map every script capability to web route/UI surface.
- Confirm whether each capability is:
  - already parity-complete
  - partially exposed
  - missing

2. JSON contract freeze for all current engines
- Freeze request/response schemas for:
  - `/api/factor-lab`
  - `/api/signal-construction`
  - `/api/signal-construction/backtest`
  - `/api/stock-filter`
  - `/api/dashboard`
- Add contract fixtures from real payloads.

3. Reproducibility baseline harness
- Add golden test vectors for representative runs:
  - 1 factor-lab run
  - 1 technical signal backtest
  - 1 stock filter run (percentile mode)
- Define tolerance bands for metric drift.

4. Data/metadata inventory
- Document all runtime data dependencies:
  - `data/`
  - `Regime Filter/`
  - generated artifacts in `public/data/`
- Separate immutable inputs vs generated outputs vs user state.

5. Observability minimum baseline
- Standard request/run IDs across TS and Python boundaries.
- Standard error shape (`code`, `message`, `details`, `run_id`).

### Phase 0 Exit Criteria
- 100% of current user-visible features mapped in parity matrix.
- Engine contracts documented and fixture-backed.
- Golden-run harness passes with accepted tolerances.
- No blind spots in data dependency inventory.

## Phase 1 - Foundation

### Objective
Introduce platform primitives that let all workflows behave consistently: typed contracts, async runs, artifact storage, and reusable execution services.

### Foundation Pillars
1. Unified domain model
- Introduce shared entities:
  - `SignalDefinition`
  - `RunRequest`
  - `RunResult`
  - `BacktestMetrics`
  - `ArtifactRef`
  - `ScreenerRequest`
  - `AgentToolSession`

2. Async execution layer
- Move heavy compute paths from blocking HTTP requests to run-oriented jobs.
- Introduce run lifecycle states:
  - `queued`
  - `running`
  - `succeeded`
  - `failed`
  - `cancelled`

3. Artifact and metadata persistence
- Keep JSON store for compatibility in Phase 1, but add structured run/artifact storage.
- Persist:
  - run inputs
  - normalized outputs
  - logs
  - derived chart payloads

4. Unified catalog service
- Build one canonical signal/factor catalog from `Models/configs` and/or engine introspection.
- Ensure Signal Lab, Factor Lab, and downstream services consume the same catalog contract.

5. Agent orchestration baseline hardening
- Centralize MCP tooling policy and per-agent tool budgets.
- Enable controlled override of tool-call limits from config.
- Reuse common tool context across all agents.

### Phase 1 Scope (In)
- Backend contracts and run orchestration
- Storage model for runs and artifacts
- Detailed results payload v2 for backtests
- Shared catalog + shared validation
- Agent tool-orchestration reliability controls

### Phase 1 Scope (Out)
- Full UX redesign of all pages
- Live brokerage execution
- Multi-market expansion beyond BIST
- Full multi-tenant auth/permissions

### Phase 1 Exit Criteria
- Heavy workflows run through run IDs and status polling.
- Backtest/screener results are persisted and reloadable.
- One catalog contract powers all signal/factor selectors.
- Agents use centralized tool policy and no hardcoded silent limits.
- Dashboard and Signal Lab continue to function without regression.

## Parity Matrix Skeleton (Phase 0 Working Artifact)
| Capability | Current Engine | Current Route/UI | Parity Target |
|---|---|---|---|
| Model factor backtest | `dashboard/factor_lab_api.py` | `/api/factor-lab`, `src/app/factor-lab/page.tsx` | Contract + fixtures + run artifact |
| Technical signal build | `dashboard/signal_construction_api.py` | `/api/signal-construction`, `src/app/signal-construction/page.tsx` | Contract + fixtures + run artifact |
| Technical backtest | `dashboard/signal_construction_api.py` | `/api/signal-construction/backtest` | Contract + fixtures + run artifact |
| Stock screener | `dashboard/stock_filter_api.py` | `/api/stock-filter`, `src/app/stock-filter/page.tsx` | Contract + fixtures + run artifact |
| Dashboard aggregation | `dashboard/generate_dashboard_data.py` | `/api/dashboard`, `src/app/dashboard/page.tsx` | Snapshot parity + refresh telemetry |
| Signal publishing | `src/lib/server/signalStore.ts` | `/api/signal-construction/publish` | Structured metadata + compatibility |

## Risk Register
1. Runtime limits on serverless for heavy jobs.
- Mitigation: async queue + chunked artifacts + progressive result writes.

2. Drift between Python outputs and TS assumptions.
- Mitigation: contract fixtures and schema validation at boundaries.

3. Fragile file-based state in ephemeral environments.
- Mitigation: run/artifact store abstraction behind adapters.

4. Tool-call exhaustion in agent loops.
- Mitigation: per-agent budgets, stop conditions, compact tool plans.

## Milestone Sequence
1. Phase 0: parity matrix + contracts + harness
2. Phase 1A: domain contracts + run model
3. Phase 1B: async execution + persistence
4. Phase 1C: detailed results + catalog unification + agent hardening
