# Phase 1 Sprint Board - Foundation

## Cadence
- Sprint length: 1 week
- Total foundation window: 4 sprints
- Parallel lanes: Platform (TS), Engine (Python), UI Integration

## Ticket Board
| ID | Sprint | Priority | Ticket | Dependencies | Definition of Done |
|---|---|---|---|---|---|
| FND-001 | S1 | P0 | Define shared run/contracts schema (TS) | None | Typed request/response models exist and are used by API handlers. |
| FND-002 | S1 | P0 | Normalize Python engine payload schemas | FND-001 | Python outputs conform to one normalized envelope with `run_id`, `meta`, `result`, `error`. |
| FND-003 | S1 | P0 | Implement run/artifact store abstraction | FND-001 | Run metadata and artifact index are persisted behind one storage interface. |
| FND-004 | S1 | P1 | Add run lifecycle API (`create`, `get`, `list`) | FND-001, FND-003 | New run endpoints work with status transitions and error payloads. |
| FND-005 | S2 | P0 | Build async execution worker path for heavy jobs | FND-003, FND-004 | Factor/signal backtests run as jobs with polling, not blocking request path. |
| FND-006 | S2 | P0 | Add unified catalog service for factors/signals | FND-001 | Signal Lab and Factor Lab consume same catalog contract. |
| FND-007 | S2 | P1 | Add detailed backtest analytics payload v2 | FND-002 | Backtest response includes yearly/risk/turnover diagnostics and artifact refs. |
| FND-008 | S3 | P0 | Integrate run APIs into Signal Lab/Factor Lab UI | FND-004, FND-005, FND-006 | UI shows run status, errors, and persisted results with reload support. |
| FND-009 | S3 | P1 | Dashboard data adapter for persisted run artifacts | FND-003, FND-007 | Dashboard reads latest valid run artifacts without breaking existing data. |
| FND-010 | S3 | P0 | Agent orchestration policy unification (tool budgets + shared MCP) | None | All agents use centralized policy; max tool calls configurable per agent. |
| FND-011 | S4 | P0 | Parity/golden-run test harness | FND-002, FND-005, FND-007 | Golden vectors pass in CI for factor, signal, screener payloads. |
| FND-012 | S4 | P1 | Observability and run diagnostics | FND-004, FND-005 | Structured logs and trace IDs connect UI request -> run -> python engine. |

## File-Level Implementation Map

### FND-001 - Shared Contracts (TS)
Create:
- `src/lib/contracts/run.ts`
- `src/lib/contracts/factor.ts`
- `src/lib/contracts/signal.ts`
- `src/lib/contracts/screener.ts`

Modify:
- `src/app/api/factor-lab/route.ts`
- `src/app/api/signal-construction/route.ts`
- `src/app/api/signal-construction/backtest/route.ts`
- `src/app/api/stock-filter/route.ts`

### FND-002 - Python Envelope Normalization
Modify:
- `dashboard/factor_lab_api.py`
- `dashboard/signal_construction_api.py`
- `dashboard/stock_filter_api.py`
- `api/index.py`

Create:
- `dashboard/common_response.py`

### FND-003 - Run/Artifact Store
Create:
- `src/lib/server/runStore.ts`
- `src/lib/server/artifactStore.ts`
- `data/run_store.json` (bootstrap file for local dev)

Modify:
- `src/lib/server/signalStore.ts`

### FND-004 - Run Lifecycle API
Create:
- `src/app/api/runs/route.ts`
- `src/app/api/runs/[runId]/route.ts`

Modify:
- `src/app/api/factor-lab/route.ts`
- `src/app/api/signal-construction/backtest/route.ts`

### FND-005 - Async Job Execution
Create:
- `src/lib/server/jobQueue.ts`
- `src/lib/server/pythonRunner.ts`
- `src/lib/server/runExecutor.ts`

Modify:
- `src/lib/server/factorLabPython.ts`
- `src/lib/server/signalConstructionPython.ts`

### FND-006 - Unified Catalog Service
Create:
- `src/lib/server/catalogService.ts`

Modify:
- `dashboard/factor_lab_api.py`
- `src/app/api/factor-lab/route.ts`
- `src/app/signal-lab/page.tsx`
- `src/app/factor-lab/page.tsx`

### FND-007 - Detailed Backtest Payload v2
Modify:
- `dashboard/factor_lab_api.py`
- `dashboard/signal_construction_api.py`
- `Models/portfolio_engine.py`

Create:
- `dashboard/analytics_payloads.py`

### FND-008 - UI Run-State Integration
Modify:
- `src/app/factor-lab/page.tsx`
- `src/app/signal-construction/page.tsx`
- `src/app/signal-lab/page.tsx`

Create:
- `src/components/RunStatusPanel.tsx`
- `src/components/RunHistoryList.tsx`

### FND-009 - Dashboard Artifact Adapter
Modify:
- `src/lib/server/dashboardData.ts`
- `dashboard/generate_dashboard_data.py`
- `src/app/api/dashboard/route.ts`

### FND-010 - Agent Tool Policy Unification
Create:
- `src/lib/agents/policy.ts`

Modify:
- `src/lib/agents/tool-orchestrator.ts`
- `src/app/api/agents/_handler.ts`
- `src/app/api/agents/research/route.ts`
- `src/lib/agents/borsa-mcp-client.ts`

### FND-011 - Golden-Run Harness
Create:
- `tests/contracts/factor-lab.fixture.json`
- `tests/contracts/signal-backtest.fixture.json`
- `tests/contracts/stock-filter.fixture.json`
- `tests/parity/run-parity.spec.ts`

Modify:
- `package.json` (add parity test script)

### FND-012 - Observability
Create:
- `src/lib/server/telemetry.ts`

Modify:
- `src/lib/agents/logging.ts`
- `src/lib/server/factorLabPython.ts`
- `src/lib/server/signalConstructionPython.ts`
- `src/lib/server/stockFilterPython.ts`
- `src/app/api/*` relevant run endpoints

## Sprint-by-Sprint Outcome Targets

### Sprint 1 Target
- Contracts, schema envelope, and run storage primitives are merged.
- No feature behavior change required yet.

### Sprint 2 Target
- Heavy compute paths support async run flow.
- Catalog is unified and consumed by UI selectors.

### Sprint 3 Target
- Signal Lab and Factor Lab operate on run IDs and persisted results.
- Agent orchestration uses centralized tool policy.

### Sprint 4 Target
- Parity test harness is green.
- Observability gives end-to-end run traceability.

## Risks and Trigger Conditions
| Risk | Trigger | Mitigation |
|---|---|---|
| Vercel runtime saturation | Run durations near timeout or memory limits | Move compute to queued worker process, stream artifacts incrementally. |
| Schema drift between TS/Python | UI parse failures after engine changes | Contract fixtures + versioned envelope. |
| Local JSON store contention | Concurrent writes or corruption | Add atomic write strategy and optional SQLite adapter path. |
| Agent tool exhaustion persists | Frequent `max tool calls` responses | Per-agent policy file + planner step to batch/reduce calls. |
