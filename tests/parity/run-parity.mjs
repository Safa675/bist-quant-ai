import { readFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const FIXTURE_DIR = path.resolve(__dirname, "..", "contracts");

function readJson(fileName) {
  const filePath = path.join(FIXTURE_DIR, fileName);
  const raw = readFileSync(filePath, "utf-8");
  return JSON.parse(raw);
}

function fail(message) {
  throw new Error(message);
}

function assertObject(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    fail(`${label} must be an object.`);
  }
}

function assertArray(value, label) {
  if (!Array.isArray(value)) {
    fail(`${label} must be an array.`);
  }
}

function assertNonEmptyString(value, label) {
  if (typeof value !== "string" || !value.trim()) {
    fail(`${label} must be a non-empty string.`);
  }
}

function assertNumber(value, label) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    fail(`${label} must be a finite number.`);
  }
}

function assertBoolean(value, label) {
  if (typeof value !== "boolean") {
    fail(`${label} must be a boolean.`);
  }
}

function assertNullableNumber(value, label) {
  if (value !== null && (typeof value !== "number" || !Number.isFinite(value))) {
    fail(`${label} must be a finite number or null.`);
  }
}

function validateEnvelope(envelope, label) {
  assertObject(envelope, label);
  assertNonEmptyString(envelope.run_id, `${label}.run_id`);
  assertObject(envelope.meta, `${label}.meta`);

  if (!("result" in envelope)) {
    fail(`${label}.result is required.`);
  }

  if (!("error" in envelope)) {
    fail(`${label}.error is required.`);
  }

  if (envelope.error !== null) {
    assertObject(envelope.error, `${label}.error`);
    assertNonEmptyString(envelope.error.code, `${label}.error.code`);
    assertNonEmptyString(envelope.error.message, `${label}.error.message`);
  }
}

function validateFactorBacktestFixture() {
  const envelope = readJson("factor-lab.fixture.json");
  validateEnvelope(envelope, "factor_fixture");
  assertObject(envelope.result, "factor_fixture.result");

  const result = envelope.result;
  assertObject(result.meta, "factor_fixture.result.meta");
  assertNonEmptyString(result.meta.mode, "factor_fixture.result.meta.mode");
  assertNonEmptyString(result.meta.start_date, "factor_fixture.result.meta.start_date");
  assertNonEmptyString(result.meta.end_date, "factor_fixture.result.meta.end_date");
  assertNumber(result.meta.top_n, "factor_fixture.result.meta.top_n");
  assertArray(result.meta.factors, "factor_fixture.result.meta.factors");

  assertObject(result.metrics, "factor_fixture.result.metrics");
  assertNumber(result.metrics.cagr, "factor_fixture.result.metrics.cagr");
  assertNumber(result.metrics.sharpe, "factor_fixture.result.metrics.sharpe");
  assertNumber(result.metrics.sortino, "factor_fixture.result.metrics.sortino");
  assertNumber(result.metrics.max_dd, "factor_fixture.result.metrics.max_dd");
  assertNumber(result.metrics.total_return, "factor_fixture.result.metrics.total_return");
  assertNumber(result.metrics.win_rate, "factor_fixture.result.metrics.win_rate");
  assertNullableNumber(result.metrics.beta, "factor_fixture.result.metrics.beta");
  assertNumber(result.metrics.rebalance_count, "factor_fixture.result.metrics.rebalance_count");
  assertNumber(result.metrics.trade_count, "factor_fixture.result.metrics.trade_count");

  assertArray(result.composite_top, "factor_fixture.result.composite_top");
  assertObject(result.factor_top_symbols, "factor_fixture.result.factor_top_symbols");
  assertArray(result.current_holdings, "factor_fixture.result.current_holdings");
  assertArray(result.equity_curve, "factor_fixture.result.equity_curve");
  assertArray(result.benchmark_curve, "factor_fixture.result.benchmark_curve");

  if (result.analytics_v2 !== undefined) {
    assertObject(result.analytics_v2, "factor_fixture.result.analytics_v2");
  }
}

function validateSignalBacktestFixture() {
  const envelope = readJson("signal-backtest.fixture.json");
  validateEnvelope(envelope, "signal_fixture");
  assertObject(envelope.result, "signal_fixture.result");

  const result = envelope.result;
  assertObject(result.meta, "signal_fixture.result.meta");
  assertNonEmptyString(result.meta.mode, "signal_fixture.result.meta.mode");
  assertNonEmptyString(result.meta.universe, "signal_fixture.result.meta.universe");
  assertNonEmptyString(result.meta.period, "signal_fixture.result.meta.period");
  assertNonEmptyString(result.meta.interval, "signal_fixture.result.meta.interval");
  assertArray(result.meta.indicators, "signal_fixture.result.meta.indicators");
  assertNumber(result.meta.buy_threshold, "signal_fixture.result.meta.buy_threshold");
  assertNumber(result.meta.sell_threshold, "signal_fixture.result.meta.sell_threshold");
  assertNumber(result.meta.max_positions, "signal_fixture.result.meta.max_positions");

  assertObject(result.metrics, "signal_fixture.result.metrics");
  assertNumber(result.metrics.cagr, "signal_fixture.result.metrics.cagr");
  assertNumber(result.metrics.sharpe, "signal_fixture.result.metrics.sharpe");
  assertNumber(result.metrics.max_dd, "signal_fixture.result.metrics.max_dd");
  assertNumber(result.metrics.ytd, "signal_fixture.result.metrics.ytd");
  assertNumber(result.metrics.total_return, "signal_fixture.result.metrics.total_return");
  assertNumber(result.metrics.volatility, "signal_fixture.result.metrics.volatility");
  assertNumber(result.metrics.win_rate, "signal_fixture.result.metrics.win_rate");
  assertNullableNumber(result.metrics.beta, "signal_fixture.result.metrics.beta");
  assertNonEmptyString(result.metrics.last_rebalance, "signal_fixture.result.metrics.last_rebalance");

  assertArray(result.signals, "signal_fixture.result.signals");
  assertArray(result.indicator_summaries, "signal_fixture.result.indicator_summaries");
  assertArray(result.current_holdings, "signal_fixture.result.current_holdings");
  assertArray(result.equity_curve, "signal_fixture.result.equity_curve");
  assertArray(result.benchmark_curve, "signal_fixture.result.benchmark_curve");

  if (result.analytics_v2 !== undefined) {
    assertObject(result.analytics_v2, "signal_fixture.result.analytics_v2");
  }
}

function validateStockFilterFixture() {
  const envelope = readJson("stock-filter.fixture.json");
  validateEnvelope(envelope, "stock_filter_fixture");
  assertObject(envelope.result, "stock_filter_fixture.result");

  const result = envelope.result;
  assertObject(result.meta, "stock_filter_fixture.result.meta");
  assertNonEmptyString(result.meta.as_of, "stock_filter_fixture.result.meta.as_of");
  assertNumber(result.meta.execution_ms, "stock_filter_fixture.result.meta.execution_ms");
  assertNumber(result.meta.total_matches, "stock_filter_fixture.result.meta.total_matches");
  assertNumber(result.meta.returned_rows, "stock_filter_fixture.result.meta.returned_rows");
  assertNonEmptyString(result.meta.sort_by, "stock_filter_fixture.result.meta.sort_by");
  assertBoolean(result.meta.sort_desc, "stock_filter_fixture.result.meta.sort_desc");

  assertArray(result.columns, "stock_filter_fixture.result.columns");
  assertArray(result.rows, "stock_filter_fixture.result.rows");
  assertArray(result.applied_filters, "stock_filter_fixture.result.applied_filters");

  if (result.applied_percentile_filters !== undefined) {
    assertArray(result.applied_percentile_filters, "stock_filter_fixture.result.applied_percentile_filters");
  }
}

function main() {
  validateFactorBacktestFixture();
  validateSignalBacktestFixture();
  validateStockFilterFixture();
  console.info("Parity fixtures validated successfully.");
}

try {
  main();
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`Parity validation failed: ${message}`);
  process.exitCode = 1;
}
