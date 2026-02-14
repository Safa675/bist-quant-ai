import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Models.common.crypto_client import CryptoClient
from Models.common.fund_analyzer import FundCategory, TEFASAnalyzer
from Models.common.fx_commodities_client import FXCommoditiesClient
from Models.common.us_stock_client import USStockClient


def test_tefas_fund_integration() -> None:
    """TEFAS analyzer should parse MCP results and expose utility methods."""

    async def _run() -> None:
        analyzer = TEFASAnalyzer()
        analyzer._call_mcp_async = AsyncMock(
            side_effect=[
                {
                    "data": [
                        {
                            "fund_code": "AAK",
                            "name": "AAK Equity Fund",
                            "category": "equity",
                            "nav": 1.24,
                            "return_1y": 42.0,
                            "expense_ratio": 1.2,
                            "aum": 10_000_000,
                        },
                        {
                            "fund_code": "IPB",
                            "name": "IPB Equity Fund",
                            "category": "equity",
                            "nav": 2.02,
                            "return_1y": 35.5,
                            "expense_ratio": 1.1,
                            "aum": 8_000_000,
                        },
                    ]
                },
                {
                    "data": [
                        {
                            "fund_code": "AAK",
                            "name": "AAK Equity Fund",
                            "category": "equity",
                            "nav": 1.24,
                            "return_1y": 42.0,
                            "expense_ratio": 1.2,
                            "aum": 10_000_000,
                        }
                    ]
                },
                {
                    "data": [
                        {
                            "fund_code": "AAK",
                            "name": "AAK Equity Fund",
                            "category": "equity",
                            "nav": 1.24,
                            "return_1y": 42.0,
                            "expense_ratio": 1.2,
                            "aum": 10_000_000,
                        },
                        {
                            "fund_code": "IPB",
                            "name": "IPB Equity Fund",
                            "category": "equity",
                            "nav": 2.02,
                            "return_1y": 35.5,
                            "expense_ratio": 1.1,
                            "aum": 8_000_000,
                        },
                    ]
                },
            ]
        )

        try:
            funds_df = await analyzer.get_fund_data(limit=5)
            assert isinstance(funds_df, pd.DataFrame)
            assert len(funds_df) <= 5
            assert not funds_df.empty

            specific_fund = await analyzer.get_fund_data(fund_code="AAK")
            assert isinstance(specific_fund, pd.DataFrame)
            assert len(specific_fund) >= 1
            assert specific_fund.iloc[0]["fund_code"] == "AAK"

            top_equity = await analyzer.get_top_performers(FundCategory.EQUITY, limit=2)
            assert isinstance(top_equity, pd.DataFrame)
            assert len(top_equity) == 2
            assert top_equity.iloc[0]["return_1y"] >= top_equity.iloc[1]["return_1y"]
        finally:
            await analyzer.close()

    asyncio.run(_run())


def test_crypto_integration() -> None:
    """Crypto client should parse market, history and ranking data."""

    async def fake_call(_tool: str, params: dict):
        if params.get("history"):
            return {
                "data": [
                    {"date": "2026-01-01", "open": 100.0, "high": 110.0, "low": 95.0, "close": 108.0, "volume": 1000},
                    {"date": "2026-01-02", "open": 108.0, "high": 112.0, "low": 107.0, "close": 111.0, "volume": 1100},
                ]
            }
        if params.get("action") == "pairs":
            return {"data": {"pairs": ["BTC/TRY", "ETH/TRY"]}}
        return {
            "data": [
                {"symbol": "BTC/TRY", "price": 100000.0, "change_24h": 2.0, "volume_24h": 10_000_000, "market_cap": 1_000_000_000, "source": "btcturk"},
                {"symbol": "ETH/TRY", "price": 5000.0, "change_24h": -1.0, "volume_24h": 5_000_000, "market_cap": 500_000_000, "source": "btcturk"},
            ]
        }

    async def _run() -> None:
        client = CryptoClient()
        client._call_mcp_async = AsyncMock(side_effect=fake_call)

        try:
            crypto_df = await client.get_crypto_markets(exchange="btcturk")
            assert isinstance(crypto_df, pd.DataFrame)
            assert not crypto_df.empty

            first_symbol = crypto_df.iloc[0]["symbol"]
            history = await client.get_crypto_history(first_symbol, "btcturk")
            assert isinstance(history, pd.DataFrame)
            assert not history.empty
            assert {"open", "high", "low", "close"}.issubset(history.columns)

            top = await client.get_top_gainers_losers("btcturk", count=1)
            assert "gainers" in top and "losers" in top
            assert len(top["gainers"]) == 1
            assert len(top["losers"]) == 1

            pairs = await client.get_crypto_pairs("btcturk")
            assert "BTC/TRY" in pairs
        finally:
            await client.close()

    asyncio.run(_run())


def test_us_stock_integration() -> None:
    """US stock client should support search/profile/quote/history/sector endpoints."""

    async def fake_call(tool: str, params: dict):
        if tool == "search_symbol":
            return {"data": [{"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"}]}
        if tool == "get_profile":
            return {
                "data": {
                    "name": "Apple Inc.",
                    "sector": "Technology",
                    "industry": "Consumer Electronics",
                    "market_cap": 3_000_000_000_000,
                    "pe_ratio": 29.1,
                    "pb_ratio": 45.0,
                    "dividend_yield": 0.5,
                    "eps": 6.1,
                    "revenue": 380_000_000_000,
                    "profit_margin": 25.0,
                }
            }
        if tool == "get_quick_info":
            return {
                "data": {
                    "price": 210.0,
                    "change": 1.5,
                    "change_percent": 0.72,
                    "volume": 45_000_000,
                    "high": 212.0,
                    "low": 207.0,
                    "open": 208.0,
                }
            }
        if tool == "get_historical_data":
            return {
                "data": [
                    {"date": "2026-01-01", "open": 200.0, "high": 205.0, "low": 198.0, "close": 202.0, "volume": 42_000_000},
                    {"date": "2026-01-02", "open": 202.0, "high": 206.0, "low": 201.0, "close": 205.0, "volume": 38_000_000},
                ]
            }
        if tool == "get_sector_comparison":
            return {
                "data": [
                    {"sector": "Technology", "performance": 1.2},
                    {"sector": "Healthcare", "performance": 0.6},
                ]
            }
        return {"data": {}}

    async def _run() -> None:
        client = USStockClient()
        client._call_mcp_async = AsyncMock(side_effect=fake_call)

        try:
            search_results = await client.search_us_stocks("AAPL")
            assert isinstance(search_results, pd.DataFrame)
            assert not search_results.empty

            first_symbol = search_results.iloc[0]["symbol"]
            stock_info = await client.get_us_stock_info(first_symbol)
            assert stock_info.symbol == first_symbol
            assert stock_info.name == "Apple Inc."

            quotes = await client.get_us_stock_quotes([first_symbol])
            assert isinstance(quotes, pd.DataFrame)
            assert len(quotes) >= 1
            assert float(quotes.iloc[0]["price"]) > 0

            history = await client.get_us_stock_history(first_symbol)
            assert isinstance(history, pd.DataFrame)
            assert not history.empty

            sectors = await client.get_us_sector_performance()
            assert isinstance(sectors, pd.DataFrame)
            assert not sectors.empty
        finally:
            await client.close()

    asyncio.run(_run())


def test_fx_integration() -> None:
    """FX and commodity client should return normalized dataframes."""

    async def fake_call(_tool: str, params: dict):
        if params.get("history"):
            return {
                "data": [
                    {"date": "2026-01-01", "open": 30.0, "high": 30.2, "low": 29.8, "close": 30.1, "volume": 1000},
                    {"date": "2026-01-02", "open": 30.1, "high": 30.3, "low": 30.0, "close": 30.2, "volume": 1200},
                ]
            }
        if params.get("asset_type") == "commodity":
            return {
                "data": [
                    {"commodity": "gold", "type": "metal", "unit": "oz", "price": 2900.0, "change_percent": 0.5},
                    {"commodity": "oil", "type": "oil", "unit": "barrel", "price": 75.0, "change_percent": -0.2},
                ]
            }
        return {
            "data": [
                {"pair": "USD/TRY", "bid": 30.0, "ask": 30.02, "last_price": 30.01, "change_percent": 0.4},
                {"pair": "EUR/USD", "bid": 1.08, "ask": 1.081, "last_price": 1.0805, "change_percent": -0.1},
            ]
        }

    async def _run() -> None:
        client = FXCommoditiesClient()
        client._call_mcp_async = AsyncMock(side_effect=fake_call)

        try:
            fx_df = await client.get_fx_rates(["USDTRY", "EURUSD"])
            assert isinstance(fx_df, pd.DataFrame)
            assert not fx_df.empty
            assert {"pair", "last_price"}.issubset(fx_df.columns)

            first_pair = fx_df.iloc[0]["pair"]
            history = await client.get_fx_history(first_pair)
            assert isinstance(history, pd.DataFrame)
            assert not history.empty

            commodities = await client.get_commodity_prices(["gold", "oil"])
            assert isinstance(commodities, pd.DataFrame)
            assert not commodities.empty

            metals = await client.get_precious_metals()
            assert isinstance(metals, pd.DataFrame)

            corr = await client.get_correlation_matrix(["USDTRY", "EURUSD"])
            # Correlation matrix may be empty with very short series, but should be a DataFrame.
            assert isinstance(corr, pd.DataFrame)
        finally:
            await client.close()

    asyncio.run(_run())
