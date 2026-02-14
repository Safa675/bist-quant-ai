"""Unified multi-asset adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import pandas as pd

from .crypto_client import CryptoClient
from .fund_analyzer import TEFASAnalyzer
from .fx_commodities_client import FXCommoditiesClient
from .portfolio_analytics import PortfolioAnalytics
from .us_stock_client import USStockClient


@dataclass
class MultiAssetData:
    symbol: str
    asset_type: str
    quote: pd.Series
    history: pd.DataFrame
    technical_indicators: pd.DataFrame
    fundamentals: Optional[Dict] = None
    allocation: Optional[pd.DataFrame] = None


class MultiAssetAdapter:
    """Adapter for Turkish stocks, TEFAS funds, crypto, US stocks, and FX."""

    def __init__(self):
        self.analytics = PortfolioAnalytics()
        self.fund_analyzer = TEFASAnalyzer()
        self.crypto_client = CryptoClient()
        self.us_stock_client = USStockClient()
        self.fx_client = FXCommoditiesClient()

    async def get_multi_asset_data(self, symbol: str, asset_type: str = "stock") -> MultiAssetData:
        """Fetch comprehensive data for a given asset type."""
        kind = asset_type.lower().strip()
        if kind == "stock":
            return await self._get_stock_data(symbol)
        if kind == "fund":
            return await self._get_fund_data(symbol)
        if kind == "crypto":
            return await self._get_crypto_data(symbol)
        if kind == "us_stock":
            return await self._get_us_stock_data(symbol)
        if kind == "fx":
            return await self._get_fx_data(symbol)
        raise ValueError(f"Unsupported asset type: {asset_type}")

    async def _get_stock_data(self, symbol: str) -> MultiAssetData:
        """Get data for Turkish stock via borsapy."""
        import borsapy as bp

        ticker = bp.Ticker(symbol)
        quote_raw = getattr(ticker, "info", {})
        quote = quote_raw if isinstance(quote_raw, pd.Series) else pd.Series(quote_raw)

        history = ticker.history(period="1mo")
        technical_indicators = self._calculate_technical_indicators(history)

        fundamentals = {
            "balance_sheet": getattr(ticker, "balance_sheet", pd.DataFrame()),
            "income_stmt": getattr(ticker, "income_stmt", pd.DataFrame()),
            "cash_flow": getattr(ticker, "cash_flow", pd.DataFrame()),
            "financial_ratios": getattr(ticker, "financial_ratios", pd.DataFrame()),
        }

        return MultiAssetData(
            symbol=symbol,
            asset_type="stock",
            quote=quote,
            history=history,
            technical_indicators=technical_indicators,
            fundamentals=fundamentals,
        )

    async def _get_fund_data(self, symbol: str) -> MultiAssetData:
        """Get data for TEFAS fund."""
        fund_df = await self.fund_analyzer.get_fund_data(fund_code=symbol)
        if fund_df.empty:
            raise ValueError(f"No fund data found for {symbol}")

        fund_info = fund_df.iloc[0].to_dict()
        quote = pd.Series(fund_info)

        allocation = await self.fund_analyzer.get_fund_allocation(symbol)
        history = await self.fund_analyzer.get_fund_history(symbol, period="1y")
        technical_indicators = self._calculate_technical_indicators(history)

        return MultiAssetData(
            symbol=symbol,
            asset_type="fund",
            quote=quote,
            history=history,
            technical_indicators=technical_indicators,
            fundamentals=fund_info,
            allocation=allocation,
        )

    async def _get_crypto_data(self, symbol: str) -> MultiAssetData:
        """Get data for cryptocurrency."""
        markets = await self.crypto_client.get_crypto_markets(exchange="btcturk")
        if markets.empty or "symbol" not in markets.columns:
            raise ValueError("No crypto market data found")

        normalized_symbol = symbol.upper()
        row = markets[markets["symbol"].astype(str).str.upper() == normalized_symbol]
        if row.empty and "/" not in normalized_symbol:
            row = markets[markets["symbol"].astype(str).str.upper().str.startswith(normalized_symbol)]

        if row.empty:
            raise ValueError(f"No crypto data found for {symbol}")

        quote = pd.Series(row.iloc[0].to_dict())
        history = await self.crypto_client.get_crypto_history(symbol, exchange="btcturk", period="1mo", interval="1d")
        technical_indicators = self._calculate_technical_indicators(history)

        return MultiAssetData(
            symbol=symbol,
            asset_type="crypto",
            quote=quote,
            history=history,
            technical_indicators=technical_indicators,
        )

    async def _get_us_stock_data(self, symbol: str) -> MultiAssetData:
        """Get data for US stock."""
        profile = await self.us_stock_client.get_us_stock_info(symbol)

        quotes = await self.us_stock_client.get_us_stock_quotes([symbol])
        quote = quotes.iloc[0] if not quotes.empty else pd.Series(dtype=float)

        history = await self.us_stock_client.get_us_stock_history(symbol)
        technical_indicators = self._calculate_technical_indicators(history)

        return MultiAssetData(
            symbol=symbol,
            asset_type="us_stock",
            quote=quote,
            history=history,
            technical_indicators=technical_indicators,
            fundamentals=profile.__dict__,
        )

    async def _get_fx_data(self, symbol: str) -> MultiAssetData:
        """Get data for FX pair."""
        fx_df = await self.fx_client.get_fx_rates([symbol])
        if fx_df.empty:
            raise ValueError(f"No FX data found for {symbol}")

        row = fx_df.iloc[0].to_dict()
        quote = pd.Series(row)

        pair = row.get("pair", symbol)
        history = await self.fx_client.get_fx_history(str(pair))
        technical_indicators = self._calculate_technical_indicators(history)

        return MultiAssetData(
            symbol=symbol,
            asset_type="fx",
            quote=quote,
            history=history,
            technical_indicators=technical_indicators,
        )

    def _calculate_technical_indicators(self, history: pd.DataFrame) -> pd.DataFrame:
        """Calculate SMA/RSI/MACD for any history table with close data."""
        if history is None or history.empty:
            return pd.DataFrame()

        close_col = self._pick_close_col(history)
        if close_col is None:
            return pd.DataFrame()

        df = history.copy()
        close = pd.to_numeric(df[close_col], errors="coerce")
        if close.dropna().empty:
            return pd.DataFrame()

        out = pd.DataFrame(index=df.index)
        out["SMA_20"] = close.rolling(window=20, min_periods=20).mean()
        out["SMA_50"] = close.rolling(window=50, min_periods=50).mean()

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window=14, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=14).mean()
        rs = gain / loss.replace(0, pd.NA)
        out["RSI"] = 100 - (100 / (1 + rs))

        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        out["MACD"] = exp1 - exp2
        out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()

        return out.dropna(how="all")

    async def get_multi_asset_portfolio_analysis(
        self,
        holdings: Dict[str, Dict[str, Union[float, int, str]]],
    ) -> Dict:
        """Analyze a multi-asset portfolio."""
        metrics = await self.analytics.calculate_multi_asset_portfolio_metrics(holdings)
        diversification = await self.analytics.get_diversification_metrics(holdings)

        asset_performances: Dict[str, Dict[str, float] | Dict[str, str]] = {}
        for symbol, info in holdings.items():
            asset_type = str(info.get("asset_type", "stock"))
            try:
                asset_data = await self.get_multi_asset_data(symbol, asset_type)
                close_col = self._pick_close_col(asset_data.history)
                if close_col is None:
                    asset_performances[symbol] = {"error": "No close column in history"}
                    continue

                close = pd.to_numeric(asset_data.history[close_col], errors="coerce")
                recent_returns = close.pct_change().dropna().tail(30)
                if recent_returns.empty:
                    asset_performances[symbol] = {"error": "Insufficient history"}
                    continue

                avg_return = float(recent_returns.mean())
                volatility = float(recent_returns.std())
                asset_performances[symbol] = {
                    "avg_return": avg_return,
                    "volatility": volatility,
                    "sharpe": float(avg_return / volatility) if volatility != 0 else 0.0,
                }
            except Exception as exc:
                asset_performances[symbol] = {"error": str(exc)}

        return {
            "portfolio_metrics": metrics,
            "diversification": diversification,
            "asset_performances": asset_performances,
        }

    async def close(self) -> None:
        """Close async clients."""
        await self.analytics.close_clients()

    @staticmethod
    def _pick_close_col(df: pd.DataFrame) -> Optional[str]:
        if df is None or df.empty:
            return None
        for col in ("Close", "close", "Adj Close", "adj_close", "nav", "price"):
            if col in df.columns:
                return col
        return None
