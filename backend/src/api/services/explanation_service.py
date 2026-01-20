"""Explanation service for generating LLM-powered trading explanations.

This service provides:
- GPT-4o-mini powered explanations of trading recommendations
- Intelligent caching that only regenerates when values change significantly
- Plain English explanations of technical and sentiment analysis
"""

import hashlib
import logging
import os
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ExplanationService:
    """Service for generating AI explanations of trading recommendations.

    Uses OpenAI GPT-4o-mini to generate plain English explanations
    of the trading dashboard values and recommendations.
    """

    # Cache TTL (explanations are cached until values change significantly)
    CACHE_TTL = timedelta(hours=1)

    # Confidence change threshold for regeneration (5%)
    CONFIDENCE_THRESHOLD = 0.05

    # VIX change threshold for regeneration
    VIX_THRESHOLD = 2.0

    def __init__(self):
        self._lock = Lock()
        self._client = None
        self._initialized = False

        # Cache storage
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._last_values: Dict[str, Any] = {}

    def initialize(self) -> bool:
        """Initialize the OpenAI client."""
        if self._initialized:
            return True

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set - explanation service disabled")
            return False

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
            self._initialized = True
            logger.info("ExplanationService initialized successfully")
            return True
        except ImportError:
            logger.error("OpenAI package not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return False

    def _compute_values_hash(self, values: Dict[str, Any]) -> str:
        """Compute a hash of the significant values for cache key."""
        # Extract key values that affect the explanation
        key_parts = [
            str(values.get("direction", "")),
            str(values.get("should_trade", "")),
            # Round confidence to nearest 5% for stability
            str(round(values.get("confidence", 0) * 20) / 20),
            # Round VIX to nearest integer
            str(round(values.get("vix", 0))),
            # Include timeframe agreement
            str(values.get("agreement_count", 0)),
        ]

        hash_input = "|".join(key_parts)
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]

    def _should_regenerate(self, current_values: Dict[str, Any]) -> bool:
        """Check if we should regenerate the explanation based on value changes."""
        if not self._last_values:
            return True

        # Check direction change
        if current_values.get("direction") != self._last_values.get("direction"):
            return True

        # Check should_trade change
        if current_values.get("should_trade") != self._last_values.get("should_trade"):
            return True

        # Check significant confidence change
        conf_diff = abs(
            current_values.get("confidence", 0) -
            self._last_values.get("confidence", 0)
        )
        if conf_diff >= self.CONFIDENCE_THRESHOLD:
            return True

        # Check significant VIX change
        vix_diff = abs(
            current_values.get("vix", 0) -
            self._last_values.get("vix", 0)
        )
        if vix_diff >= self.VIX_THRESHOLD:
            return True

        # Check agreement count change
        if current_values.get("agreement_count") != self._last_values.get("agreement_count"):
            return True

        return False

    def _cleanup_expired_cache(self) -> None:
        """Remove expired cache entries.

        Must be called while holding self._lock.
        This prevents memory from growing with old, unused cache entries.
        """
        now = datetime.now()
        expired_keys = [
            k for k, v in self._cache.items()
            if now - v["generated_at"] > self.CACHE_TTL
        ]
        for key in expired_keys:
            del self._cache[key]
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def _build_prompt(self, data: Dict[str, Any]) -> str:
        """Build the prompt for GPT-4o-mini."""
        # Extract values
        direction = data.get("direction", "unknown")
        confidence = data.get("confidence", 0)
        should_trade = data.get("should_trade", False)
        vix = data.get("vix")
        symbol = data.get("symbol", "EUR/USD")
        asset_type = data.get("asset_type", "forex")
        current_price = data.get("current_price")

        # Component signals
        component_directions = data.get("component_directions", {})
        component_confidences = data.get("component_confidences", {})
        agreement_count = data.get("agreement_count", 0)
        all_agree = data.get("all_agree", False)
        market_regime = data.get("market_regime", "unknown")

        # Build the recommendation text
        if should_trade:
            recommendation = "BUY" if direction == "long" else "SELL"
        else:
            recommendation = "HOLD"

        # Format timeframe breakdown
        tf_breakdown = []
        for tf in ["1H", "4H", "D"]:
            if tf in component_directions:
                dir_text = "bullish" if component_directions[tf] == 1 else "bearish"
                conf_text = f"{component_confidences.get(tf, 0) * 100:.0f}%"
                tf_breakdown.append(f"{tf}: {dir_text} ({conf_text})")

        # VIX interpretation
        vix_text = "N/A"
        if vix is not None:
            if vix < 15:
                vix_text = f"{vix:.1f} (low volatility - calm markets)"
            elif vix < 20:
                vix_text = f"{vix:.1f} (normal volatility)"
            elif vix < 30:
                vix_text = f"{vix:.1f} (elevated volatility - increased uncertainty)"
            else:
                vix_text = f"{vix:.1f} (high volatility - fear in markets)"

        # Convert confidence to descriptive level
        conf_pct = confidence * 100
        if conf_pct >= 85:
            conf_level = f"very high ({conf_pct:.0f}%)"
        elif conf_pct >= 75:
            conf_level = f"high ({conf_pct:.0f}%)"
        elif conf_pct >= 65:
            conf_level = f"medium ({conf_pct:.0f}%)"
        else:
            conf_level = f"low ({conf_pct:.0f}%)"

        # Format timeframe breakdown with 1D instead of D
        tf_formatted = []
        for tf in ["1H", "4H", "D"]:
            if tf in component_directions:
                dir_text = "bullish" if component_directions[tf] == 1 else "bearish"
                tf_conf = component_confidences.get(tf, 0) * 100
                tf_name = "1D" if tf == "D" else tf
                tf_formatted.append(f"{tf_name}: {dir_text} ({tf_conf:.0f}%)")

        prompt = f"""Summarize this AI trading recommendation in 1-2 concise sentences.

Data:
- Asset: {symbol} ({asset_type})
- Price: {current_price if current_price else 'N/A'}
- Recommendation: {recommendation}
- Confidence: {conf_level}
- Timeframes: {', '.join(tf_formatted) if tf_formatted else 'N/A'}
- Agreement: {agreement_count}/3 timeframes{' (unanimous)' if all_agree else ''}
- Market regime: {market_regime.replace('_', ' ')}
- Market volatility: {vix_text}

Write 1-2 sentences that:
1. State the recommendation with confidence level (e.g., "high (76%)")
2. Mention which timeframes agree (use 1H, 4H, 1D - never just "D")
3. Include market volatility context (never use "VIX" acronym - say "market volatility" or "fear index")

Example: "BUY with high (76%) confidence as all timeframes (1H, 4H, 1D) show bullish signals. Market volatility is normal at 18.8."
"""

        return prompt

    def generate_explanation(
        self,
        prediction: Dict[str, Any],
        vix: Optional[float] = None,
        current_price: Optional[float] = None,
        symbol: str = "EUR/USD",
        asset_type: str = "forex",
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Generate an explanation for the current trading recommendation.

        Args:
            prediction: The prediction dict from model_service
            vix: Current VIX value
            current_price: Current market price
            symbol: Trading symbol
            asset_type: Type of asset (forex, crypto, stock)
            force_refresh: Force regeneration even if cached

        Returns:
            Dict with explanation text and metadata
        """
        if not self._initialized:
            if not self.initialize():
                return {
                    "explanation": None,
                    "error": "Explanation service not available",
                    "cached": False,
                }

        # Prepare values for comparison
        current_values = {
            "direction": prediction.get("direction"),
            "confidence": prediction.get("confidence", 0),
            "should_trade": prediction.get("should_trade", False),
            "vix": vix or 0,
            "agreement_count": prediction.get("agreement_count", 0),
        }

        # Check cache
        values_hash = self._compute_values_hash(current_values)

        with self._lock:
            if not force_refresh and values_hash in self._cache:
                cached = self._cache[values_hash]
                cache_age = datetime.now() - cached["generated_at"]
                if cache_age < self.CACHE_TTL:
                    logger.debug(f"Returning cached explanation ({cache_age.seconds}s old)")
                    return {
                        "explanation": cached["explanation"],
                        "generated_at": cached["generated_at"].isoformat(),
                        "cached": True,
                        "values_hash": values_hash,
                    }

            # Check if values changed significantly
            if not force_refresh and not self._should_regenerate(current_values):
                # Return previous explanation if available
                if self._cache:
                    last_cached = list(self._cache.values())[-1]
                    return {
                        "explanation": last_cached["explanation"],
                        "generated_at": last_cached["generated_at"].isoformat(),
                        "cached": True,
                        "values_hash": values_hash,
                    }

        # Build data for prompt
        data = {
            **prediction,
            "vix": vix,
            "current_price": current_price,
            "symbol": symbol,
            "asset_type": asset_type,
        }

        try:
            prompt = self._build_prompt(data)

            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You summarize trading data concisely. Use specific numbers. No fluff or filler words. Max 2 sentences."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.3,
            )

            explanation = response.choices[0].message.content.strip()

            # Cache the result
            with self._lock:
                # First, clean up expired entries to prevent memory leak
                self._cleanup_expired_cache()

                self._cache[values_hash] = {
                    "explanation": explanation,
                    "generated_at": datetime.now(),
                }
                self._last_values = current_values.copy()

                # Keep cache size manageable (max 10 entries, oldest-first eviction)
                if len(self._cache) > 10:
                    # Remove oldest entry by generated_at time
                    oldest_key = min(
                        self._cache.keys(),
                        key=lambda k: self._cache[k]["generated_at"]
                    )
                    del self._cache[oldest_key]

            logger.info("Generated new explanation")

            return {
                "explanation": explanation,
                "generated_at": datetime.now().isoformat(),
                "cached": False,
                "values_hash": values_hash,
            }

        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            return {
                "explanation": None,
                "error": str(e),
                "cached": False,
            }

    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            "initialized": self._initialized,
            "cache_entries": len(self._cache),
            "api_key_set": bool(os.environ.get("OPENAI_API_KEY")),
        }

    def clear_cache(self) -> None:
        """Clear the explanation cache."""
        with self._lock:
            self._cache.clear()
            self._last_values.clear()
        logger.info("Explanation cache cleared")


# Singleton instance
explanation_service = ExplanationService()
