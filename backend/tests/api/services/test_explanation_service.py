"""Unit tests for ExplanationService."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock


class TestExplanationServiceInitialization:
    """Test ExplanationService initialization."""

    def test_initial_state(self):
        """Test service starts with correct initial state."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()

        assert service._client is None
        assert service._initialized is False
        assert service._cache == {}
        assert service._last_values == {}

    def test_initialize_without_api_key(self):
        """Test initialize fails gracefully without API key."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()

        with patch.dict('os.environ', {}, clear=True):
            result = service.initialize()

        assert result is False
        assert service._initialized is False

    def test_initialize_with_api_key(self):
        """Test initialize succeeds with API key."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            with patch('src.api.services.explanation_service.OpenAI') as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client

                result = service.initialize()

        assert result is True
        assert service._initialized is True
        assert service._client == mock_client

    def test_initialize_already_initialized(self):
        """Test initialize returns True immediately if already initialized."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()
        service._initialized = True

        result = service.initialize()

        assert result is True

    def test_initialize_import_error(self):
        """Test initialize handles missing openai package."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            with patch('src.api.services.explanation_service.OpenAI', side_effect=ImportError):
                result = service.initialize()

        assert result is False
        assert service._initialized is False

    def test_initialize_exception(self):
        """Test initialize handles OpenAI client creation errors."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            with patch('src.api.services.explanation_service.OpenAI', side_effect=Exception("Connection error")):
                result = service.initialize()

        assert result is False
        assert service._initialized is False


class TestExplanationServiceCaching:
    """Test ExplanationService caching logic."""

    def test_compute_values_hash_same_values(self):
        """Test hash is same for identical values."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()

        values1 = {
            "direction": "long",
            "should_trade": True,
            "confidence": 0.725,
            "vix": 16.3,
            "agreement_count": 3,
        }

        values2 = {
            "direction": "long",
            "should_trade": True,
            "confidence": 0.725,
            "vix": 16.3,
            "agreement_count": 3,
        }

        hash1 = service._compute_values_hash(values1)
        hash2 = service._compute_values_hash(values2)

        assert hash1 == hash2

    def test_compute_values_hash_different_direction(self):
        """Test hash differs when direction changes."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()

        values1 = {
            "direction": "long",
            "should_trade": True,
            "confidence": 0.725,
            "vix": 16.3,
            "agreement_count": 3,
        }

        values2 = {
            "direction": "short",
            "should_trade": True,
            "confidence": 0.725,
            "vix": 16.3,
            "agreement_count": 3,
        }

        hash1 = service._compute_values_hash(values1)
        hash2 = service._compute_values_hash(values2)

        assert hash1 != hash2

    def test_compute_values_hash_rounds_confidence(self):
        """Test hash rounds confidence to nearest 5% for stability."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()

        # 0.721 and 0.729 should both round to 0.70 (nearest 5%)
        values1 = {"direction": "long", "should_trade": True, "confidence": 0.721, "vix": 15, "agreement_count": 3}
        values2 = {"direction": "long", "should_trade": True, "confidence": 0.729, "vix": 15, "agreement_count": 3}

        hash1 = service._compute_values_hash(values1)
        hash2 = service._compute_values_hash(values2)

        assert hash1 == hash2

    def test_compute_values_hash_rounds_vix(self):
        """Test hash rounds VIX to nearest integer for stability."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()

        # 16.3 and 16.7 should both round to 16
        values1 = {"direction": "long", "should_trade": True, "confidence": 0.72, "vix": 16.3, "agreement_count": 3}
        values2 = {"direction": "long", "should_trade": True, "confidence": 0.72, "vix": 16.7, "agreement_count": 3}

        hash1 = service._compute_values_hash(values1)
        hash2 = service._compute_values_hash(values2)

        assert hash1 == hash2

    def test_should_regenerate_no_last_values(self):
        """Test should_regenerate returns True when no last values."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()

        current_values = {
            "direction": "long",
            "should_trade": True,
            "confidence": 0.72,
            "vix": 16.5,
            "agreement_count": 3,
        }

        assert service._should_regenerate(current_values) is True

    def test_should_regenerate_direction_change(self):
        """Test should_regenerate returns True when direction changes."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()
        service._last_values = {
            "direction": "long",
            "should_trade": True,
            "confidence": 0.72,
            "vix": 16.5,
            "agreement_count": 3,
        }

        current_values = {
            "direction": "short",
            "should_trade": True,
            "confidence": 0.72,
            "vix": 16.5,
            "agreement_count": 3,
        }

        assert service._should_regenerate(current_values) is True

    def test_should_regenerate_should_trade_change(self):
        """Test should_regenerate returns True when should_trade changes."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()
        service._last_values = {
            "direction": "long",
            "should_trade": True,
            "confidence": 0.72,
            "vix": 16.5,
            "agreement_count": 3,
        }

        current_values = {
            "direction": "long",
            "should_trade": False,
            "confidence": 0.72,
            "vix": 16.5,
            "agreement_count": 3,
        }

        assert service._should_regenerate(current_values) is True

    def test_should_regenerate_confidence_threshold(self):
        """Test should_regenerate detects significant confidence changes."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()
        service._last_values = {
            "direction": "long",
            "should_trade": True,
            "confidence": 0.70,
            "vix": 16.5,
            "agreement_count": 3,
        }

        # 4% change - should not regenerate (threshold is 5%)
        current_values_small = {
            "direction": "long",
            "should_trade": True,
            "confidence": 0.74,
            "vix": 16.5,
            "agreement_count": 3,
        }
        assert service._should_regenerate(current_values_small) is False

        # 6% change - should regenerate
        current_values_large = {
            "direction": "long",
            "should_trade": True,
            "confidence": 0.76,
            "vix": 16.5,
            "agreement_count": 3,
        }
        assert service._should_regenerate(current_values_large) is True

    def test_should_regenerate_vix_threshold(self):
        """Test should_regenerate detects significant VIX changes."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()
        service._last_values = {
            "direction": "long",
            "should_trade": True,
            "confidence": 0.72,
            "vix": 15.0,
            "agreement_count": 3,
        }

        # 1.5 change - should not regenerate (threshold is 2.0)
        current_values_small = {
            "direction": "long",
            "should_trade": True,
            "confidence": 0.72,
            "vix": 16.5,
            "agreement_count": 3,
        }
        assert service._should_regenerate(current_values_small) is False

        # 2.5 change - should regenerate
        current_values_large = {
            "direction": "long",
            "should_trade": True,
            "confidence": 0.72,
            "vix": 17.5,
            "agreement_count": 3,
        }
        assert service._should_regenerate(current_values_large) is True

    def test_should_regenerate_agreement_count_change(self):
        """Test should_regenerate returns True when agreement count changes."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()
        service._last_values = {
            "direction": "long",
            "should_trade": True,
            "confidence": 0.72,
            "vix": 16.5,
            "agreement_count": 3,
        }

        current_values = {
            "direction": "long",
            "should_trade": True,
            "confidence": 0.72,
            "vix": 16.5,
            "agreement_count": 2,
        }

        assert service._should_regenerate(current_values) is True

    def test_should_regenerate_no_change(self):
        """Test should_regenerate returns False when values unchanged."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()
        service._last_values = {
            "direction": "long",
            "should_trade": True,
            "confidence": 0.72,
            "vix": 16.5,
            "agreement_count": 3,
        }

        current_values = {
            "direction": "long",
            "should_trade": True,
            "confidence": 0.72,
            "vix": 16.5,
            "agreement_count": 3,
        }

        assert service._should_regenerate(current_values) is False


class TestExplanationServicePromptBuilding:
    """Test ExplanationService prompt building."""

    def test_build_prompt_buy_signal(self):
        """Test prompt building for BUY signal."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()

        data = {
            "direction": "long",
            "confidence": 0.75,
            "should_trade": True,
            "vix": 16.5,
            "symbol": "EUR/USD",
            "asset_type": "forex",
            "current_price": 1.08543,
            "component_directions": {"1H": 1, "4H": 1, "D": 1},
            "component_confidences": {"1H": 0.80, "4H": 0.75, "D": 0.72},
            "agreement_count": 3,
            "all_agree": True,
            "market_regime": "trending_normal",
        }

        prompt = service._build_prompt(data)

        assert "BUY" in prompt
        assert "EUR/USD" in prompt
        assert "75%" in prompt
        assert "1H: bullish (80%)" in prompt
        assert "4H: bullish (75%)" in prompt
        assert "D: bullish (72%)" in prompt
        assert "3/3 timeframes agree (unanimous)" in prompt

    def test_build_prompt_sell_signal(self):
        """Test prompt building for SELL signal."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()

        data = {
            "direction": "short",
            "confidence": 0.72,
            "should_trade": True,
            "vix": 18.2,
            "symbol": "EUR/USD",
            "asset_type": "forex",
            "current_price": 1.08123,
            "component_directions": {"1H": -1, "4H": -1, "D": -1},
            "component_confidences": {"1H": 0.75, "4H": 0.72, "D": 0.68},
            "agreement_count": 3,
            "all_agree": True,
            "market_regime": "trending_high_vol",
        }

        prompt = service._build_prompt(data)

        assert "SELL" in prompt
        assert "72%" in prompt
        assert "1H: bearish (75%)" in prompt
        assert "4H: bearish (72%)" in prompt
        assert "D: bearish (68%)" in prompt

    def test_build_prompt_hold_signal(self):
        """Test prompt building for HOLD signal."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()

        data = {
            "direction": "long",
            "confidence": 0.65,
            "should_trade": False,
            "vix": 22.5,
            "symbol": "EUR/USD",
            "asset_type": "forex",
            "current_price": 1.08321,
            "component_directions": {"1H": 1, "4H": -1, "D": 1},
            "component_confidences": {"1H": 0.68, "4H": 0.62, "D": 0.60},
            "agreement_count": 2,
            "all_agree": False,
            "market_regime": "ranging_high_vol",
        }

        prompt = service._build_prompt(data)

        assert "HOLD" in prompt
        assert "65%" in prompt
        assert "2/3 timeframes agree" in prompt
        assert "(unanimous)" not in prompt

    def test_build_prompt_vix_interpretation(self):
        """Test VIX interpretation in prompt."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()

        base_data = {
            "direction": "long",
            "confidence": 0.75,
            "should_trade": True,
            "symbol": "EUR/USD",
            "asset_type": "forex",
            "component_directions": {},
            "component_confidences": {},
            "agreement_count": 3,
            "all_agree": True,
        }

        # Low VIX
        data_low = {**base_data, "vix": 12.5}
        prompt_low = service._build_prompt(data_low)
        assert "low volatility - calm markets" in prompt_low

        # Normal VIX
        data_normal = {**base_data, "vix": 17.0}
        prompt_normal = service._build_prompt(data_normal)
        assert "normal volatility" in prompt_normal

        # Elevated VIX
        data_elevated = {**base_data, "vix": 25.0}
        prompt_elevated = service._build_prompt(data_elevated)
        assert "elevated volatility - increased uncertainty" in prompt_elevated

        # High VIX
        data_high = {**base_data, "vix": 35.0}
        prompt_high = service._build_prompt(data_high)
        assert "high volatility - fear in markets" in prompt_high

    def test_build_prompt_no_vix(self):
        """Test prompt building without VIX data."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()

        data = {
            "direction": "long",
            "confidence": 0.75,
            "should_trade": True,
            "vix": None,
            "symbol": "EUR/USD",
            "asset_type": "forex",
            "component_directions": {},
            "component_confidences": {},
        }

        prompt = service._build_prompt(data)

        assert "VIX (Market Fear Index): N/A" in prompt

    def test_build_prompt_crypto_asset(self):
        """Test prompt building for crypto asset."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()

        data = {
            "direction": "long",
            "confidence": 0.75,
            "should_trade": True,
            "vix": 16.5,
            "symbol": "BTC/USD",
            "asset_type": "crypto",
            "current_price": 45123.45,
            "component_directions": {},
            "component_confidences": {},
        }

        prompt = service._build_prompt(data)

        assert "BTC/USD" in prompt
        assert "Crypto" in prompt


class TestExplanationServiceGeneration:
    """Test ExplanationService explanation generation."""

    def test_generate_explanation_not_initialized(self):
        """Test generate_explanation returns error when not initialized."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()

        with patch.dict('os.environ', {}, clear=True):
            prediction = {
                "direction": "long",
                "confidence": 0.75,
                "should_trade": True,
            }

            result = service.generate_explanation(prediction)

        assert result["explanation"] is None
        assert "not available" in result["error"]
        assert result["cached"] is False

    def test_generate_explanation_returns_cached(self):
        """Test generate_explanation returns cached result when available."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()
        service._initialized = True
        service._client = Mock()

        # Set up cache
        cache_time = datetime.now()
        values_hash = service._compute_values_hash({
            "direction": "long",
            "confidence": 0.75,
            "should_trade": True,
            "vix": 16,
            "agreement_count": 3,
        })

        service._cache[values_hash] = {
            "explanation": "Cached explanation",
            "generated_at": cache_time,
        }

        prediction = {
            "direction": "long",
            "confidence": 0.75,
            "should_trade": True,
            "agreement_count": 3,
        }

        result = service.generate_explanation(prediction, vix=16.5)

        assert result["explanation"] == "Cached explanation"
        assert result["cached"] is True
        assert result["values_hash"] == values_hash

    def test_generate_explanation_expired_cache(self):
        """Test generate_explanation regenerates when cache expired."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()
        service._initialized = True

        # Mock OpenAI client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Fresh explanation"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        service._client = mock_client

        # Set up expired cache
        cache_time = datetime.now() - timedelta(hours=2)
        values_hash = service._compute_values_hash({
            "direction": "long",
            "confidence": 0.75,
            "should_trade": True,
            "vix": 16,
            "agreement_count": 3,
        })

        service._cache[values_hash] = {
            "explanation": "Old explanation",
            "generated_at": cache_time,
        }

        prediction = {
            "direction": "long",
            "confidence": 0.75,
            "should_trade": True,
            "agreement_count": 3,
        }

        result = service.generate_explanation(prediction, vix=16.5)

        assert result["explanation"] == "Fresh explanation"
        assert result["cached"] is False
        assert mock_client.chat.completions.create.called

    def test_generate_explanation_force_refresh(self):
        """Test generate_explanation with force_refresh bypasses cache."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()
        service._initialized = True

        # Mock OpenAI client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Fresh explanation"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        service._client = mock_client

        # Set up valid cache
        cache_time = datetime.now()
        values_hash = service._compute_values_hash({
            "direction": "long",
            "confidence": 0.75,
            "should_trade": True,
            "vix": 16,
            "agreement_count": 3,
        })

        service._cache[values_hash] = {
            "explanation": "Cached explanation",
            "generated_at": cache_time,
        }

        prediction = {
            "direction": "long",
            "confidence": 0.75,
            "should_trade": True,
            "agreement_count": 3,
        }

        result = service.generate_explanation(prediction, vix=16.5, force_refresh=True)

        assert result["explanation"] == "Fresh explanation"
        assert result["cached"] is False
        assert mock_client.chat.completions.create.called

    def test_generate_explanation_api_call(self):
        """Test generate_explanation makes correct OpenAI API call."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()
        service._initialized = True

        # Mock OpenAI client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated explanation"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        service._client = mock_client

        prediction = {
            "direction": "long",
            "confidence": 0.75,
            "should_trade": True,
            "agreement_count": 3,
            "component_directions": {"1H": 1, "4H": 1, "D": 1},
            "component_confidences": {"1H": 0.80, "4H": 0.75, "D": 0.72},
        }

        result = service.generate_explanation(prediction, vix=16.5, symbol="EUR/USD")

        assert result["explanation"] == "Generated explanation"
        assert result["cached"] is False

        # Verify API call
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["max_tokens"] == 200
        assert call_kwargs["temperature"] == 0.7

    def test_generate_explanation_api_error(self):
        """Test generate_explanation handles API errors gracefully."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()
        service._initialized = True

        # Mock OpenAI client to raise error
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        service._client = mock_client

        prediction = {
            "direction": "long",
            "confidence": 0.75,
            "should_trade": True,
            "agreement_count": 3,
        }

        result = service.generate_explanation(prediction, vix=16.5)

        assert result["explanation"] is None
        assert "API error" in result["error"]
        assert result["cached"] is False

    def test_generate_explanation_cache_limit(self):
        """Test generate_explanation maintains cache size limit."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()
        service._initialized = True

        # Mock OpenAI client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated explanation"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        service._client = mock_client

        # Fill cache with 10 entries
        for i in range(10):
            service._cache[f"hash_{i}"] = {
                "explanation": f"Explanation {i}",
                "generated_at": datetime.now(),
            }

        assert len(service._cache) == 10

        # Generate new explanation (should trigger cache cleanup)
        prediction = {
            "direction": "long",
            "confidence": 0.75,
            "should_trade": True,
            "agreement_count": 3,
        }

        service.generate_explanation(prediction, vix=16.5)

        # Cache should still be at 10 (oldest removed)
        assert len(service._cache) == 10


class TestExplanationServiceStatus:
    """Test ExplanationService status methods."""

    def test_get_status_not_initialized(self):
        """Test get_status when service not initialized."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()

        with patch.dict('os.environ', {}, clear=True):
            status = service.get_status()

        assert status["initialized"] is False
        assert status["cache_entries"] == 0
        assert status["api_key_set"] is False

    def test_get_status_initialized(self):
        """Test get_status when service initialized."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()
        service._initialized = True
        service._cache = {"key1": {"explanation": "test"}}

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            status = service.get_status()

        assert status["initialized"] is True
        assert status["cache_entries"] == 1
        assert status["api_key_set"] is True

    def test_clear_cache(self):
        """Test clear_cache clears both cache and last values."""
        from src.api.services.explanation_service import ExplanationService

        service = ExplanationService()
        service._cache = {"key1": {"explanation": "test"}}
        service._last_values = {"direction": "long"}

        service.clear_cache()

        assert service._cache == {}
        assert service._last_values == {}
