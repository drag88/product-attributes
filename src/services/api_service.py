from anthropic import AsyncAnthropic
from anthropic._exceptions import (
    APIError, APITimeoutError, 
    APIConnectionError, RateLimitError
)
import asyncio
import time
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any, List
from tenacity import (
    retry, stop_after_attempt, 
    wait_exponential, retry_if_exception_type, RetryCallState, before_sleep_log
)
import cohere
from anthropic import Anthropic

@dataclass
class APIRateConfig:
    max_retries: int = 3
    base_delay: int = 4
    max_delay: int = 60
    concurrent_limit: int = 3
    batch_size: int = 5

class APIRateLimiter:
    def __init__(self, config: APIRateConfig):
        self.semaphore = asyncio.Semaphore(config.concurrent_limit)
        self.last_request_time = 0
        self.min_request_interval = 1.0

    async def wait_if_needed(self):
        now = time.time()
        if now - self.last_request_time < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval)
        self.last_request_time = now

CLAUDE_PRICING = {
    "claude-3-5-sonnet-20240620": {
        "base_input": 3.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
        "output": 15.00
    },
    "claude-3-haiku-20240307": {
        "base_input": 0.25,
        "cache_write": 0.30,
        "cache_read": 0.03,
        "output": 1.25
    }
}

logger = logging.getLogger(__name__)

def log_retry_attempt(retry_state: RetryCallState) -> None:
    logger.warning(
        f"Retry attempt {retry_state.attempt_number} "
        f"after {retry_state.outcome.exception()} "
        f"- Next retry in {retry_state.next_action.sleep} seconds"
    )

class APIService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.anthropic_client = Anthropic()
        self.cohere_client = cohere.Client(
            api_key=config["api"]["cohere"]["api_key"]
        )
        self.client = AsyncAnthropic(api_key=config["api_key"])
        self.rate_limiter = APIRateLimiter(APIRateConfig())
        self.model = config.get("model", "claude-3-5-sonnet-20240620")

    def calculate_cost(self, response: Any) -> Dict[str, Any]:
        """Calculate API call cost based on token usage."""
        pricing = self.config["api"]["anthropic"]["pricing"]
        
        # Input costs
        if hasattr(response.usage, 'cache_creation_input_tokens'):
            input_cost = (
                response.usage.cache_creation_input_tokens / 1_000_000
            ) * pricing["cache_write"]
            cache_status = "WRITE"
        elif hasattr(response.usage, 'cache_read_input_tokens'):
            input_cost = (
                response.usage.cache_read_input_tokens / 1_000_000
            ) * pricing["cache_read"]
            cache_status = "READ"
        else:
            input_cost = (
                response.usage.input_tokens / 1_000_000
            ) * pricing["base_input"]
            cache_status = "MISS"
        
        # Output costs
        output_cost = (
            response.usage.output_tokens / 1_000_000
        ) * pricing["output"]
        
        return {
            "cache_status": cache_status,
            "input_tokens": (
                response.usage.cache_creation_input_tokens
                if hasattr(response.usage, 'cache_creation_input_tokens')
                else response.usage.cache_read_input_tokens
                if hasattr(response.usage, 'cache_read_input_tokens')
                else response.usage.input_tokens
            ),
            "output_tokens": response.usage.output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost
        }

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def call_api(self, **kwargs) -> Any:
        """Call Anthropic API with retry logic."""
        try:
            async with self.rate_limiter.semaphore:
                await self.rate_limiter.wait_if_needed()
                response = await self.anthropic_client.messages.create(**kwargs)
                return response
        except Exception as e:
            logger.error(f"Anthropic API Error: {str(e)}", exc_info=True)
            raise

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def call_cohere_api(
        self,
        method: str,
        **kwargs
    ) -> Any:
        """Call Cohere API with retry logic."""
        try:
            api_method = getattr(self.cohere_client, method)
            return api_method(**kwargs)
        except Exception as e:
            logger.error(
                f"Cohere API Error: {str(e)}, Method: {method}",
                exc_info=True
            )
            raise 