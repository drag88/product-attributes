from anthropic import AsyncAnthropic
from anthropic._exceptions import (
    APIError, APITimeoutError, 
    APIConnectionError, RateLimitError
)

import asyncio
import time
import logging
import csv
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Any, List
from tenacity import (
    retry, stop_after_attempt, 
    wait_exponential, retry_if_exception_type, RetryCallState, before_sleep_log
)
import cohere
from anthropic import Anthropic
import os
from dotenv import load_dotenv
from threading import Lock

load_dotenv()  # Load environment variables from .env file

@dataclass
class APIRateConfig:
    max_retries: int
    base_delay: int 
    max_delay: int
    concurrent_limit: int
    batch_size: int

class APIRateLimiter:
    def __init__(self, config: APIRateConfig):
        self.semaphore = asyncio.Semaphore(config.concurrent_limit)
        self.last_request_time = 0
        self.min_request_interval = 1.0 / config.batch_size
        self.retry_config = config

    async def wait_if_needed(self):
        """Wait to maintain rate limit if needed"""
        elapsed = time.time() - self.last_request_time
        wait_time = max(self.min_request_interval - elapsed, 0)
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        self.last_request_time = time.time()

logger = logging.getLogger(__name__)

def log_retry_attempt(retry_state: RetryCallState) -> None:
    logger.warning(
        f"Retry attempt {retry_state.attempt_number} "
        f"after {retry_state.outcome.exception()} "
        f"- Next retry in {retry_state.next_action.sleep} seconds"
    )

class CostTracker:
    def __init__(self):
        self._lock = Lock()
        self.anthropic_costs = []
        self.cohere_costs = []

    def track_anthropic(self, response):
        with self._lock:
            cost = self._calculate_anthropic_cost(response)
            self.anthropic_costs.append({
                'timestamp': datetime.now().isoformat(),
                'model': response.model,
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens,
                'cost': cost
            })

    def track_cohere(self, response, method):
        with self._lock:
            self.cohere_costs.append({
                'timestamp': datetime.now().isoformat(),
                'method': method,
                'response': response.__dict__
            })

    def export_report(self, output_dir: Path):
        with self._lock:
            self._export_csv(output_dir / 'anthropic_costs.csv', self.anthropic_costs)
            self._export_csv(output_dir / 'cohere_costs.csv', self.cohere_costs)

    def _export_csv(self, path: Path, data: List[Dict]):
        if not data:
            return
            
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

class APIService:
    def __init__(self, config: Dict):
        # Get API keys from environment
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        cohere_api_key = os.getenv("COHERE_API_KEY")
        
        if not anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY not found in environment")

        # Initialize clients with env vars
        self.async_client = AsyncAnthropic(api_key=anthropic_api_key)
        self.sync_client = Anthropic(api_key=anthropic_api_key)  # Only if needed for sync operations
        self.cohere_client = cohere.Client(api_key=cohere_api_key)

        # Rest of config values from YAML
        self.anthropic_config = config["api"]["anthropic"]
        self.cohere_config = config["api"]["cohere"]
        # self.client = AsyncAnthropic(api_key=anthropic_api_key)
        self.rate_limiter = APIRateLimiter(APIRateConfig(
            max_retries=self.anthropic_config["max_retries"],
            base_delay=self.anthropic_config["base_delay"],
            max_delay=self.anthropic_config["max_delay"],
            concurrent_limit=self.anthropic_config["concurrent_limit"],
            batch_size=self.anthropic_config["batch_size"]
        ))
        self.model = config.get("model", "claude-3-5-sonnet-20240620")

        # Enhanced components
        self.cost_tracker = CostTracker()

    def calculate_cost(self, response: Any) -> Dict[str, Any]:
        """Calculate API call cost based on token usage."""
        pricing = self.anthropic_config["pricing"]
        
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
                response = await self.async_client.messages.create(**kwargs)
                self.cost_tracker.track_anthropic(response)
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
            response = api_method(**kwargs)
            self.cost_tracker.track_cohere(response, method)
            return response
        except Exception as e:
            logger.error(
                f"Cohere API Error: {str(e)}, Method: {method}",
                exc_info=True
            )
            raise

    def export_cost_report(self, output_dir: str):
        """Export accumulated cost data"""
        self.cost_tracker.export_report(Path(output_dir))