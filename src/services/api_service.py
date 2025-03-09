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
        
        # Default pricing for different models
        self._pricing = {
            "claude-3-5-sonnet-20240620": {
                "input": 3.00,
                "output": 15.00
            },
            "claude-3-opus-20240229": {
                "input": 15.00,
                "output": 75.00
            }
        }
    
    def _calculate_anthropic_cost(self, response) -> float:
        """Calculate cost based on token usage."""
        try:
            if not hasattr(response, 'usage') or not response.usage:
                return 0.0
                
            model = getattr(response, 'model', 'claude-3-5-sonnet-20240620')
            
            # Use default pricing if model not found
            pricing = self._pricing.get(model, {
                "input": 3.00,
                "output": 15.00
            })
            
            input_cost = (response.usage.input_tokens / 1_000_000) * pricing["input"]
            output_cost = (response.usage.output_tokens / 1_000_000) * pricing["output"]
            return input_cost + output_cost
        except Exception as e:
            logger.error(f"Error calculating cost: {str(e)}")
            return 0.0
    
    def track_anthropic(self, response):
        """Track cost for Anthropic API call."""
        try:
            with self._lock:
                if not hasattr(response, 'usage'):
                    logger.warning("Response missing usage information")
                    return
                    
                cost = self._calculate_anthropic_cost(response)
                timestamp = datetime.now().isoformat()
                model = getattr(response, 'model', 'unknown')
                
                self.anthropic_costs.append({
                    'timestamp': timestamp,
                    'model': model,
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens,
                    'cost': cost
                })
        except Exception as e:
            logger.error(f"Error tracking Anthropic cost: {str(e)}")
    
    def track_cohere(self, response, method):
        """Track cost for Cohere API call."""
        with self._lock:
            timestamp = datetime.now().isoformat()
            
            # Get tokens using proper attribute access
            try:
                billed_units = getattr(response.meta, 'billed_units', None)
                tokens = billed_units.input_tokens if billed_units else 0
            except AttributeError:
                tokens = 0
            
            self.cohere_costs.append({
                'timestamp': timestamp,
                'method': method,
                'model': getattr(response, 'model', 'unknown'),
                'tokens': tokens,
                'cost': 0.0  # Placeholder
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
            logger.warning("ANTHROPIC_API_KEY not found in environment variables")
        
        if not cohere_api_key:
            logger.warning("COHERE_API_KEY not found in environment variables")
        
        # Initialize clients with env vars
        self.async_client = AsyncAnthropic(api_key=anthropic_api_key)
        self.sync_client = Anthropic(api_key=anthropic_api_key)  # Only if needed for sync operations
        self.cohere_client = cohere.Client(
            api_key=cohere_api_key,
            client_name="product-attributes"  # Valid parameter
        )
        
        # Default configuration values
        default_anthropic_config = {
            "model": "claude-3-5-sonnet-20240620",
            "max_retries": 3,
            "base_delay": 4,
            "max_delay": 60,
            "concurrent_limit": 3,
            "batch_size": 5,
            "pricing": {
                "claude-3-5-sonnet-20240620": {
                    "input": 3.00,
                    "output": 15.00
                },
                "claude-3-opus-20240229": {
                    "input": 15.00,
                    "output": 75.00
                }
            }
        }
        
        default_cohere_config = {
            "model": "embed-english-v3.0",
            "max_retries": 3,
            "base_delay": 4,
            "max_delay": 60
        }
        
        # Rest of config values from YAML, with fallbacks to defaults
        api_config = config.get("api", {})
        self.anthropic_config = api_config.get("anthropic", default_anthropic_config)
        self.cohere_config = api_config.get("cohere", default_cohere_config)
        
        self.rate_limiter = APIRateLimiter(APIRateConfig(
            max_retries=self.anthropic_config.get("max_retries", 3),
            base_delay=self.anthropic_config.get("base_delay", 4),
            max_delay=self.anthropic_config.get("max_delay", 60),
            concurrent_limit=self.anthropic_config.get("concurrent_limit", 3),
            batch_size=self.anthropic_config.get("batch_size", 5)
        ))
        self.model = self.anthropic_config.get("model", "claude-3-5-sonnet-20240620")
        
        # Enhanced components
        self.cost_tracker = CostTracker()

    def calculate_cost(self, response: Any) -> Dict[str, Any]:
        """Calculate cost of API call based on response."""
        try:
            # Get pricing from config
            pricing = self.anthropic_config.get("pricing", {})
            model = getattr(response, "model", self.model)
            
            # Default pricing if not found in config
            default_pricing = {
                "input": 3.00,
                "output": 15.00
            }
            
            model_pricing = pricing.get(model, default_pricing)
            
            # Get usage from response
            usage = getattr(response, "usage", None)
            if not usage:
                return {
                    "cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "model": model
                }
            
            # Calculate cost
            input_cost = (usage.input_tokens / 1_000_000) * model_pricing.get("input", 3.00)
            output_cost = (usage.output_tokens / 1_000_000) * model_pricing.get("output", 15.00)
            
            return {
                "cost": input_cost + output_cost,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "model": model
            }
        except Exception as e:
            logger.error(f"Error calculating cost: {str(e)}", exc_info=True)
            return {
                "cost": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "model": self.model,
                "error": str(e)
            }

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def call_api(self, messages: List[Dict], model: str, **kwargs) -> Dict:
        """Call Anthropic API with retry logic."""
        try:
            # Move system prompt to top-level parameter
            system_prompt = next((m['content'] for m in messages if m['role'] == 'system'), "")
            user_messages = [m for m in messages if m['role'] != 'system']
            
            response = await self.async_client.messages.create(
                system=system_prompt,
                messages=user_messages,
                model=model,
                **kwargs
            )
            self.cost_tracker.track_anthropic(response)
            return response
        except APIError as e:
            logger.error(f"Anthropic API Error: {str(e)}")
            raise

    async def call_cohere_api(self, method: str, **kwargs):
        """ Unified Cohere API handler """
        try:
            # Use async client for Cohere
            client = self.cohere_client
            
            # Match working version's parameter structure
            if method == "embed":
                logger.debug(f"Calling Cohere embed API with method: {method}")
                
                # Log key parameters to debug API call
                input_type = kwargs.get('input_type', 'search_document')
                has_texts = bool(kwargs.get('texts', []))
                has_images = bool(kwargs.get('images', []))
                
                logger.debug(
                    f"Cohere embed params: input_type={input_type}, "
                    f"has_texts={has_texts}, has_images={has_images}"
                )
                
                # Extract key parameters
                texts = kwargs.get('texts', [])
                images = kwargs.get('images', [])
                model = kwargs.get('model', self.cohere_config["model"])
                embedding_types = kwargs.get('embedding_types', ["float"])
                
                # Validate input type against content
                if input_type == "image" and not has_images:
                    logger.error("Input type is 'image' but no images were provided")
                if input_type == "search_document" and not has_texts:
                    logger.error("Input type is 'search_document' but no texts were provided")
                    
                # Make API call
                try:
                    response = client.embed(
                        texts=texts,
                        images=images,
                        model=model,
                        input_type=input_type,
                        embedding_types=embedding_types
                    )
                    
                    # Track usage and cost
                    self.cost_tracker.track_cohere(response, method)
                    
                    # Enhanced response logging
                    logger.debug(f"Cohere API response type: {type(response)}")
                    
                    # Convert response to dictionary if possible
                    if hasattr(response, 'to_dict'):
                        try:
                            response_dict = response.to_dict()
                            logger.debug(f"Response dictionary keys: {list(response_dict.keys())}")
                            return response
                        except Exception as e:
                            logger.warning(f"Failed to convert response to dict: {str(e)}")
                    
                    # Use json serialization as fallback for debugging
                    if hasattr(response, '__dict__'):
                        try:
                            import json
                            response_str = json.dumps(response.__dict__, default=str)
                            logger.debug(f"Response structure: {response_str[:500]}...")
                        except Exception as e:
                            logger.warning(f"Failed to serialize response: {str(e)}")
                    
                    return response
                    
                except Exception as e:
                    logger.error(f"Error during Cohere API call: {str(e)}")
                    raise
                
            raise ValueError(f"Unsupported Cohere method: {method}")
            
        except Exception as e:
            logger.error(f"Cohere API error: {str(e)}")
            raise

    def export_cost_report(self, output_dir: str):
        """Export accumulated cost data"""
        self.cost_tracker.export_report(Path(output_dir))