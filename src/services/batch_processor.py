import json
import logging
import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

from .api_service import APIService, APIRateLimiter
from .attribute_generator import AttributeGenerator

logger = logging.getLogger(__name__)

class ProcessingStats:
    def __init__(self):
        self.successful_products = {}  # product_id -> processing time
        self.failed_products = {}  # product_id -> error reason
        self.total_tokens = 0
        self.total_time = 0
        
    def add_success(self, product_id: str, processing_time: float, tokens: int):
        self.successful_products[product_id] = processing_time
        self.total_tokens += tokens
        self.total_time += processing_time
        logger.info(f"✓ Successfully processed product {product_id} in {processing_time:.2f}s (Tokens: {tokens})")
        
    def add_failure(self, product_id: str, error_reason: str):
        self.failed_products[product_id] = error_reason
        logger.error(f"✗ Failed to process product {product_id}: {error_reason}")
        
    def print_summary(self):
        total_products = len(self.successful_products) + len(self.failed_products)
        success_rate = (len(self.successful_products)/total_products)*100 if total_products > 0 else 0
        avg_time = self.total_time/max(1, len(self.successful_products))
        
        summary = [
            "\nProcessing Summary:",
            "=" * 50,
            f"\nSuccessfully Processed Products ({len(self.successful_products)}):",
            *[f"✓ Product {pid}: {time:.2f}s" for pid, time in self.successful_products.items()],
            f"\nFailed Products ({len(self.failed_products)}):",
            *[f"✗ Product {pid}: {reason}" for pid, reason in self.failed_products.items()],
            "\nOverall Statistics:",
            f"Total Products Attempted: {total_products}",
            f"Success Rate: {success_rate:.1f}%",
            f"Total Processing Time: {self.total_time:.1f}s",
            f"Average Time Per Success: {avg_time:.1f}s",
            f"Total Tokens Used: {self.total_tokens}",
            "=" * 50
        ]
        
        summary_text = "\n".join(summary)
        logger.info(summary_text)
        print(summary_text)

class IncrementalSaver:
    def __init__(self, json_path: Path):
        self.json_path = json_path
        self.buffer = []
        self.processed_products = self._load_processed_products()
        
    def _load_processed_products(self) -> set:
        processed = set()
        
        if self.json_path.exists():
            try:
                with open(self.json_path, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        return processed
                        
                    # Split content into individual JSON objects
                    current_obj = ""
                    brace_count = 0
                    
                    for char in content:
                        current_obj += char
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            
                        if brace_count == 0 and current_obj.strip():
                            try:
                                data = json.loads(current_obj)
                                processed.update(data.keys())
                                current_obj = ""
                            except json.JSONDecodeError:
                                continue
                            
            except Exception as e:
                logger.error(f"Error reading JSON file: {str(e)}")
        
        if processed:
            logger.info(f"Loaded {len(processed)} processed products")
        return processed

    def save(self, product_id: str, result: dict):
        if product_id in self.processed_products:
            return
            
        # Convert Timestamp to string for JSON serialization
        if 'processed_at' in result:
            result['processed_at'] = str(result['processed_at'])
            
        self.buffer.append((product_id, result))
        self.processed_products.add(product_id)
        
        # Always flush after adding to ensure immediate save
        self._flush_buffer()
    
    def _flush_buffer(self):
        if not self.buffer:
            return
            
        # Read existing content
        existing_data = {}
        if self.json_path.exists():
            try:
                with open(self.json_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        current_obj = ""
                        brace_count = 0
                        
                        for char in content:
                            current_obj += char
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                
                            if brace_count == 0 and current_obj.strip():
                                try:
                                    data = json.loads(current_obj)
                                    existing_data.update(data)
                                    current_obj = ""
                                except json.JSONDecodeError:
                                    continue
                                    
            except Exception as e:
                logger.error(f"Error reading existing JSON data: {str(e)}")
        
        # Update with new data
        for product_id, result in self.buffer:
            existing_data[product_id] = result
            
        # Write all data back
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.json_path, 'w') as f:
            for i, (product_id, data) in enumerate(existing_data.items()):
                if i > 0:
                    f.write('\n')
                json.dump({product_id: data}, f)
                
        self.buffer = []

class BatchProcessor:
    def __init__(
        self,
        api_service: APIService,
        attribute_generator: AttributeGenerator,
        output_dir: Path,
        batch_size: int = 10
    ):
        self.api_service = api_service
        self.attribute_generator = attribute_generator
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        
        # Initialize saver for each product type
        self.savers = {}
        
    def _get_saver(self, product_type: str) -> IncrementalSaver:
        if product_type not in self.savers:
            json_path = self.output_dir / f"{product_type}_attributes.json"
            self.savers[product_type] = IncrementalSaver(json_path)
        return self.savers[product_type]

    async def process_product(
        self,
        product_data: Dict[str, Any],
        product_type: str,
        image_path: Optional[str] = None,
        rate_limiter: Optional[APIRateLimiter] = None
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Process a single product with optional rate limiting"""
        if rate_limiter:
            async with rate_limiter.semaphore:
                await rate_limiter.wait_if_needed()
                return await self._process_product_internal(
                    product_data,
                    product_type,
                    image_path
                )
        return await self._process_product_internal(
            product_data,
            product_type,
            image_path
        )

    async def _process_product_internal(
        self,
        product_data: Dict[str, Any],
        product_type: str,
        image_path: Optional[str] = None
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        try:
            product_id = str(product_data['product_id'])
            logger.info(f"Starting processing for product {product_id}")
            
            start_time = time.time()
            
            # Generate attributes
            attributes = await self.attribute_generator.generate_attributes(
                product_data,
                product_type,
                image_path
            )
            
            if not attributes:
                return product_id, None
                
            # Add metadata
            attributes.update({
                'processed_at': datetime.now(),
                'inference_time': time.time() - start_time
            })
            
            return product_id, attributes
            
        except Exception as e:
            logger.error(
                f"Error processing product {product_data.get('product_id', 'unknown')}: {str(e)}", 
                exc_info=True
            )
            return str(product_data.get('product_id', 'unknown')), None

    async def batch_process(
        self,
        df: pd.DataFrame,
        product_type: str,
        image_path_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process products in batches"""
        saver = self._get_saver(product_type)
        stats = ProcessingStats()
        
        # Filter already processed products
        df = df[~df['product_id'].astype(str).isin(saver.processed_products)]
        total_products = len(df)
        total_batches = (total_products + self.batch_size - 1) // self.batch_size
        
        if total_products == 0:
            logger.info("No new products to process")
            return {}
            
        logger.info(f"Starting processing of {total_products} products in {total_batches} batches")
        
        results = {}
        rows = df.to_dict('records')
        
        for batch_num in range(0, total_products, self.batch_size):
            start_time = time.time()
            batch = rows[batch_num:batch_num + self.batch_size]
            
            # Process batch concurrently
            tasks = [
                self.process_product(
                    row,
                    product_type,
                    row.get(image_path_column) if image_path_column else None,
                    self.api_service.rate_limiter
                )
                for row in batch
            ]
            batch_results = await asyncio.gather(*tasks)
            
            for product_id, result in batch_results:
                if result:
                    results[product_id] = result
                    processing_time = result['inference_time']
                    tokens = result.get('input_tokens', 0) + result.get('output_tokens', 0)
                    stats.add_success(product_id, processing_time, tokens)
                    saver.save(product_id, result)
                else:
                    stats.add_failure(product_id, "No result returned")
            
            # Dynamic delay based on batch processing time
            batch_time = time.time() - start_time
            if batch_num < total_products - self.batch_size:
                delay = 1 if batch_time < 1 else 0.5
                await asyncio.sleep(delay)
            
            logger.info(f"Batch {batch_num // self.batch_size + 1} completed in {time.time() - start_time:.1f}s")
        
        stats.print_summary()
        return results 