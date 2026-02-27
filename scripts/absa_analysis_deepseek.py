"""
DeepSeek LLM-based ABSA for Vietnamese Reviews - Optimized Version
Phân tích Aspect-Based Sentiment Analysis tối ưu với DeepSeek API
"""

import pandas as pd
import json
import time
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import hashlib
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# DeepSeek API Configuration
DEEPSEEK_API_KEYS = [
    "sk-4f7ea1d492d7437dbf9f72aa84c7f041",  # 👈 Dán API key của bạn vào đây (xóa text này)
    # Có thể thêm nhiều keys để backup khi hết quota
]

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"  # Model tốt nhất cho chat/analysis

# Aspects to analyze (simplified)
ASPECTS = ["service", "staff", "quality", "facility", "cleanliness", "price", "ambiance", "food"]

class DeepSeekABSA:
    def __init__(self, input_file: str, output_file: str, checkpoint_file: str = "absa_checkpoint.txt"):
        self.input_file = input_file
        self.output_file = output_file
        self.checkpoint_file = checkpoint_file
        self.cache_file = output_file.replace('.csv', '_cache.json')
        
        # API management
        self.current_api_key_index = 0
        self.client = self._create_client()
        self.lock = threading.Lock()
        
        # Performance settings
        self.batch_size = 5  # Xử lý 5 reviews/lần để tối ưu
        self.max_workers = 3  # 3 threads song song
        self.request_delay = 0.5  # 500ms giữa các request
        self.last_request_time = time.time()
        
        # Cache management
        self.cache = self._load_cache()
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _create_client(self) -> OpenAI:
        """Tạo OpenAI client cho DeepSeek"""
        return OpenAI(
            api_key=DEEPSEEK_API_KEYS[self.current_api_key_index],
            base_url=DEEPSEEK_BASE_URL
        )
    
    def _switch_api_key(self):
        """Chuyển sang API key khác khi cần"""
        with self.lock:
            self.current_api_key_index = (self.current_api_key_index + 1) % len(DEEPSEEK_API_KEYS)
            self.client = self._create_client()
            print(f"\n🔑 Switched to API key #{self.current_api_key_index + 1}")
    
    def _load_cache(self) -> Dict:
        """Load cache từ file"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Lưu cache"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Cache save error: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """Tạo cache key từ review text"""
        # Handle NaN/None/float values
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def create_optimized_prompt(self, reviews: List[str]) -> str:
        """
        Tạo prompt tối ưu cho batch reviews
        Ngắn gọn, rõ ràng, tiết kiệm token
        """
        
        # Format reviews với index
        reviews_text = "\n".join([f"{i+1}. {r[:500]}" for i, r in enumerate(reviews)])  # Limit 500 chars/review
        
        prompt = f"""Analyze sentiment for these reviews. For each review, rate 8 aspects: service, staff, quality, facility, cleanliness, price, ambiance, food.

Reviews:
{reviews_text}

Output JSON array (one object per review):
[
  {{
    "service": {{"m": 1, "s": "positive", "c": 0.9}},
    "staff": {{"m": 0, "s": null, "c": null}},
    ...
  }}
]

Rules:
- m: mentioned (1=yes, 0=no)
- s: sentiment ("positive"/"neutral"/"negative" if mentioned, null otherwise)
- c: confidence (0-1 float if mentioned, null otherwise)
- High confidence (0.8-1.0): explicit words (great, terrible, excellent, awful)
- Medium (0.5-0.8): clear opinion (good, bad, nice, dirty)
- Low (0.3-0.5): vague (ok, fine)

Return ONLY the JSON array, no extra text."""
        
        return prompt
    
    def analyze_batch(self, reviews: List[Tuple[int, str]], retry_count: int = 3) -> List[Optional[Dict]]:
        """
        Phân tích batch reviews cùng lúc - tiết kiệm API calls
        """
        
        # Check cache first
        results = []
        uncached_reviews = []
        uncached_indices = []
        
        for idx, review_text in reviews:
            cache_key = self._get_cache_key(review_text)
            if cache_key in self.cache:
                results.append((idx, self.cache[cache_key]))
                self.cache_hits += 1
            else:
                uncached_reviews.append(review_text)
                uncached_indices.append(idx)
                results.append((idx, None))
                self.cache_misses += 1
        
        # Nếu tất cả đều có trong cache
        if not uncached_reviews:
            return [r[1] for r in sorted(results, key=lambda x: x[0])]
        
        # Call API for uncached reviews
        for attempt in range(retry_count):
            try:
                # Rate limiting
                with self.lock:
                    elapsed = time.time() - self.last_request_time
                    if elapsed < self.request_delay:
                        time.sleep(self.request_delay - elapsed)
                    self.last_request_time = time.time()
                
                # API call
                prompt = self.create_optimized_prompt(uncached_reviews)
                
                response = self.client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a sentiment analysis expert. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistency
                    max_tokens=2000,
                    response_format={"type": "json_object"} if len(uncached_reviews) == 1 else None
                )
                
                # Parse response
                response_text = response.choices[0].message.content.strip()
                
                # Clean response
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                # Parse JSON
                parsed = json.loads(response_text)
                
                # Handle single vs batch response
                if isinstance(parsed, dict):
                    # Single review response
                    batch_results = [parsed]
                elif isinstance(parsed, list):
                    batch_results = parsed
                else:
                    raise ValueError("Unexpected response format")
                
                # Validate and cache results
                if len(batch_results) == len(uncached_reviews):
                    for i, (review_idx, review_text) in enumerate(zip(uncached_indices, uncached_reviews)):
                        result = batch_results[i]
                        # Validate result structure
                        if self._validate_result(result):
                            cache_key = self._get_cache_key(review_text)
                            self.cache[cache_key] = result
                            
                            # Update results list
                            for j, (idx, _) in enumerate(results):
                                if idx == review_idx:
                                    results[j] = (idx, result)
                                    break
                    
                    # Save cache periodically
                    if len(self.cache) % 20 == 0:
                        self._save_cache()
                    
                    # Return sorted results
                    return [r[1] for r in sorted(results, key=lambda x: x[0])]
                else:
                    print(f"⚠️ Result count mismatch: expected {len(uncached_reviews)}, got {len(batch_results)}")
                    
            except Exception as e:
                error_msg = str(e).lower()
                
                if "rate_limit" in error_msg or "429" in error_msg:
                    wait_time = 2 ** attempt
                    print(f"⚠️ Rate limit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif "quota" in error_msg or "insufficient" in error_msg:
                    print(f"⚠️ Quota exceeded, switching API key...")
                    self._switch_api_key()
                    time.sleep(5)
                elif "invalid" in error_msg and "key" in error_msg:
                    print(f"❌ Invalid API key, switching...")
                    self._switch_api_key()
                    time.sleep(2)
                else:
                    print(f"❌ Error (attempt {attempt + 1}/{retry_count}): {e}")
                    time.sleep(1)
        
        # Return whatever we have (with Nones for failed ones)
        return [r[1] for r in sorted(results, key=lambda x: x[0])]
    
    def _validate_result(self, result: Dict) -> bool:
        """Validate ABSA result structure"""
        if not isinstance(result, dict):
            return False
        
        for aspect in ASPECTS:
            if aspect not in result:
                return False
            
            aspect_data = result[aspect]
            if not isinstance(aspect_data, dict):
                return False
            
            # Check required fields
            if 'm' not in aspect_data:
                return False
            
            mentioned = aspect_data['m']
            if mentioned == 1:
                if 's' not in aspect_data or 'c' not in aspect_data:
                    return False
                if aspect_data['s'] not in ['positive', 'neutral', 'negative']:
                    return False
        
        return True
    
    def convert_to_columns(self, absa_result: Dict) -> Dict:
        """Convert compact ABSA result to column format"""
        columns = {}
        
        for aspect in ASPECTS:
            aspect_data = absa_result.get(aspect, {})
            mentioned = aspect_data.get('m', 0)
            
            if mentioned == 0:
                columns[f"deepseek_aspect_{aspect}_sentiment"] = None
                columns[f"deepseek_aspect_{aspect}_confidence"] = None
            else:
                columns[f"deepseek_aspect_{aspect}_sentiment"] = aspect_data.get('s')
                columns[f"deepseek_aspect_{aspect}_confidence"] = aspect_data.get('c')
        
        return columns
    
    def get_last_processed_index(self) -> int:
        """Get last processed row index from checkpoint"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return int(f.read().strip())
            except:
                return -1
        return -1
    
    def save_checkpoint(self, index: int):
        """Save checkpoint"""
        with open(self.checkpoint_file, 'w') as f:
            f.write(str(index))
    
    def process_dataset(self, sample_size: Optional[int] = None):
        """Process entire dataset with batch optimization"""
        
        print("="*80)
        print("DEEPSEEK LLM-BASED ABSA - OPTIMIZED VERSION")
        print("="*80)
        
        # Load data
        df = pd.read_csv(self.input_file, encoding='utf-8')
        print(f"\n✓ Loaded {len(df)} reviews")
        
        # Check for existing output
        last_index = self.get_last_processed_index()
        
        if sample_size:
            print(f"\n🧪 Testing mode: Processing first {sample_size} rows")
            last_index = -1
            df = df.head(sample_size)
            if os.path.exists(self.output_file):
                try:
                    os.remove(self.output_file)
                except:
                    pass
        elif last_index >= 0:
            print(f"\n📍 Resuming from row {last_index + 1}")
        else:
            print(f"\n🆕 Starting fresh analysis")
            if os.path.exists(self.output_file):
                os.remove(self.output_file)
        
        # Process in batches
        start_idx = last_index + 1 if last_index >= 0 else 0
        total = len(df)
        
        print(f"\n⏳ Processing rows {start_idx} to {total-1}")
        print(f"📦 Batch size: {self.batch_size} | Workers: {self.max_workers}")
        print(f"💾 Cache: {len(self.cache)} entries loaded\n")
        
        results = []
        
        for batch_start in range(start_idx, total, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total)
            batch_rows = df.iloc[batch_start:batch_end]
            
            # Progress
            progress = (batch_end / total) * 100
            print(f"[{batch_end}/{total}] ({progress:.1f}%) Processing batch {batch_start}-{batch_end-1}...", end=" ")
            
            # Prepare batch - skip NaN/empty text
            batch_reviews = []
            for i, row in batch_rows.iterrows():
                text = row['text']
                # Convert NaN/None to empty string
                if not isinstance(text, str) or pd.isna(text):
                    text = "No review text provided."
                batch_reviews.append((i, text))
            
            # Analyze batch
            start_time = time.time()
            batch_results = self.analyze_batch(batch_reviews)
            elapsed = time.time() - start_time
            
            # Process results
            success_count = 0
            for (idx, row), absa_result in zip(batch_reviews, batch_results):
                result_row = df.loc[idx].to_dict()
                
                if absa_result:
                    columns = self.convert_to_columns(absa_result)
                    result_row.update(columns)
                    success_count += 1
                else:
                    # Failed - add None values
                    for aspect in ASPECTS:
                        result_row[f"deepseek_aspect_{aspect}_sentiment"] = None
                        result_row[f"deepseek_aspect_{aspect}_confidence"] = None
                
                results.append(result_row)
            
            print(f"✓ ({success_count}/{len(batch_reviews)}) in {elapsed:.1f}s")
            
            # Save batch
            self.save_batch(results)
            self.save_checkpoint(batch_end - 1)
            results = []
        
        # Final save
        self._save_cache()
        
        print("\n" + "="*80)
        print("✅ COMPLETED!")
        print(f"📊 Cache stats: {self.cache_hits} hits, {self.cache_misses} misses")
        print(f"💰 Cache hit rate: {self.cache_hits/(self.cache_hits+self.cache_misses)*100:.1f}%")
        print(f"📁 Results: {self.output_file}")
        print(f"💾 Cache: {self.cache_file}")
        print("="*80)
    
    def save_batch(self, batch_results: List[Dict]):
        """Save batch results to CSV"""
        if not batch_results:
            return
        
        df_batch = pd.DataFrame(batch_results)
        
        if os.path.exists(self.output_file):
            df_batch.to_csv(self.output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            df_batch.to_csv(self.output_file, index=False, encoding='utf-8-sig')

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="DeepSeek LLM-based ABSA - Optimized")
    parser.add_argument("--input", default="cleaned_reviews.csv", help="Input CSV file")
    parser.add_argument("--output", default="sentiment9/absa_deepseek_results.csv", help="Output CSV file")
    parser.add_argument("--sample", type=int, default=None, help="Process only first N rows")
    parser.add_argument("--checkpoint", default="sentiment9/absa_deepseek_checkpoint.txt", help="Checkpoint file")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size (default: 5)")
    parser.add_argument("--workers", type=int, default=3, help="Max workers (default: 3)")

    args = parser.parse_args()
    
    # Create analyzer
    analyzer = DeepSeekABSA(
        input_file=args.input,
        output_file=args.output,
        checkpoint_file=args.checkpoint
    )
    
    # Override settings if provided
    if args.batch_size:
        analyzer.batch_size = args.batch_size
    if args.workers:
        analyzer.max_workers = args.workers
    
    # Process
    analyzer.process_dataset(sample_size=args.sample)

if __name__ == "__main__":
    main()
