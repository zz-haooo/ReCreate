# Option 1 load from huggingface
import os
from datasets import load_dataset
ds1000 = list(
    load_dataset(
        "xlangai/DS-1000",
        cache_dir=os.getenv("DS1000_CACHE_DIR", None),  # Optional: set DS1000_CACHE_DIR env var for custom cache
    )["test"]
)

# Option 2 load from raw jsonl.gz
# import gzip
# ds1000 = [json.loads(l) for l in gzip.open("data/ds1000.jsonl.gz", "rt").readlines()]