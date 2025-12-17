import pandas as pd
import time
from pathlib import Path

TIMING_FILE = Path("processed/timings.csv")

def timeit(func):
    """Decorator to measure execution time and log it."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        timing = pd.DataFrame([{
            "step": func.__name__,
            "time_sec": end - start
        }])
        TIMING_FILE.parent.mkdir(exist_ok=True)
        if TIMING_FILE.exists():
            timing.to_csv(TIMING_FILE, mode='a', header=False, index=False)
        else:
            timing.to_csv(TIMING_FILE, index=False)
        print(f"[{func.__name__}] Time elapsed: {end - start:.2f} sec")
        return result
    return wrapper

def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def load_csv(path: Path, header='infer', names=None):
    if not path.exists():
        return None
    return pd.read_csv(path, header=header, names=names)
