import sys
import json
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm  # Ensure you have run: pip install tqdm

# --- PATH CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent 

# --- IMPORT HELPERS (Resilient Loading) ---
def try_import_transformers():
    try:
        from transformers import pipeline
        import torch  # Need torch to check for GPU availability
        return pipeline, torch
    except Exception:
        return None, None

def try_import_vader():
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
        return nltk, SentimentIntensityAnalyzer
    except Exception:
        return None, None

def try_import_textblob():
    try:
        from textblob import TextBlob
        return TextBlob
    except Exception:
        return None

# --- ANALYSIS ENGINES ---

# Update only this function in your code
def analyze_with_transformers(pipeline_fn, torch_mod, texts: List[str], batch_size: int = 64) -> List[Tuple[str, float]]:
    # 1. Explicitly detect device
    if torch_mod.cuda.is_available():
        device_type = "cuda:0"
        device_name = f"NVIDIA GPU ({torch_mod.cuda.get_device_name(0)})"
    elif hasattr(torch_mod.backends, "mps") and torch_mod.backends.mps.is_available():
        device_type = "mps"
        device_name = "Apple Silicon GPU"
    else:
        device_type = "cpu"
        device_name = "CPU"
    
    print(f"--- Logic: {device_name} selected ---")
    
    # 2. Fix 'No model supplied' warning by explicitly naming a model
    # 'distilbert-base-uncased-finetuned-sst-2-english' is the industry standard for fast sentiment
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    
    pipe = pipeline_fn(
        "sentiment-analysis", 
        model=model_name, 
        device=device_type
    )
    
    results = []
    
    # 3. Use the pipeline as an iterator to fix the 'sequential' warning
    # This is the most efficient way to use a GPU in Transformers
    for out in tqdm(pipe(texts, batch_size=batch_size), total=len(texts), desc="Transformers", unit="row"):
        label = out.get("label", "NEUTRAL")
        score = float(out.get("score", 0.0))
        results.append((label.upper(), score))
        
    return results

def analyze_with_vader(nltk_module, VaderClass, texts: List[str]) -> List[Tuple[str, float]]:
    try:
        nltk_module.download("vader_lexicon", quiet=True)
    except Exception:
        pass
    analyzer = VaderClass()
    out = []
    
    for t in tqdm(texts, desc="VADER Progress", unit="row"):
        if not isinstance(t, str) or not t.strip():
            out.append(("NEUTRAL", 0.0))
            continue
        scores = analyzer.polarity_scores(t)
        compound = float(scores.get("compound", 0.0))
        lbl = "POSITIVE" if compound >= 0.05 else ("NEGATIVE" if compound <= -0.05 else "NEUTRAL")
        out.append((lbl, compound))
    return out

def analyze_with_textblob(TextBlobClass, texts: List[str]) -> List[Tuple[str, float]]:
    out = []
    for t in tqdm(texts, desc="TextBlob Progress", unit="row"):
        if not isinstance(t, str) or not t.strip():
            out.append(("NEUTRAL", 0.0))
            continue
        tb = TextBlobClass(t)
        polarity = float(tb.sentiment.polarity)
        lbl = "POSITIVE" if polarity > 0 else ("NEGATIVE" if polarity < 0 else "NEUTRAL")
        out.append((lbl, polarity))
    return out

# --- CORE LOGIC ---
def compute_title_sentiment(df: pd.DataFrame, title_col: str = "title") -> pd.DataFrame:
    texts = df[title_col].fillna("").astype(str).tolist()
    res = None

    # 1. Try Transformers (GPU optimized)
    pipeline_fn, torch_mod = try_import_transformers()
    if pipeline_fn and torch_mod:
        try:
            res = analyze_with_transformers(pipeline_fn, torch_mod, texts)
        except Exception as e:
            print(f"Transformers failed ({e}), falling back to VADER...")
            res = None

    # 2. Try VADER Fallback
    if res is None:
        nltk_mod, v_class = try_import_vader()
        if nltk_mod and v_class:
            try:
                res = analyze_with_vader(nltk_mod, v_class, texts)
            except Exception:
                res = None

    # 3. Try TextBlob Fallback
    if res is None:
        tb_class = try_import_textblob()
        if tb_class:
            try:
                res = analyze_with_textblob(tb_class, texts)
            except Exception:
                res = None

    # 4. Final Trivial Fallback
    if res is None:
        print("All engines failed. Using trivial fallback.")
        res = [("NEUTRAL" if not t.strip() else "POSITIVE", 0.0) for t in texts]

    labels, scores = zip(*res)
    df["title_sentiment"] = list(labels)
    df["title_sentiment_score"] = list(scores)
    return df

def main(csv_path: Optional[Path] = None, out_path: Optional[Path] = None):
    if csv_path is None:
        csv_path = PROJECT_ROOT / "data" / "dataset_B_news_subset_3500.csv"
    if out_path is None:
        out_path = PROJECT_ROOT / "data" / "updated_dataset_b.csv"

    if not csv_path.exists():
        print(f"Error: Input CSV not found at {csv_path}")
        sys.exit(1)

    print(f"Reading: {csv_path.name}")
    df = pd.read_csv(csv_path)

    title_col = next((c for c in ["title", "Title"] if c in df.columns), None)
    if not title_col:
        print("Error: No title column found.")
        sys.exit(1)

    df = compute_title_sentiment(df, title_col=title_col)
    
    df.to_csv(out_path, index=False)
    print(f"\nSuccess! Saved to: {out_path}")

    # Final Sanity Check Printout
    print("-" * 30)
    print("Sentiment Distribution Summary:")
    print(df["title_sentiment"].value_counts())
    print("-" * 30)
    

if __name__ == "__main__":
    in_csv_arg = Path(sys.argv[1]) if len(sys.argv) >= 2 else None
    out_csv_arg = Path(sys.argv[2]) if len(sys.argv) >= 3 else None

    if not in_csv_arg:
        config_path = PROJECT_ROOT / "config" / "default_inputs.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    in_val = cfg.get("in_csv") or cfg.get("input_path")
                    out_val = cfg.get("out_csv") or cfg.get("output_path")
                    if in_val: in_csv_arg = Path(in_val)
                    if out_val: out_csv_arg = Path(out_val)
            except Exception as e:
                print(f"Warning: Failed to parse config: {e}")

    main(in_csv_arg, out_csv_arg)


