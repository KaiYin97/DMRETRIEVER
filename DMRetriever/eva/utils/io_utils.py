# -*- coding: utf-8 -*-

import json
import glob
import os
from typing import List
from pathlib import Path


def build_ordered_corpus(corpus_dir: str, out_fp: str) -> List[str]:
    """Build an ordered text corpus from multiple JSON files."""
    files = sorted(glob.glob(os.path.join(corpus_dir, "*.json")))
    corpus: List[str] = []
    for fp in files:
        arr = json.load(open(fp, encoding="utf-8"))
        for txt in arr:
            txt = txt.strip()
            if txt:
                corpus.append(txt)
    Path(out_fp).parent.mkdir(parents=True, exist_ok=True)
    with open(out_fp, "w", encoding="utf-8") as wf:
        json.dump(corpus, wf, ensure_ascii=False)
    print(f"✓ built ordered corpus: {out_fp} ({len(corpus):,d} passages)")
    return corpus


def load_ordered_corpus(corpus_json: str) -> List[str]:
    """Load corpus built by build_ordered_corpus with fixed order."""
    return json.load(open(corpus_json, encoding="utf-8"))


def load_test_file(path: str):
    """Load test file and extract queries."""
    data = json.load(open(path, encoding="utf-8"))
    queries = [item["user_query"].strip() for item in data]
    return data, queries
