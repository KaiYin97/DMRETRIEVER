## DMRetriever/code/eva/utils/config.py
"""Global configuration."""

from pathlib import Path
import os


DATA_ROOT           = Path("DMRetriever/data")
C_TEST_SET_DIR      = DATA_ROOT / "C_test_set"
E_TEST_RES_DIR      = DATA_ROOT / "E_test_res"
D_TRAIN_OUTPUT_DIR  = DATA_ROOT / "D_train_output"  

CORPUS_DIR          = C_TEST_SET_DIR / "corpus"
TEST_QUERY_EVA_DIR  = C_TEST_SET_DIR / "test_query_eva"
TEST_QUERY_TEST_DIR = C_TEST_SET_DIR / "test_query_test"

TEST_QUERY_DIR      = TEST_QUERY_EVA_DIR

# Output/Cache roots
BASELINE_INDEX_DIR  = E_TEST_RES_DIR / "corpus_embeddings"
LABEL_POOL_DIR      = E_TEST_RES_DIR / "label_pools"
QUERY_EMB_DIR       = E_TEST_RES_DIR / "query_embeddings"
PERF_ROOT           = E_TEST_RES_DIR / "performance"

# Misc data
QRELS_ROOT          = C_TEST_SET_DIR / "qrels_with_added"
CHECKPOINT_ROOT     = D_TRAIN_OUTPUT_DIR
E_TEST_ROOT         = E_TEST_RES_DIR  # optional root alias


# Ordered corpus JSONs
ORDERED_CORPUS_EVA_JSON  = CORPUS_DIR / "ordered_corpus_eva.json"
ORDERED_CORPUS_FULL_JSON = CORPUS_DIR / "ordered_corpus_full.json"

# Query dirs per split
EVA_TEST_DIR = {
    "eva":  TEST_QUERY_EVA_DIR,
    "test": TEST_QUERY_TEST_DIR,
}

# Corpus config per split
EVA_TEST_CFG = {
    "eva": {
        "CORPUS_JSON": ORDERED_CORPUS_EVA_JSON,
        "CORPUS_BASE": CORPUS_DIR,
    },
    "test": {
        "CORPUS_JSON": ORDERED_CORPUS_FULL_JSON,
        "CORPUS_BASE": CORPUS_DIR,
    },
}


TASK2PREFIX = {
    "FactCheck": "Given the claim, retrieve most relevant document that supports or refutes the claim",
    "NLI":       "Given the premise, retrieve most relevant hypothesis that is entailed by the premise",
    "QA":        "Given the question, retrieve most relevant passage that best answers the question",
    "QAdoc":     "Given the question, retrieve the most relevant document that answers the question",
    "STS":       "Given the sentence, retrieve the sentence with the same meaning",
    "Twitter":   "Given the user query, retrieve the most relevant Twitter text that meets the request"
}

DEFAULT_BATCH   = 4
DEFAULT_MAXLEN  = 512
DEFAULT_TOPK    = 10
USE_FP16        = True
