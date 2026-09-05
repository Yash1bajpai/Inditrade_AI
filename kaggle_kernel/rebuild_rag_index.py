"""
Kaggle kernel: rebuild the IndiTrade AI RAG index + retrieval evaluation.

Heavy compute (embedding 5k+ chunks, hybrid eval over 500 questions) runs on
Kaggle, never on the local laptop. Artifacts are zipped into /kaggle/working
as rag_index_v2.zip for download back to the repo.

Steps:
  1. Clone the repo (public, main branch — contains the chunked indexer).
  2. Fetch policy JSONLs from the GitHub data-v1 release.
  3. Run src/rag/indexer.py (GPU-accelerated bge-small embedding).
  4. Run src/rag/evaluate_retrieval.py in dense + hybrid modes.
  5. Zip qdrant_index/, bm25_index.pkl(+.lzma), meta, eval reports.
"""
import os
import subprocess
import sys

WORK = "/kaggle/working"
REPO = f"{WORK}/repo"


def run(cmd, **kwargs):
    print(f"\n$ {cmd}\n{'-' * 70}", flush=True)
    result = subprocess.run(cmd, shell=True, **kwargs)
    if result.returncode != 0:
        # Fail the kernel loudly — a silent COMPLETE with an empty zip cost us
        # one roundtrip already.
        print(f"!! command failed with exit code {result.returncode}", flush=True)
        sys.exit(result.returncode)
    return result


# 1) Environment: pin qdrant-client so the on-disk storage format matches the
#    local machine (qdrant local storage is not guaranteed cross-version).
run("pip install -q 'qdrant-client==1.18.0' rank_bm25 2>&1 | tail -2")

# 2) Code + data
if not os.path.exists(REPO):
    run(f"git clone --depth 1 https://github.com/Yash1bajpai/Inditrade_AI {REPO}")
os.chdir(REPO)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/cache", exist_ok=True)

BASE = "https://github.com/Yash1bajpai/Inditrade_AI/releases/download/data-v1"
for fname in [
    "dgft_policy_chunks.jsonl",
    "dgft_ocr_chunks.jsonl",
    "pib_press_releases.jsonl",
    "policy_qa_dataset.jsonl",
]:
    run(f"curl -fsL --retry 3 {BASE}/{fname} -o data/processed/{fname}")
run("wc -l data/processed/*.jsonl")

# 3) Rebuild index (chunked, GPU embed)
run("python src/rag/indexer.py", stdout=sys.stdout, stderr=sys.stderr)

# 4) Retrieval quality eval on the chunked index
run("python -m src.rag.evaluate_retrieval --mode dense --max-questions 500 "
    "--output reports/rag_eval_dense_chunked.json", stdout=sys.stdout, stderr=sys.stderr)
run("python -m src.rag.evaluate_retrieval --mode hybrid --max-questions 500 "
    "--output reports/rag_eval_hybrid_chunked.json", stdout=sys.stdout, stderr=sys.stderr)

# 5) Package artifacts for download (exclude qdrant lock files)
run("cd .. && zip -rq rag_index_v2.zip "
    "repo/data/cache/qdrant_index "
    "repo/data/cache/bm25_index.pkl "
    "repo/data/cache/bm25_index.pkl.lzma "
    "repo/data/cache/rag_index_meta.json "
    "repo/reports "
    "-x '*.lock'", stdout=sys.stdout, stderr=sys.stderr)
run(f"ls -la {WORK} && mv {WORK}/rag_index_v2.zip {WORK}/rag_index_v2.zip 2>/dev/null; echo DONE")

print("\n=== KERNEL COMPLETE: rag_index_v2.zip ready in output ===", flush=True)
