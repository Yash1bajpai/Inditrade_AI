import json
import os
import tempfile

def read_jsonl(filepath):
    """Read a JSONL file and return a list of dictionaries."""
    if not os.path.exists(filepath):
        print("Dataset not found!")
        return None

    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_jsonl(filepath, rows):
    """Atomically write a list of dictionaries to a JSONL file.

    Writes to a temp file in the same directory, flushes and fsyncs it, then
    os.replace()s it over the target. This matters because callers such as
    clean_ocr_answers.py rewrite policy_qa_dataset.jsonl in place: a crash
    partway through a plain "w" write would leave the dataset truncated with no
    way to recover it.
    """
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=directory,
        prefix=os.path.basename(filepath) + ".",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, filepath)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
