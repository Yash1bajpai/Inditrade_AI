import os, time, tarfile, tempfile, hashlib, json, requests
from qdrant_client import QdrantClient
from qdrant_client.models import Document, VectorParams, Distance, PointStruct

URL = os.environ["QDRANT_URL"].rstrip("/")
KEY = os.environ["QDRANT_API_KEY"]
TAR = "/tmp/rag-dense-index.tar.gz"
EXPECTED_SHA = "fd8a1fb23a2a00f8a30c1cdddf46dfa845dd781d88f735709d767746298a8957"
SERVING = "trade_policy_e5_small_v1"      # name the live backend queries
STAGING = "trade_policy_e5_small_v1_stg"  # built and validated before any switch
BACKUP  = "trade_policy_e5_small_v1_bak"  # exact vector copy of serving; survives serving deletion
SCRATCH = "restore_path_probe_tmp"
MODEL = "intfloat/multilingual-e5-small"  # Qdrant-hosted FREE model (docs: cost Free, no token charges)
HDR = {"api-key": KEY}

sha = hashlib.sha256(open(TAR, "rb").read()).hexdigest()
print("tarball sha256:", sha)
assert sha == EXPECTED_SHA, "artifact mismatch - abort before any mutation"

tmp = tempfile.mkdtemp()
with tarfile.open(TAR) as t:
    t.extractall(tmp, filter="data")
local = QdrantClient(path=os.path.join(tmp, "qdrant_index"))
payloads, offset = [], None
while True:
    pts, offset = local.scroll("trade_policy_compliance", limit=1000, offset=offset,
                               with_payload=True, with_vectors=False)
    payloads.extend(p.payload for p in pts)
    if offset is None: break
assert len(payloads) == 6353, f"expected 6353, got {len(payloads)}"
print("payloads:", len(payloads))

cloud = QdrantClient(url=URL, api_key=KEY, cloud_inference=True, timeout=180)
cols = [c.name for c in cloud.get_collections().collections]
print("collections:", cols)

# ACTUAL capacity evidence (free tier: 1 GB RAM / 4 GB disk, no card on account).
r = requests.get(f"{URL}/telemetry?details_level=4", headers=HDR, timeout=60)
assert r.status_code == 200, f"telemetry unavailable (HTTP {r.status_code}) - abort before any mutation"
tel = r.json()
open("/tmp/telemetry.json", "w").write(json.dumps(tel, indent=2))
def find_keys(o, needle, path=""):
    hits = []
    if isinstance(o, dict):
        for k, v in o.items():
            if needle in str(k).lower():
                hits.append((path + "/" + str(k), v))
            hits += find_keys(v, needle, path + "/" + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o[:8]):
            hits += find_keys(v, needle, f"{path}[{i}]")
    return hits
for p, v in find_keys(tel, "disk")[:12]:
    print("telemetry:", p, "=", str(v)[:120])
for p, v in find_keys(tel, "memory")[:6]:
    print("telemetry:", p, "=", str(v)[:120])
print("telemetry captured (see log); per-collection points:")
total_points = 0
for name in cols:
    n = cloud.get_collection(name).points_count
    total_points += n
    print(f"  {name}: {n} points")

def copy_collection(src, dst):
    """Exact vector+payload copy, no re-embedding."""
    if dst in [c.name for c in cloud.get_collections().collections]:
        cloud.delete_collection(dst)
    cloud.create_collection(dst, vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    offset, copied = None, 0
    while True:
        pts, offset = cloud.scroll(src, limit=256, offset=offset, with_payload=True, with_vectors=True)
        if pts:
            cloud.upsert(dst, points=[PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in pts], wait=True)
            copied += len(pts)
        if offset is None: break
    return copied

# 1) PROVE the copy-based rollback route BEFORE touching serving (scratch round-trip).
cloud.create_collection(SCRATCH, vectors_config=VectorParams(size=384, distance=Distance.COSINE))
cloud.upsert(SCRATCH, points=[PointStruct(id=1, vector=Document(text="restore probe", model=MODEL),
                                          payload={"k": "v"})], wait=True)
n = copy_collection(SCRATCH, SCRATCH + "_copy")
assert n == 1 and cloud.get_collection(SCRATCH + "_copy").points_count == 1, "COPY ROUTE BROKEN - abort, serving untouched"
cloud.delete_collection(SCRATCH); cloud.delete_collection(SCRATCH + "_copy")
print("rollback route PROVEN (exact vector copy into independent collection)")

# 2) BACKUP serving into an independent collection (survives serving deletion; nothing leaves the cluster).
serving_pre = cloud.get_collection(SERVING).points_count if SERVING in cols else 0
print("serving pre-state points:", serving_pre)
copied = copy_collection(SERVING, BACKUP)
assert copied == serving_pre and cloud.get_collection(BACKUP).points_count == serving_pre, "backup copy incomplete - abort"
# spot-check vector equality on one point
a = cloud.scroll(SERVING, limit=1, with_vectors=True)[0][0]
b = cloud.retrieve(BACKUP, [a.id], with_vectors=True)[0]
assert a.vector == b.vector and a.payload == b.payload, "backup fidelity check failed - abort"
print("backup collection verified:", copied, "points, vector+payload identical")

# 3) Build STAGING fully; serving keeps answering throughout.
if STAGING in cols:
    cloud.delete_collection(STAGING)
cloud.create_collection(STAGING, vectors_config=VectorParams(size=384, distance=Distance.COSINE))
B, t0 = 32, time.time()
for i in range(0, len(payloads), B):
    batch = payloads[i:i+B]
    cloud.upsert(STAGING, points=[{"id": i+j, "vector": Document(text=p["text"], model=MODEL),
                                   "payload": p} for j, p in enumerate(batch)], wait=True)
    if (i//B) % 25 == 0:
        print(f"staged {i+len(batch)}/{len(payloads)} ({time.time()-t0:.0f}s)")
assert cloud.get_collection(STAGING).points_count == 6353, "staging count mismatch"

# 4) Validate STAGING before any switch.
for q in ("agarwood", "What is the latest export policy for Agarwood?"):
    hits = cloud.query_points(STAGING, query=Document(text=q, model=MODEL), limit=5, with_payload=True).points
    print(f"staging probe [{q}]:")
    for h in hits:
        print("  ", h.payload.get("parent_doc_id"), "|", h.payload.get("date"), "|", round(h.score, 4))
top = cloud.query_points(STAGING, query=Document(text="agarwood", model=MODEL), limit=3, with_payload=True).points
assert any((h.payload.get("notification_no") or "").startswith("46/2026") or "agar" in (h.payload.get("title") or "").lower()
           for h in top), "staging validation failed - abort, serving untouched"
print("staging VALIDATED")

# 5) Switch with instant rollback: alias serving NAME to staging; rollback = alias to BACKUP.
try:
    cloud.delete_collection(SERVING)
    try:
        cloud.delete_alias(SERVING)
    except Exception:
        pass
    cloud.create_alias(SERVING, STAGING)
    chk = cloud.query_points(SERVING, query=Document(text="agarwood", model=MODEL), limit=3, with_payload=True).points
    assert chk, "serving-name query empty after switch"
    print("alias", SERVING, "->", STAGING, "| serving-name probe:",
          [(h.payload.get("parent_doc_id"), h.payload.get("date")) for h in chk])
except Exception as e:
    print("SWITCH FAILED:", str(e)[:200])
    try:
        cloud.delete_alias(SERVING)
    except Exception:
        pass
    cloud.create_alias(SERVING, BACKUP)
    chk = cloud.query_points(SERVING, query=Document(text="agarwood", model=MODEL), limit=1, with_payload=True).points
    print("ROLLED BACK:", SERVING, "->", BACKUP, "| probe points:", len(chk))
    raise
print("SYNC OK - serving name now reads refreshed corpus; backup collection retained:", BACKUP)
