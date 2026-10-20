import json,lzma,pickle
from pathlib import Path
import pytest
from scripts.build_apeda_verified_index import build
from src.rag.sparse_index import ProdBM25Index

CANDIDATE='data/reviewed/apeda_verified4_candidate.jsonl'
AUDIT='data/reviewed/apeda_vetted.json'

def test_approved_addition_preserves_every_existing_chunk(tmp_path):
    out=tmp_path/'candidate.xz'
    base=tmp_path/'before.xz'
    existing=pickle.load(lzma.open('data/cache/bm25_index.pkl.lzma','rb'))
    existing['docs']=[x for x in existing['docs'] if not x.get('doc_id','').startswith('APEDA_NOTIFICATION_')]
    with lzma.open(base,'wb') as f:pickle.dump(existing,f)
    old,new,result=build(base,CANDIDATE,AUDIT,out)
    assert result['docs'][:len(old)]==old
    assert len(new)==4 and {x['notification_no'] for x in new}=={'41/2026-27','36/2026-27','34/2026-27','35/2026-27'}
    assert all(x['verification']=='visually_verified_with_logged_corrections' and x['source_note'] and x['url'].startswith('https://apeda.gov.in/') for x in new)
    assert 'exceeding ₹' in next(x['text'] for x in new if x['notification_no']=='36/2026-27')
    index=ProdBM25Index.load(str(out))
    for query,n in [('RoDTEP September December 2026','41/2026-27'),('RCMC FOB 3,00,000','36/2026-27'),('Wheat Flour 1101 0000','34/2026-27'),('Durum Wheat 10011900 10019910','35/2026-27')]:
        assert any(h.payload['notification_no']==n and h.payload.get('source')=='APEDA_MIRROR_OF_DGFT' for h in index.search(query,limit=8))

def test_unreviewed_or_tampered_text_is_rejected(tmp_path):
    rows=[json.loads(l) for l in Path(CANDIDATE).read_text().splitlines()]
    rows[0]['clean_text']+=' altered'
    candidate=tmp_path/'rows.jsonl';candidate.write_text('\n'.join(json.dumps(x) for x in rows))
    with pytest.raises(ValueError):build('data/cache/bm25_index.pkl.lzma',candidate,AUDIT,tmp_path/'out.xz')
    rows=rows[:3];candidate.write_text('\n'.join(json.dumps(x) for x in rows))
    with pytest.raises(ValueError):build('data/cache/bm25_index.pkl.lzma',candidate,AUDIT,tmp_path/'out.xz')
