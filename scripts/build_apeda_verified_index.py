"""Add exactly the four approved audited APEDA documents to an existing sparse corpus.
No scraping, cloud writes, release upload or deployment. Defaults to candidate output.
"""
import argparse,copy,hashlib,json,lzma,pickle,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rank_bm25 import BM25Okapi
from src.rag.sparse_index import tokenize_for_bm25
APPROVED={'APEDA_NOTIFICATION_'+n+'_2026_27' for n in ('41','36','34','35')}

def build(base, candidate, audit, output):
    original=pickle.load(lzma.open(base,'rb'))
    rows=[json.loads(l) for l in Path(candidate).read_text().splitlines() if l.strip()]
    entries={x['chunk_id']:x for x in json.loads(Path(audit).read_text())['entries']}
    if len(rows)!=4 or {r['chunk_id'] for r in rows}!=APPROVED:
        raise ValueError('Only the four approved documents are allowed')
    docs=copy.deepcopy(original['docs']);added=[]
    for row in rows:
        e=entries[row['chunk_id']]
        if not (row['pdf_sha256']==e['pdf_sha256'] and hashlib.sha256(row['clean_text'].encode()).hexdigest()==e['transcription_sha256'] and e['second_pass']['verified'] is True and row['verification']=='visually_verified_with_logged_corrections'):
            raise ValueError('Unverified candidate, hash mismatch, or audit drift')
        if any(d.get('parent_doc_id')==row['chunk_id'] for d in docs):
            raise ValueError('Document already indexed')
        # The four checked one-page documents stay whole, preserving clauses and exceptions.
        # No truncation: query contexts receive the full text of the selected parent.
        payload={**row,'doc_id':row['chunk_id']+'::c0','parent_doc_id':row['chunk_id'],
                 'chunk_index':0,'num_chunks':1,'doc_type':'APEDA_Mirror_Agent_Checked',
                 'title':row['subject'],'ministry':'DGFT via APEDA, delayed partial mirror, agent-checked',
                 'text':row['clean_text'],'char_length':len(row['clean_text'])}
        added.append(payload)
    docs+=added
    result={'docs':docs,'bm25':BM25Okapi([tokenize_for_bm25(x['text']) for x in docs])}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    with lzma.open(output,'wb') as f:pickle.dump(result,f)
    return original['docs'],added,result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',default='data/cache/bm25_index.pkl.lzma');p.add_argument('--output',default='/tmp/apeda_sparse_candidate.pkl.lzma');a=p.parse_args()
    old,new,_=build(a.base,'data/reviewed/apeda_verified4_candidate.jsonl','data/reviewed/apeda_vetted.json',a.output)
    print(f'Preserved {len(old)} chunks, added {len(new)} complete audited documents -> {a.output}')
