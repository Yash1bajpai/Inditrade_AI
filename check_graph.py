import sqlite3, glob, json

db = sqlite3.connect('.agents/graph_memory.sqlite')
files_in_graph = {}
for m in db.execute("SELECT id, properties FROM Nodes WHERE label='File'").fetchall():
    props = json.loads(m[1]) if m[1] else {}
    files_in_graph[m[0]] = props.get('path', '')

all_py_files = glob.glob('src/**/*.py', recursive=True)
missing = []
for f in all_py_files:
    f_normalized = f.replace('\\', '/')
    found = False
    for p in files_in_graph.values():
        if f_normalized in p or p in f_normalized:
            found = True
            break
    if not found and not f_normalized.endswith('__init__.py'):
        missing.append(f_normalized)

print('Missing files in graph:', missing)
