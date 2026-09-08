import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open('Prueba_Tecnica_VoC_NLP_IA_Generativa.ipynb','r',encoding='utf-8') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    ct = cell['cell_type']
    src = ''.join(cell['source'])[:120].replace('\n',' ')
    # strip non-ascii to avoid cp1252 crash
    src_safe = src.encode('ascii','replace').decode('ascii')
    print(f"Cell {i:02d} [{ct}]: {src_safe}")
