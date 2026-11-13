"""Verify source manifest, English text and excluded artifact types."""
import ast
import hashlib
import json
import re
from pathlib import Path

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'provenance/files.sha256.json').read_text())
for name, expected in manifest.items():
    path=root/name
    assert path.is_file(), name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected, name
    assert path.suffix.lower() not in {'.npz','.npy','.pt','.pth','.csv','.log','.dll','.so','.exe'}, name
    text=path.read_text(encoding='utf-8-sig')
    assert not re.search(r'[\u4e00-\u9fff]',text), name
    if path.suffix=='.py': ast.parse(text,filename=name)
print(f'Verified {len(manifest)} source/config/documentation files')
