"""Isolated audit-only repair: terminal CSV may end mid-row. No simulation retry."""
from common import *
import types,sys,shutil,traceback

def main():
 out=ROOT/'execution/truncated_log_repair';out.mkdir(exist_ok=True)
 for name in ('status.json','stdout.log','stderr.log'):
  dst=out/('original_'+name)
  if not dst.exists():shutil.copy2(ROOT/'execution'/name,dst)
 source=(HERE/'validation.py').read_text(encoding='utf-8-sig')
 start=source.index(" vv={k:m.array(physics,k)")
 end=source.index(' m.zero_added_loads',start)
 block=source[start:end]
 source=source[:start]+" retained_rows=len(physics)\n valid_rows=[r for r in physics if all(r.get(f'{key}_{i}') not in (None,'') for key in ('ld','qd','fd','v','damping') for i in range(6))]\n if len(valid_rows)!=len(physics):\n  assert not complete_physics,'Complete physical trace cannot contain truncated rows'\n physics=valid_rows\n if physics:\n"+''.join(' '+line+'\n' for line in block.splitlines())+source[end:]
 source=source.replace("retained_physics_samples=len(physics),", "retained_physics_samples=retained_rows,valid_physics_rows=len(physics),")
 source=source.replace('final row retained, whole-prefix physical audit unavailable.', 'terminal file may contain a partial final row; whole-prefix physical audit unavailable.')
 once(out/'repair_protocol.json',dict(change='Only skip incomplete terminal physical CSV rows from formula check, explicitly report retained/valid rows. Original raw files preserved. No full-episode metrics or full audit pass for terminal failures.',parent_sha256=sha(HERE/'validation.py'),source=source,script_sha256=sha(__file__)))
 module=types.ModuleType('validation');module.__file__=str(HERE/'validation.py');exec(compile(source,'de101_terminal_csv_repair','exec'),module.__dict__);sys.modules['validation']=module
 import run
 original_verify=run.verify
 def verify():
  original_verify();assert sha(__file__)==read(out/'repair_protocol.json')['script_sha256']
 run.freeze=lambda:verify()
 run.verify=verify
 run.main()
if __name__=='__main__':
 try:main()
 except Exception:status('engineering_failed',traceback=traceback.format_exc());raise
