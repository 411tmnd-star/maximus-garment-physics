"""REQ-03: run one variant (zone A/B extract+spacing, then assemble) entirely inside scratch.
usage: python -I -B run_variant.py <name> <tools_dir_name: tools_pristine|tools_req03> [p2 flags ...] [-- p3 flags ...]"""
import sys, os, shutil, subprocess, time, json
BASE = os.path.normpath(os.path.expanduser('~/Desktop/better_style_execution_workspace/08_KNOWLEDGE/stage3_engineering/opus_requests/REQ-03_bst_factor_isolation'))
SCR = os.path.join(BASE, 'scratch'); PY = os.path.normpath(os.path.expanduser('~/AppData/Local/Programs/Python/Python313/python.exe'))
MASTER = os.path.join(SCR, 'master', 'camp_shirt_master.blend')
name, tdir = sys.argv[1], sys.argv[2]; rest = sys.argv[3:]
p2f, p3f = (rest[:rest.index('--')], rest[rest.index('--') + 1:]) if '--' in rest else (rest, [])
tools = os.path.join(SCR, tdir); out = os.path.join(SCR, 'runs', name); os.makedirs(out, exist_ok=True)
assert out.startswith(SCR) and 'HUMAN_ENGINE' not in out
shutil.copy(os.path.join(SCR, 'out_orig', 'zone_c_stones.npz'), out)          # zone C is not re-solved (it is pruned after A/B)
runner = os.path.join(BASE, 'scripts', 'runner.py'); log = open(os.path.join(BASE, 'logs', f'run_{name}.log'), 'w')
tim = {}
for step, script, flags in (('p2', 'p2_zone_ab.py', p2f), ('p3', 'p3_assemble.py', p3f)):
    t0 = time.time(); log.write(f'### {step} {script} {flags}\n'); log.flush()
    r = subprocess.run([PY, '-I', '-B', runner, tools, os.path.join(tools, script), MASTER, out] + flags, stdout=log, stderr=subprocess.STDOUT, cwd=out)
    tim[step] = round(time.time() - t0, 1); log.write(f'### {step} exit={r.returncode} wall={tim[step]} s\n'); log.flush()
    if r.returncode: break
json.dump(dict(name=name, tools=tdir, p2_flags=p2f, p3_flags=p3f, wall_s=tim, exit=r.returncode), open(os.path.join(out, 'req03_run.json'), 'w'), indent=1)
print(name, tim, 'exit', r.returncode)
