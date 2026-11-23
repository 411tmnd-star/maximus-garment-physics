"""REQ-03 runner: run a tool script from a given tools dir under `python -I -B` (isolated mode drops the script dir from sys.path)."""
import sys, runpy
tools, script = sys.argv[1], sys.argv[2]
sys.path.insert(0, tools)
sys.argv = [script] + sys.argv[3:]
runpy.run_path(script, run_name='__main__')
