p='tools/stones_core.py'; s=open(p).read()
s=s.replace("Nf = unit(N0 + a_ * T + b_ * B)","Nf = unit(N0 - a_ * T - b_ * B)")
a=s.index("def even_positions"); b=s.index("class Grid2")
new = (
'def even_positions(L, pitch, closed, phase=0.0):\n'
'    """arc positions with an integer count. Closed: pitch stretched (never below `pitch`) so the loop closes exactly.\n'
'    Open: run centred; phase 0.5 gives the brick offset against a phase-0 neighbour row."""\n'
'    n = int(math.floor(L / pitch))\n'
'    if n < 1: return np.array([])\n'
'    if closed: p = L / n; return (phase * p + p * np.arange(n)) % L\n'
'    off = 0.5 * (L - n * pitch)\n'
'    return off + pitch * (np.arange(n) + 0.5) if phase else off + pitch * np.arange(n + 1)\n\n\n')
s=s[:a]+new+s[b:]
open(p,'w').write(s); print('patched')
