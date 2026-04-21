import subprocess

# Full optimization
commands = [
    "python3 prog.py -f Surface_21 -lt X -li 0 -s 100 -qd 12 -cd 12",
    "python3 prog.py -f Surface_41 -lt X -li 0 -s 100 -qd 12 -cd 12",
    "python3 prog.py -f Surface_61 -lt X -li 0 -s 100 -qd 12 -cd 12",
    "python3 prog.py -f Surface_81 -lt X -li 0 -s 100 -qd 12 -cd 12",
    "python3 prog.py -f Surface_101 -lt X -li 0 -s 100 -qd 12 -cd 12",
]

for cmd in commands:
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)