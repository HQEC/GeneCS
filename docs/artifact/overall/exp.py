import subprocess

# Full optimization
commands_1 = [
    "python3 prog.py -f BB_72 -lt X -li 0 -s 100 -qd 12 -cd 12",
    "python3 prog.py -f BB_90 -lt X -li 0 -s 100 -qd 12 -cd 12",
    "python3 prog.py -f BB_108 -lt X -li 0 -s 100 -qd 12 -cd 12",
    "python3 prog.py -f BB_144 -lt X -li 0 -s 100 -qd 12 -cd 12",
    "python3 prog.py -f BB_288 -lt X -li 0 -s 100 -qd 12 -cd 12",
    "python3 prog.py -f HP_450 -lt X -li 0 -s 100 -qd 12 -cd 12",
    "python3 prog.py -f HP_882_50 -lt X -li 0 -s 100 -qd 12 -cd 12",
    "python3 prog.py -f HP_882_98 -lt X -li 0 -s 100 -qd 12 -cd 12",
    "python3 prog.py -f HP_1800 -lt X -li 0 -s 100 -qd 12 -cd 12",
    "python3 prog.py -f HP_1922 -lt X -li 0 -s 100 -qd 12 -cd 12",
    "python3 prog.py -f QT_81 -lt X -li 0 -s 100 -qd 12 -cd 7",
    "python3 prog.py -f QT_90 -lt X -li 0 -s 100 -qd 12 -cd 7",
    "python3 prog.py -f QT_117 -lt X -li 0 -s 100 -qd 12 -cd 7",
    "python3 prog.py -f QT_126 -lt X -li 0 -s 100 -qd 12 -cd 7",
    "python3 prog.py -f QT_135 -lt X -li 0 -s 100 -qd 12 -cd 7",
    "python3 prog.py -f Surface_5 -lt X -li 0 -s 100 -qd 12 -cd 7",
    "python3 prog.py -f Surface_7 -lt X -li 0 -s 100 -qd 12 -cd 7",
    "python3 prog.py -f Surface_9 -lt X -li 0 -s 100 -qd 12 -cd 7",
    "python3 prog.py -f Surface_11 -lt X -li 0 -s 100 -qd 12 -cd 7",
    "python3 prog.py -f Surface_13 -lt X -li 0 -s 100 -qd 12 -cd 9",
    "python3 prog.py -f Color_5 -lt X -li 0 -s 100 -qd 12 -cd 7",
    "python3 prog.py -f Color_7 -lt X -li 0 -s 100 -qd 12 -cd 7",
    "python3 prog.py -f Color_9 -lt X -li 0 -s 100 -qd 12 -cd 7",
    "python3 prog.py -f Color_11 -lt X -li 0 -s 100 -qd 12 -cd 7",
    "python3 prog.py -f Color_13 -lt X -li 0 -s 100 -qd 12 -cd 7",
]

for cmd in commands_1:
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)