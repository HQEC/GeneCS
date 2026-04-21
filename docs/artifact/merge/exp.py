import subprocess

# Full optimization
commands_1 = [
    "python3 prog.py -f1 BB_72 -f2 Surface_5 -lt1 X -lt2 X -li1 0 -li2 0 -qd1 12 -qd2 12 -cd1 12 -cd2 7 -s 100 -a 5",
    "python3 prog.py -f1 BB_90 -f2 Surface_9 -lt1 X -lt2 X -li1 0 -li2 0 -qd1 12 -qd2 12 -cd1 12 -cd2 9 -s 100 -a 9",
    "python3 prog.py -f1 BB_144 -f2 Surface_11 -lt1 X -lt2 X -li1 0 -li2 0 -qd1 12 -qd2 12 -cd1 12 -cd2 7 -s 100 -a 11",
    "python3 prog.py -f1 BB_72 -f2 Color_5 -lt1 X -lt2 X -li1 0 -li2 0 -qd1 12 -qd2 12 -cd1 12 -cd2 7 -s 100 -a 5",
    "python3 prog.py -f1 BB_90 -f2 Color_9 -lt1 X -lt2 X -li1 0 -li2 0 -qd1 12 -qd2 12 -cd1 12 -cd2 7 -s 100 -a 9",
    "python3 prog.py -f1 BB_144 -f2 Color_11 -lt1 X -lt2 X -li1 0 -li2 0 -qd1 12 -qd2 12 -cd1 12 -cd2 7 -s 100 -a 11",
    "python3 prog.py -f1 HP_1922 -f2 Surface_11 -lt1 X -lt2 X -li1 0 -li2 0 -qd1 12 -qd2 12 -cd1 12 -cd2 7 -s 100 -a 11",
    "python3 prog.py -f1 HP_1922 -f2 Color_11 -lt1 X -lt2 X -li1 0 -li2 0 -qd1 12 -qd2 12 -cd1 12 -cd2 7 -s 100 -a 11",
    "python3 prog.py -f1 QT_135 -f2 Surface_5 -lt1 X -lt2 X -li1 0 -li2 0 -qd1 12 -qd2 12 -cd1 7 -cd2 7 -s 100 -a 4",
    "python3 prog.py -f1 QT_135 -f2 Color_5 -lt1 X -lt2 X -li1 0 -li2 0 -qd1 12 -qd2 12 -cd1 7 -cd2 7 -s 100 -a 4",
]

for cmd in commands_1:
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


# qldpc+qldpc
commands_2 = [
    "python3 prog.py -f1 BB_144 -f2 BB_144 -lt1 X -lt2 X -li1 0 -li2 0 -qd1 12 -qd2 12 -cd1 12 -cd2 12 -s 100 -a 12",
    "python3 prog.py -f1 HP_1922 -f2 HP_1922 -lt1 X -lt2 X -li1 0 -li2 0 -qd1 12 -qd2 12 -cd1 12 -cd2 12 -s 100 -a 12",
    "python3 prog.py -f1 QT_135 -f2 QT_135 -lt1 X -lt2 X -li1 0 -li2 0 -qd1 12 -qd2 12 -cd1 12 -cd2 12 -s 100 -a 4",
    "python3 prog.py -f1 BB_144 -f2 HP_1922 -lt1 X -lt2 X -li1 0 -li2 0 -qd1 12 -qd2 12 -cd1 12 -cd2 12 -s 100 -a 12",
    "python3 prog.py -f1 BB_90 -f2 HP_882_50 -lt1 X -lt2 X -li1 0 -li2 0 -qd1 12 -qd2 12 -cd1 12 -cd2 12 -s 100 -a 10",
]

for cmd in commands_2:
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# multi-qubit
commands_3 = [
    "python3 prog.py -f1 BB_72 -f2 BB_72 -lt1 X -lt2 X -li1 0 1 2 3 4 5 6 7 8 9 10 11 -li2 0 1 2 3 4 5 6 7 8 9 10 11 -qd1 9 -qd2 9 -cd1 9 -cd2 9 -s 100 -a 6",
    "python3 prog.py -f1 BB_90 -f2 BB_90 -lt1 X -lt2 X -li1 0 1 2 3 4 5 6 7 -li2 0 1 2 3 4 5 6 7 -qd1 9 -qd2 9 -cd1 11 -cd2 11 -s 100 -a 10",
    "python3 prog.py -f1 BB_108 -f2 BB_108 -lt1 X -lt2 X -li1 0 1 2 3 4 5 6 7 -li2 0 1 2 3 4 5 6 7 -qd1 9 -qd2 9 -cd1 11 -cd2 11 -s 100 -a 10",
    "python3 prog.py -f1 BB_144 -f2 BB_144 -lt1 X -lt2 X -li1 0 1 2 3 4 5 6 7 8 9 10 11 -li2 0 1 2 3 4 5 6 7 8 9 10 11 -qd1 9 -qd2 9 -cd1 11 -cd2 11 -s 100 -a 12",
    "python3 prog.py -f1 BB_288 -f2 BB_288 -lt1 X -lt2 X -li1 0 1 2 3 4 5 6 7 8 9 10 11 -li2 0 1 2 3 4 5 6 7 8 9 10 11 -qd1 11 -qd2 11 -cd1 12 -cd2 12 -s 100 -a 18",
]

for cmd in commands_3:
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)