import sys,os
from pathlib import Path
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
