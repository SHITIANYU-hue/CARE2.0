from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
ORIGINAL=ROOT/"experiments/care_replay/protocols/suzuki_second_dataset_v1"
sys.path.insert(0,str(ROOT/"experiments/care_replay/scripts"))
sys.path.insert(0,str(ORIGINAL))
import common, generation, receipts
