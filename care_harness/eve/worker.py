"""Standalone subprocess entry point. Only the public view is passed as input."""

import importlib.util
import json
from pathlib import Path
import random
import sys


def main():
    solver_path, input_path, output_path = map(Path, sys.argv[1:])
    view = json.loads(input_path.read_text())
    seed = (int(view["seed"]) + 1000003 * int(view["round_index"])) % (2 ** 32)
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    spec = importlib.util.spec_from_file_location("evolved_policy", solver_path)
    solver = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = solver
    spec.loader.exec_module(solver)
    decision = solver.select_candidate(view)
    output_path.write_text(json.dumps(decision))


if __name__ == "__main__":
    main()
