"""Official EvE shell-step client, copied into its public application snapshot."""

import json
import os
from pathlib import Path
import sys
from urllib.request import Request, urlopen


def main():
    solver_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ["EVE_SOLVER_ROOT"]) / "solver.py"
    request = Request(os.environ["CARE_EVE_EVALUATOR_URL"],
                      data=json.dumps({"solver": solver_path.read_text()}).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=3600) as response:
        payload = json.loads(response.read())
    if "EVE_EVAL_LOG_ROOT" in os.environ:
        output = Path(os.environ["EVE_EVAL_LOG_ROOT"])
        output.mkdir(parents=True, exist_ok=True)
        # JSON is valid YAML; EvE consumes this exact score.yaml path.
        (output / "score.yaml").write_text(json.dumps(payload, indent=2))
        if payload.get("status") == "error":
            (output / "error.txt").write_text(payload["summary"])
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
