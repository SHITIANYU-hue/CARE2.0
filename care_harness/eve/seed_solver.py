"""Non-evolved seed policy: nearest-observation estimate plus distance exploration.

This is a starting point for the official EvE search, not a reported EvE result.
The policy is stateless; every call receives the complete allowed observation history.
"""

import math


def select_candidate(view):
    candidates = {row["candidate_id"]: row for row in view["candidates"]}
    observations = view["observations"]
    observed = {row["candidate_id"] for row in observations}
    remaining = [row for key, row in candidates.items() if key not in observed]
    values = [row["outcome"] for row in observations]
    scale = max(values) - min(values) or 1.0

    def score(candidate):
        distances = []
        for observation in observations:
            reference = candidates[observation["candidate_id"]]
            distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(
                candidate["numeric_features"], reference["numeric_features"])))
            distances.append((distance, observation["outcome"]))
        distance, predicted = min(distances)
        return predicted + 0.1 * scale * distance / (1.0 + distance)

    return max(remaining, key=score)["candidate_id"]
