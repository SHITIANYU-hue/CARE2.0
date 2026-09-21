import random


class RandomAgent:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def select(self, view: dict) -> dict:
        seen = {o["candidate_id"] for o in view["observations"]}
        available = [c["candidate_id"] for c in view["candidates"] if c["candidate_id"] not in seen]
        return {"candidate_id": self.rng.choice(available), "diagnostics": {"method": "random"}}
