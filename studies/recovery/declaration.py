"""The declaration of the recovery pair (pair.json) and the trainings that a program of the study is given, read
alike by scene.py, readings.py, exposure.py and outcome.py."""
import json
from pathlib import Path

KEYS = {"question", "trainings", "saves", "actor_key", "control_dt_s", "placed_starts", "first_rows", "fall_recovery",
        "quantities", "margins", "classes", "outcomes", "unreadable_when"}


def read_declaration(path):
    """pair.json; a declaration with another set of top-level keys is refused."""
    declaration = json.loads(Path(path).read_text())
    if set(declaration) != KEYS:
        raise ValueError(f"{path}: keys {sorted(declaration)}, expected {sorted(KEYS)}")
    return declaration


def training_roots(declaration, items):
    """{(condition, seed): run directory} from --training CONDITION:SEED=RUN arguments; a condition or seed the
    declaration does not name, and a training given twice, are refused."""
    roots = {}
    for item in items:
        name, root = item.split("=", 1)
        condition, seed = name.split(":")
        key = condition, int(seed)
        if condition not in declaration["trainings"]["conditions"] or key[1] not in declaration["trainings"]["seeds"]:
            raise ValueError(f"--training {item}: {condition} seed {seed} is not a training of the declaration")
        if key in roots:
            raise ValueError(f"--training {item}: {condition} seed {seed} is given twice")
        roots[key] = Path(root)
    return roots
