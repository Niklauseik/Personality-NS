"""Dimension metadata and opposite preferences used by the DPO runners."""

from typing import Dict

DIMENSION_SPECS: Dict[str, Dict] = {
    "energy": {
        "pretty": "Energy (E/I)",
        "subtypes": [
            {"code": "e", "preferred": "extraversion", "display_name": "E性格模型"},
            {"code": "i", "preferred": "introversion", "display_name": "I性格模型"},
        ],
    },
    "information": {
        "pretty": "Information (S/N)",
        "subtypes": [
            {"code": "s", "preferred": "sensing", "display_name": "S性格模型"},
            {"code": "n", "preferred": "intuition", "display_name": "N性格模型"},
        ],
    },
    "decision": {
        "pretty": "Decision (T/F)",
        "subtypes": [
            {"code": "t", "preferred": "thinking", "display_name": "T性格模型"},
            {"code": "f", "preferred": "feeling", "display_name": "F性格模型"},
        ],
    },
    "execution": {
        "pretty": "Execution (J/P)",
        "subtypes": [
            {"code": "j", "preferred": "judging", "display_name": "J性格模型"},
            {"code": "p", "preferred": "perceiving", "display_name": "P性格模型"},
        ],
    },
}

def get_dimension_spec(name: str) -> Dict:
    key = (name or "").strip().lower()
    if key not in DIMENSION_SPECS:
        raise ValueError(f"Unsupported dimension '{name}'. "
                         f"Allowed values: {', '.join(DIMENSION_SPECS.keys())}")
    return DIMENSION_SPECS[key]

def opposite_preferred_subtype(dimension: str, preferred: str) -> str:
    spec = get_dimension_spec(dimension)
    for subtype in spec["subtypes"]:
        if subtype["preferred"] != preferred:
            return subtype["preferred"]
    raise ValueError(f"Could not find opposite subtype for {dimension}/{preferred}")
