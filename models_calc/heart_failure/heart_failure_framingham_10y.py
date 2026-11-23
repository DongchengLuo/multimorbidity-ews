"""Compatibility entry: the implementation is maintained in heart_failure_framingham.py."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_CANONICAL = Path(__file__).with_name("heart_failure_framingham.py")
_SPEC = spec_from_file_location("framingham_chf_canonical", _CANONICAL)
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

calculate_framingham_chf_risk = _MODULE.calculate_framingham_chf_risk
process_data = _MODULE.process_data


if __name__ == "__main__":
    process_data("base.csv", "item_result.csv", "CHF_Framingham_result.csv")
