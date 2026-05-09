import importlib.util
import sys
from pathlib import Path


def load_translation_smells_module():
    script = Path(__file__).resolve().parents[2] / "scripts" / "check_translation_smells.py"
    spec = importlib.util.spec_from_file_location("check_translation_smells", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rule_by_id(module, rule_id):
    return next(rule for rule in module.RULES if rule.id == rule_id)


def test_ab13hd_row_daxpy_rule_detects_discrete_row_walk_regression():
    module = load_translation_smells_module()
    rule = rule_by_id(module, "ab13hd-c-row-daxpy-contiguous")

    bad = """
        SLC_DAXPY(&n, &neg_one, &c[i * ldc], &(i32){1},
                  &dwork[ih12 + i1], &(i32){1});
    """

    assert rule.pattern.search(bad)


def test_ab13hd_row_daxpy_rule_ignores_safe_column_walk():
    module = load_translation_smells_module()
    rule = rule_by_id(module, "ab13hd-c-row-daxpy-contiguous")

    safe = """
        SLC_DAXPY(&p, &neg_one, &c[i * ldc], &(i32){1},
                  &dwork[ih12 + i1], &(i32){1});
    """

    assert rule.pattern.search(safe) is None


def test_mb04hd_bwork_rule_detects_bool_cast_regression():
    module = load_translation_smells_module()
    rule = rule_by_id(module, "mb04hd-bwork-bool-cast")

    assert rule.pattern.search("mb03kd(..., (bool *)bwork, ...);")


def test_sb10zd_bwork_rule_detects_i32_cast_regression():
    module = load_translation_smells_module()
    rule = rule_by_id(module, "sb10zd-bwork-i32-cast")

    assert rule.pattern.search("SLC_DGEES(..., (i32*)bwork, &info2);")
