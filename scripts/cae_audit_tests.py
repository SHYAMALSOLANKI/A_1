import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.stack_server.cae import CAE
from services.stack_server.quality import DynamicQualityController
from services.stack_server.truth import TruthTool


def run():
    dqc = DynamicQualityController()
    cae = CAE(
        dqc_ref=dqc,
        truth_tool=TruthTool(enabled=False),
        config={"enable_grammar": False, "enable_nli": False}
    )

    tests = [
        ("punctuation spam", ",,,,,.....", False, 0.0),
        ("trigram loop", "Pope of the Pope of the Pope of the Pope of the Pope.", False, 0.0),
        ("wiki boilerplate", "References\nOther websites\nExternal links", False, 0.0),
    ]

    for name, text, expected_pass, expected_score in tests:
        result = cae.audit(text)
        assert result["passed"] == expected_pass, f"{name} passed mismatch"
        assert result["score"] == expected_score, f"{name} score mismatch"

    ok_sentence = "This is a simple sentence with minor issues."
    ok_result = cae.audit(ok_sentence)
    assert ok_result["score"] > 0.0, "simple sentence should not hard fail"

    math_result = cae.audit("The result of 1+1 is 3.")
    assert any(v["type"] == "Math Error" for v in math_result["violations"]), "math error missing"

    print("CAE audit tests passed.")


if __name__ == "__main__":
    try:
        run()
    except AssertionError as exc:
        print(f"CAE audit tests failed: {exc}")
        sys.exit(1)
