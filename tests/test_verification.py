"""Step 3 tests: evidence verification decides SURE vs THINK."""

import pytest

from core.verification import (
    FOUND, MISSING, SURE, THINK, TOO_SHORT, check_quote, normalise, statement, verify_output,
)

# Fake collected data, as it would be shown to the model.
CORPUS = """\
IP address     MAC address        Vendor
192.168.1.1    a4:2b:b0:11:22:33  TP-LINK TECHNOLOGIES
192.168.1.20   00:1b:a9:44:55:66  Brother Industries
Gateway: 192.168.1.1
2024-05-01 03:12:44 ERROR  disk /dev/sda1 is 97% full
"""
NORM = normalise(CORPUS)


def finding(evidence, basis="observed", confidence="high", claim="Something is true"):
    return {"claim": claim, "evidence": evidence, "confidence": confidence, "basis": basis}


def run(*findings):
    return verify_output({"summary": "s", "findings": list(findings)}, CORPUS)


# --- matching single quotes -------------------------------------------------

@pytest.mark.parametrize("quote", [
    "192.168.1.20",
    "brother industries",                        # case ignored
    "192.168.1.20    00:1b:a9:44:55:66",          # extra spaces ignored
    "IP address MAC address",                    # newline/space differences ignored
    '"Gateway: 192.168.1.1"',                    # wrapping quote marks ignored
    "disk /dev/sda1 is 97% full",
    "ERROR — disk".replace(" — ", "  "),  # sanity: plain match
    "Brother​ Industries",                  # invisible character removed
    "２０２４-05-01",                              # full-width digits normalised
    "192.168.1.1 ... Brother Industries",        # ellipsis: pieces in order
])
def test_quotes_that_are_in_the_data(quote):
    assert check_quote(quote, NORM) == FOUND


@pytest.mark.parametrize("quote", [
    "192.168.1.2",                    # must not match inside 192.168.1.20
    "168.1.20",                       # a fragment of 192.168.1.20
    "a4:2b:b0:11:22:3",               # a fragment of a MAC address
    "05-01 03:12",                    # a fragment of a date
    "Netgear",                        # simply not there
    "disk /dev/sda1 is 98% full",     # one character changed
    "Brother Industries ... TP-LINK",  # pieces in the wrong order
])
def test_quotes_that_are_not_in_the_data(quote):
    assert check_quote(quote, NORM) == MISSING


@pytest.mark.parametrize("quote", ["1", "ok", "  ", "...", "a ... 192.168.1.1"])
def test_tiny_quotes_prove_nothing(quote):
    assert check_quote(quote, NORM) == TOO_SHORT


# --- verdicts ---------------------------------------------------------------

def test_observed_with_found_evidence_is_sure():
    f = run(finding(["192.168.1.1", "Gateway: 192.168.1.1"]))["findings"][0]
    v = f["verification"]
    assert v["verdict"] == SURE and v["evidence_verified"]
    assert f["basis"] == "observed" and f["confidence"] == "high"   # unchanged


def test_inferred_is_never_sure_even_with_good_evidence():
    f = run(finding(["Brother Industries"], basis="inferred"))["findings"][0]
    assert f["verification"]["verdict"] == THINK
    assert "inference" in " ".join(f["verification"]["notes"])


def test_low_confidence_observed_is_think():
    f = run(finding(["192.168.1.1"], confidence="low"))["findings"][0]
    assert f["verification"]["verdict"] == THINK


def test_made_up_evidence_is_downgraded():
    f = run(finding(["192.168.1.1", "Netgear R7000"]))["findings"][0]
    v = f["verification"]
    assert v["verdict"] == THINK and not v["evidence_verified"]
    assert f["basis"] == "inferred" and f["confidence"] == "low"
    assert v["model_said"] == {"basis": "observed", "confidence": "high"}   # history kept
    assert [c["status"] for c in v["evidence_checks"]] == [FOUND, MISSING]
    assert any("downgraded" in n for n in v["notes"])


def test_summary_counts():
    out = run(
        finding(["192.168.1.1"]),                         # SURE
        finding(["Brother Industries"], basis="inferred"),  # THINK, verified
        finding(["invented text here"]),                  # THINK, unverified
    )
    assert out["verification_summary"] == {
        "total": 3, "sure": 1, "think": 2, "unverified_evidence": 1,
    }


def test_input_is_not_modified():
    original = {"summary": "s", "findings": [finding(["nope nope"])]}
    verify_output(original, CORPUS)
    assert original["findings"][0]["basis"] == "observed"
    assert "verification" not in original["findings"][0]


def test_empty_findings_list_is_fine():
    assert verify_output({"summary": "s", "findings": []}, CORPUS)["verification_summary"]["total"] == 0


# --- statements -------------------------------------------------------------

def test_statements_read_naturally():
    assert statement("The gateway is 192.168.1.1.", SURE) == "I'm sure that the gateway is 192.168.1.1."
    assert statement("Device 20 is a printer", THINK) == "I think device 20 is a printer, but I'm not sure."
    assert statement("IP 192.168.1.1 is the router", SURE) == "I'm sure that IP 192.168.1.1 is the router."
