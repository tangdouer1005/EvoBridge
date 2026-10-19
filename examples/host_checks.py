"""Illustration of caller-owned checks, not an environment or benchmark adapter.

Register real behavioral checks in your private environment for executable
operations. A syntax check alone must not be marked behavioral=True.
"""
from evobridge import CheckResult, CheckSuite, RegisteredCheck


def policy_links(candidate):
    # An application-specific invariant, independent of task success scores.
    ok = "memory.md" in candidate.read("CONTEXT.md")
    return CheckResult("policy_links", ok, "Execution policy should reference domain memory.")


def make_checks():
    return CheckSuite({"policy_links": RegisteredCheck(policy_links)})
