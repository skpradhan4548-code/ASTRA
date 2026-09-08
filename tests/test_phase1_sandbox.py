"""
Unit and security tests for Phase 1 (Steps 1–3):
- Container & Process Sandboxing
- Network Isolation policy
- Safe path resolution & session isolation
"""

import json
import pytest
from pathlib import Path

from agent.sandbox import DockerSandbox, NetworkPolicy, ProcessSandbox, SandboxManager
from agent.tools import WORKSPACE, _safe_path, run_python, write_file, read_file


def test_safe_path_valid():
    p = _safe_path("data.txt")
    assert p == (WORKSPACE / "data.txt").resolve()
    assert str(p).startswith(str(WORKSPACE))


def test_safe_path_traversal_blocked():
    with pytest.raises(ValueError, match="escapes the workspace sandbox"):
        _safe_path("../outside.txt")

    with pytest.raises(ValueError, match="escapes the workspace sandbox"):
        _safe_path("foo/../../outside.txt")


def test_safe_path_null_byte():
    with pytest.raises(ValueError, match="Invalid file path"):
        _safe_path("safe.txt\x00malicious.txt")


def test_safe_path_session_isolation():
    session_id = "test-session-42"
    p = _safe_path("report.txt", session_id=session_id)
    expected_dir = WORKSPACE / "sessions" / session_id
    assert p.parent == expected_dir.resolve()
    assert p.name == "report.txt"

    # Session escape attempt
    with pytest.raises(ValueError, match="escapes the workspace sandbox"):
        _safe_path("../../other_session/secret.txt", session_id=session_id)


def test_process_sandbox_execution():
    box = ProcessSandbox()
    res = box.run(
        code="print('ASTRA_SANDBOX_OK')",
        workspace_dir=WORKSPACE,
        timeout_seconds=5,
        network_policy=NetworkPolicy.ISOLATED,
    )
    assert res.exit_code == 0
    assert "ASTRA_SANDBOX_OK" in res.stdout
    assert res.runtime == "process_sandbox"
    assert res.network_policy == "isolated"
    assert not res.timed_out


def test_process_sandbox_timeout():
    box = ProcessSandbox()
    res = box.run(
        code="import time; time.sleep(10)",
        workspace_dir=WORKSPACE,
        timeout_seconds=1,
        network_policy=NetworkPolicy.ISOLATED,
    )
    assert res.timed_out is True
    assert res.exit_code == -1
    assert "exceeded timeout" in res.error


def test_run_python_tool():
    raw = run_python("print(7 * 6)")
    data = json.loads(raw)
    assert data["exit_code"] == 0
    assert data["stdout"].strip() == "42"
    assert data["runtime"] in ("docker", "process_sandbox")
    assert data["network_policy"] == "isolated"


def test_session_isolated_workflow():
    session_id = "agent_run_alpha"
    # Write a file inside session
    w_res = json.loads(write_file("output.txt", "secret-payload-123", session_id=session_id))
    assert "written" in w_res

    # Read the file inside same session
    content = read_file("output.txt", session_id=session_id)
    assert content == "secret-payload-123"

    # Root workspace or other session should not see it as output.txt directly
    root_content = read_file("output.txt")
    assert "not found in workspace" in root_content
