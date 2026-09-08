"""
Unit tests for Phase 1 (Steps 4–6):
- ProcessWatchdog process tree termination
- Sandboxed shell execution (run_shell) with security blocklists
- Dynamic venv manager and package installation validation
"""

import json
import os
import subprocess
import sys
import time
import pytest
from pathlib import Path

from agent.sandbox import ProcessWatchdog, VenvManager
from agent.tools import WORKSPACE, run_shell, install_package


def test_process_watchdog_kills_process():
    # Spawn a sleeping background process
    py_bin = sys.executable
    proc = subprocess.Popen(
        [py_bin, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    pid = proc.pid
    assert proc.poll() is None

    # Kill process tree via watchdog
    ProcessWatchdog.kill_process_tree(pid)
    time.sleep(0.5)

    # Process should no longer be running
    assert proc.poll() is not None


def test_run_shell_basic_command():
    cmd = 'python -c "print(\'HELLO_SHELL\')"'
    res_json = run_shell(cmd)
    res = json.loads(res_json)
    assert res["exit_code"] == 0
    assert "HELLO_SHELL" in res["stdout"]
    assert res["runtime"] == "process_sandbox"


def test_run_shell_blocks_dangerous_command():
    # Test blocked patterns
    cmd = "rm -rf /"
    res_json = run_shell(cmd)
    res = json.loads(res_json)
    assert res["exit_code"] == 126
    assert "Security violation" in res["stderr"]

    cmd_win = "format c: /q"
    res_json2 = run_shell(cmd_win)
    res2 = json.loads(res_json2)
    assert res2["exit_code"] == 126
    assert "Security violation" in res2["stderr"]


def test_run_shell_timeout():
    cmd = 'python -c "import time; time.sleep(10)"'
    res_json = run_shell(cmd, timeout_seconds=1)
    res = json.loads(res_json)
    assert res["timed_out"] is True
    assert res["exit_code"] == -1
    assert "exceeded timeout" in res["error"]


def test_run_shell_session_isolation():
    session_id = "shell_session_1"
    # Create file via shell in session
    cmd = 'python -c "open(\'session_marker.txt\', \'w\').write(\'created_in_session\')"'
    res_json = run_shell(cmd, session_id=session_id)
    res = json.loads(res_json)
    assert res["exit_code"] == 0

    session_file = WORKSPACE / "sessions" / session_id / "session_marker.txt"
    assert session_file.exists()
    assert session_file.read_text() == "created_in_session"

    # Root workspace shouldn't have session_marker.txt directly
    assert not (WORKSPACE / "session_marker.txt").exists()


def test_install_package_validation():
    # Invalid package name with injection attempt
    res_json = install_package("pkg; rm -rf /")
    res = json.loads(res_json)
    # Sanitizer or validation handles this safely
    assert res["exit_code"] != 0 or "Invalid package" in str(res)
