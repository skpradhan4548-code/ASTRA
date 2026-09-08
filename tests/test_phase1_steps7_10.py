"""
Unit and integration tests for Phase 1 (Steps 7–10):
- File payload & extension security scanner
- Unified ToolResult schema
- Audit logging engine with SHA-256 artifact verification
"""

import json
from pathlib import Path
import pytest

from agent.audit import AuditLogger, ToolResult
from agent.security import SecurityScanner
from agent.tools import WORKSPACE, read_file, write_file


def test_security_scanner_blocked_extensions():
    blocked = ["payload.exe", "script.bat", "run.vbs", "install.ps1", "drop.scr"]
    for fname in blocked:
        valid, err = SecurityScanner.validate_file_path(fname)
        assert not valid
        assert "prohibited by security policy" in err


def test_security_scanner_allowed_extensions():
    allowed = ["data.csv", "report.md", "script.py", "config.json", "output.txt"]
    for fname in allowed:
        valid, err = SecurityScanner.validate_file_path(fname)
        assert valid
        assert err is None


def test_security_scanner_binary_masquerade():
    # Windows PE executable header "MZ"
    fake_text_payload = "MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00"
    valid, err = SecurityScanner.validate_file_content(fake_text_payload)
    assert not valid
    assert "binary payload detected" in err


def test_write_file_blocks_malicious_extension():
    res_raw = write_file("malicious.bat", "echo HACKED")
    res = json.loads(res_raw)
    assert res["success"] is False
    assert res["exit_code"] == 1
    assert "prohibited by security policy" in res["error"]
    assert not (WORKSPACE / "malicious.bat").exists()


def test_write_file_blocks_binary_masquerade():
    res_raw = write_file("innocent.txt", "MZ\x90\x00malicious binary content")
    res = json.loads(res_raw)
    assert res["success"] is False
    assert res["exit_code"] == 1
    assert "binary payload detected" in res["error"]
    assert not (WORKSPACE / "innocent.txt").exists()


def test_audit_logging_on_valid_write():
    filename = "audit_test_report.txt"
    content = "Hello ASTRA Audit Logging Verification 2026"
    res_raw = write_file(filename, content)
    res = json.loads(res_raw)

    assert res["success"] is True
    assert "artifacts" in res
    assert len(res["artifacts"]) == 1
    artifact = res["artifacts"][0]
    assert artifact["path"] == filename
    assert artifact["bytes"] == len(content)
    assert artifact["sha256"] is not None

    # Verify audit.jsonl entry
    audit_file = WORKSPACE / "audit.jsonl"
    assert audit_file.exists()
    lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 0

    latest_record = json.loads(lines[-1])
    assert latest_record["tool"] == "write_file"
    assert latest_record["success"] is True
    assert latest_record["artifacts"][0]["sha256"] == artifact["sha256"]
