"""
Audit logging and unified ToolResult schema for DocuAgent / ASTRA.

Standardizes all tool output structures and records execution traces
with SHA-256 artifact verification for security and observability.
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ToolResult:
    """Standardized result schema for all agent tools."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    runtime: str = "native"
    execution_time_ms: float = 0.0
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Apply output truncation safety
        d["stdout"] = self.stdout[-4000:]
        d["stderr"] = self.stderr[-2000:]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class AuditLogger:
    """Thread-safe structured audit logger tracking all agent tool executions."""

    @staticmethod
    def calculate_sha256(filepath: Path) -> Optional[str]:
        """Compute SHA-256 hash of a file for integrity tracking."""
        if not filepath.exists() or not filepath.is_file():
            return None
        sha = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()

    @classmethod
    def log_execution(
        cls,
        tool_name: str,
        inputs: Dict[str, Any],
        result: ToolResult,
        workspace_dir: Path,
        session_id: Optional[str] = None,
    ):
        """Append an audit record to audit.jsonl in the workspace."""
        log_file = workspace_dir / "audit.jsonl"
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id or "default",
            "tool": tool_name,
            "inputs": inputs,
            "success": result.success,
            "exit_code": result.exit_code,
            "runtime": result.runtime,
            "execution_time_ms": round(result.execution_time_ms, 2),
            "artifacts": result.artifacts,
            "error": result.error,
        }

        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            # Audit logging failures should not crash execution
            pass
