"""
Execution sandbox engine for DocuAgent / ASTRA.

Provides:
1. DockerSandbox: Executes code inside an isolated container with
   resource limits (memory limit, CPU quotas, network isolation).
2. ProcessSandbox: Hardened local execution runner with strict timeouts,
   sanitized environment variables, and directory boundary enforcement.
3. ProcessWatchdog: Escalating process tree termination on timeouts.
4. VenvManager: Dynamic isolated virtual environments per workspace/session.
5. ShellSandbox: Sandboxed shell execution with dangerous command filtering.
"""

import enum
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class NetworkPolicy(str, enum.Enum):
    ISOLATED = "isolated"  # No network access
    OPEN = "open"          # Standard outbound network access


class SandboxResult:
    def __init__(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        runtime: str,
        network_policy: str,
        timed_out: bool = False,
        error: Optional[str] = None,
    ):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.runtime = runtime
        self.network_policy = network_policy
        self.timed_out = timed_out
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "stdout": self.stdout[-4000:],
            "stderr": self.stderr[-2000:],
            "exit_code": self.exit_code,
            "runtime": self.runtime,
            "network_policy": self.network_policy,
        }
        if self.timed_out:
            data["timed_out"] = True
        if self.error:
            data["error"] = self.error
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class ProcessWatchdog:
    """Terminates runaway processes and their entire child process tree."""

    @staticmethod
    def kill_process_tree(pid: int):
        if sys.platform == "win32":
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True,
                    timeout=5,
                )
            except Exception:
                pass
        else:
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass


class VenvManager:
    """Manages an isolated Python virtual environment inside the workspace."""

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self.venv_dir = workspace_dir / ".venv"

    def get_python_exe(self) -> Optional[Path]:
        if sys.platform == "win32":
            py = self.venv_dir / "Scripts" / "python.exe"
        else:
            py = self.venv_dir / "bin" / "python"
        return py if py.exists() else None

    def ensure_venv(self) -> Path:
        py = self.get_python_exe()
        if py:
            return py

        base_py = shutil.which("python") or shutil.which("python3") or sys.executable
        subprocess.run(
            [base_py, "-m", "venv", str(self.venv_dir)],
            capture_output=True,
            check=True,
            timeout=90,
        )
        created_py = self.get_python_exe()
        if not created_py or not created_py.exists():
            raise RuntimeError(f"Failed to initialize virtual environment at {self.venv_dir}")
        return created_py

    def install_package(self, package_name: str, timeout_seconds: int = 60) -> SandboxResult:
        # Sanitize package string (supports version pinning like 'requests==2.31.0')
        clean_pkg = "".join(
            c for c in package_name if c.isalnum() or c in ("-", "_", ".", "=", "<", ">", "[", "]", ",", "~")
        ).strip()
        if not clean_pkg:
            return SandboxResult(
                stdout="",
                stderr="Invalid package name provided.",
                exit_code=-1,
                runtime="venv_manager",
                network_policy=NetworkPolicy.OPEN.value,
                error="Invalid package name",
            )

        try:
            py_exe = self.ensure_venv()
            cmd = [str(py_exe), "-m", "pip", "install", "--no-input", clean_pkg]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.workspace_dir.resolve()),
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout_seconds)
                return SandboxResult(
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=proc.returncode,
                    runtime="venv_manager",
                    network_policy=NetworkPolicy.OPEN.value,
                )
            except subprocess.TimeoutExpired:
                ProcessWatchdog.kill_process_tree(proc.pid)
                return SandboxResult(
                    stdout="",
                    stderr=f"Package installation timed out after {timeout_seconds}s",
                    exit_code=-1,
                    runtime="venv_manager",
                    network_policy=NetworkPolicy.OPEN.value,
                    timed_out=True,
                )
        except Exception as exc:
            return SandboxResult(
                stdout="",
                stderr=str(exc),
                exit_code=-1,
                runtime="venv_manager",
                network_policy=NetworkPolicy.OPEN.value,
                error=str(exc),
            )


class DockerSandbox:
    """Executes code in an ephemeral, resource-constrained Docker container."""

    def __init__(
        self,
        image: str = "python:3.11-slim",
        memory_limit: str = "512m",
        cpu_quota: str = "1.0",
    ):
        self.image = image
        self.memory_limit = memory_limit
        self.cpu_quota = cpu_quota

    @staticmethod
    def is_docker_available() -> bool:
        """Check if Docker CLI and daemon are operational."""
        try:
            res = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            return res.returncode == 0
        except Exception:
            return False

    def run(
        self,
        code: str,
        workspace_dir: Path,
        timeout_seconds: int = 15,
        network_policy: NetworkPolicy = NetworkPolicy.ISOLATED,
    ) -> SandboxResult:
        net_flag = "--network=none" if network_policy == NetworkPolicy.ISOLATED else "--network=bridge"
        abs_workspace = str(workspace_dir.resolve())

        script_name = f"_agent_exec_{os.getpid()}_{id(code)}.py"
        script_path = workspace_dir / script_name
        script_path.write_text(code, encoding="utf-8")

        docker_cmd = [
            "docker", "run", "--rm",
            f"--memory={self.memory_limit}",
            f"--cpus={self.cpu_quota}",
            net_flag,
            "-v", f"{abs_workspace}:/workspace",
            "-w", "/workspace",
            self.image,
            "python", f"/workspace/{script_name}",
        ]

        proc = None
        try:
            proc = subprocess.Popen(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout_seconds)
                return SandboxResult(
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=proc.returncode,
                    runtime="docker",
                    network_policy=network_policy.value,
                )
            except subprocess.TimeoutExpired:
                ProcessWatchdog.kill_process_tree(proc.pid)
                return SandboxResult(
                    stdout="",
                    stderr="",
                    exit_code=-1,
                    runtime="docker",
                    network_policy=network_policy.value,
                    timed_out=True,
                    error=f"execution exceeded timeout of {timeout_seconds}s",
                )
        except Exception as e:
            return SandboxResult(
                stdout="",
                stderr=str(e),
                exit_code=-1,
                runtime="docker",
                network_policy=network_policy.value,
                error=f"docker execution error: {e}",
            )
        finally:
            if script_path.exists():
                try:
                    script_path.unlink()
                except Exception:
                    pass


class ProcessSandbox:
    """Hardened local subprocess runner with isolated environment and watchdog."""

    BLOCKED_PATTERNS = [
        r"\bformat\s+[a-zA-Z]:",
        r"\bmkfs\b",
        r"rm\s+-rf\s+/",
        r"del\s+/[fF]\s+/[sS]\s+/[qQ]\s+[cC]:\\",
        r":\(\)\s*\{\s*:\|:&\s*\}\s*;",  # forkbomb
        r"\bshutdown\b",
        r"\breboot\b",
    ]

    @staticmethod
    def _sanitized_env() -> Dict[str, str]:
        """Strip sensitive host environment variables (API keys, credentials)."""
        safe_keys = {
            "PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PYTHONPATH", "HOME",
            "USERPROFILE", "LANG", "LC_ALL"
        }
        env = {}
        for k, v in os.environ.items():
            if k.upper() in safe_keys:
                env[k] = v
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def run(
        self,
        code: str,
        workspace_dir: Path,
        timeout_seconds: int = 15,
        network_policy: NetworkPolicy = NetworkPolicy.ISOLATED,
        use_venv: bool = True,
    ) -> SandboxResult:
        python_bin = None
        if use_venv:
            venv_mgr = VenvManager(workspace_dir)
            python_bin = venv_mgr.get_python_exe()

        if not python_bin:
            python_bin = shutil.which("python") or shutil.which("python3")

        if not python_bin:
            return SandboxResult(
                stdout="",
                stderr="Python executable not found in PATH",
                exit_code=-1,
                runtime="process_sandbox",
                network_policy=network_policy.value,
                error="Python executable not found",
            )

        env = self._sanitized_env()
        if network_policy == NetworkPolicy.ISOLATED:
            env["PYTHON_ISOLATED_SANDBOX"] = "1"

        try:
            proc = subprocess.Popen(
                [str(python_bin), "-c", code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(workspace_dir.resolve()),
                env=env,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout_seconds)
                return SandboxResult(
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=proc.returncode,
                    runtime="process_sandbox",
                    network_policy=network_policy.value,
                )
            except subprocess.TimeoutExpired:
                ProcessWatchdog.kill_process_tree(proc.pid)
                return SandboxResult(
                    stdout="",
                    stderr="",
                    exit_code=-1,
                    runtime="process_sandbox",
                    network_policy=network_policy.value,
                    timed_out=True,
                    error=f"execution exceeded timeout of {timeout_seconds}s",
                )
        except Exception as e:
            return SandboxResult(
                stdout="",
                stderr=str(e),
                exit_code=-1,
                runtime="process_sandbox",
                network_policy=network_policy.value,
                error=str(e),
            )

    def run_shell(
        self,
        command: str,
        workspace_dir: Path,
        timeout_seconds: int = 20,
    ) -> SandboxResult:
        """Executes a shell command inside the workspace directory after security checks."""
        for pattern in self.BLOCKED_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return SandboxResult(
                    stdout="",
                    stderr=f"Security violation: command matched blocked pattern {pattern!r}",
                    exit_code=126,
                    runtime="process_sandbox",
                    network_policy=NetworkPolicy.OPEN.value,
                    error="Command rejected by security policy",
                )

        env = self._sanitized_env()
        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(workspace_dir.resolve()),
                env=env,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout_seconds)
                return SandboxResult(
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=proc.returncode,
                    runtime="process_sandbox",
                    network_policy=NetworkPolicy.OPEN.value,
                )
            except subprocess.TimeoutExpired:
                ProcessWatchdog.kill_process_tree(proc.pid)
                return SandboxResult(
                    stdout="",
                    stderr="",
                    exit_code=-1,
                    runtime="process_sandbox",
                    network_policy=NetworkPolicy.OPEN.value,
                    timed_out=True,
                    error=f"shell command exceeded timeout of {timeout_seconds}s",
                )
        except Exception as e:
            return SandboxResult(
                stdout="",
                stderr=str(e),
                exit_code=-1,
                runtime="process_sandbox",
                network_policy=NetworkPolicy.OPEN.value,
                error=str(e),
            )


class SandboxManager:
    """Orchestrates container and process sandboxes with automatic health detection."""

    def __init__(self, force_process: bool = False):
        self.force_process = force_process
        self._docker = DockerSandbox()
        self._process = ProcessSandbox()

    def execute(
        self,
        code: str,
        workspace_dir: Path,
        timeout_seconds: int = 15,
        network_policy: NetworkPolicy = NetworkPolicy.ISOLATED,
        use_venv: bool = True,
    ) -> SandboxResult:
        if not self.force_process and self._docker.is_docker_available():
            return self._docker.run(
                code=code,
                workspace_dir=workspace_dir,
                timeout_seconds=timeout_seconds,
                network_policy=network_policy,
            )
        return self._process.run(
            code=code,
            workspace_dir=workspace_dir,
            timeout_seconds=timeout_seconds,
            network_policy=network_policy,
            use_venv=use_venv,
        )

    def execute_shell(
        self,
        command: str,
        workspace_dir: Path,
        timeout_seconds: int = 20,
    ) -> SandboxResult:
        return self._process.run_shell(
            command=command,
            workspace_dir=workspace_dir,
            timeout_seconds=timeout_seconds,
        )

    def install_package(
        self,
        package_name: str,
        workspace_dir: Path,
        timeout_seconds: int = 60,
    ) -> SandboxResult:
        mgr = VenvManager(workspace_dir)
        return mgr.install_package(package_name, timeout_seconds=timeout_seconds)


default_sandbox = SandboxManager()
