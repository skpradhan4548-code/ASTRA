"""
Security and payload scanner for DocuAgent / ASTRA.

Validates filenames, file extensions, and payload contents to prevent:
1. Execution of untrusted binary executables (.exe, .dll, .so, .dylib, .scr, etc.).
2. Script drops targeting host execution (.bat, .cmd, .vbs, .ps1, etc.).
3. Binary masquerade attacks (disguising PE/ELF binaries with .txt or .csv extensions).
"""

import os
import re
from pathlib import Path
from typing import Optional, Tuple, Union


class SecurityScanner:
    """Validates files and input arguments against malicious payload patterns."""

    BLOCKED_EXTENSIONS = {
        ".exe", ".dll", ".bat", ".cmd", ".vbs", ".vbe", ".jse", ".wsf",
        ".wsh", ".msc", ".scr", ".pif", ".hta", ".cpl", ".com", ".ps1",
        ".psm1", ".psd1", ".reg", ".inf", ".so", ".dylib", ".bin"
    }

    # Common binary executable magic bytes
    BINARY_SIGNATURES = [
        (b"MZ", "Windows PE executable / DLL"),
        (b"\x7fELF", "Linux ELF executable"),
        (b"\xfe\xed\xfa\xce", "Mach-O 32-bit executable"),
        (b"\xfe\xed\xfa\xcf", "Mach-O 64-bit executable"),
        (b"\xce\xfa\xed\xfe", "Mach-O reversed"),
        (b"\xcf\xfa\xed\xfe", "Mach-O 64-bit reversed"),
        (b"\xca\xfe\xba\xbe", "Java class file"),
    ]

    @classmethod
    def validate_file_path(cls, filename: str) -> Tuple[bool, Optional[str]]:
        """Validates filename and extension safety."""
        clean_name = os.path.basename(filename.strip())
        suffix = Path(clean_name).suffix.lower()

        if suffix in cls.BLOCKED_EXTENSIONS:
            return False, f"File extension {suffix!r} is prohibited by security policy."

        # Block hidden files or path separator tricks
        if re.search(r"[\x00-\x1f\x7f]", filename):
            return False, "Filename contains prohibited control characters."

        return True, None

    @classmethod
    def validate_file_content(
        cls, content: Union[str, bytes]
    ) -> Tuple[bool, Optional[str]]:
        """Checks for executable binary magic bytes masquerading in content."""
        if isinstance(content, str):
            raw_bytes = content.encode("utf-8", errors="ignore")[:16]
        else:
            raw_bytes = content[:16]

        for sig, description in cls.BINARY_SIGNATURES:
            if raw_bytes.startswith(sig):
                return False, f"Security violation: binary payload detected ({description})."

        return True, None

    @classmethod
    def scan_file_write(
        cls, filename: str, content: Union[str, bytes]
    ) -> Tuple[bool, Optional[str]]:
        """Composite check on both filename and payload content."""
        valid_path, err_path = cls.validate_file_path(filename)
        if not valid_path:
            return False, err_path

        valid_content, err_content = cls.validate_file_content(content)
        if not valid_content:
            return False, err_content

        return True, None
