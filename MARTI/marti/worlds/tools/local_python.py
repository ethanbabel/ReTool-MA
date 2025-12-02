import asyncio
import os
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, Tuple

from marti.worlds.tools.base import BaseToolExecutor
from marti.helpers.logging import init_logger

logger = init_logger(__name__)


class LocalPythonToolExecutor(BaseToolExecutor):
    """Simple sandboxed Python executor that runs code via a subprocess."""

    def __init__(
        self,
        timeout: int = 20,
        python_bin: str | None = None,
        max_output_chars: int = 4096,
        work_dir: str | None = None,
    ):
        self.timeout = timeout
        self.python_bin = python_bin or sys.executable
        self.max_output_chars = max_output_chars
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.gettempdir())
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def get_name(self) -> str:
        return "local_python"

    async def execute(self, parameters: Dict[str, Any], **kwargs) -> Tuple[str, Dict[str, Any]]:
        code = parameters.get("code", "")
        if not isinstance(code, str):
            code = str(code)
        if not code.strip():
            raise ValueError("Code snippet is empty.")

        # Dedent to make multi-line prompts cleaner.
        code = textwrap.dedent(code)

        tmp_dir = Path(tempfile.mkdtemp(prefix="marti-python-", dir=self.work_dir))
        script_path = tmp_dir / "snippet.py"
        script_path.write_text(code, encoding="utf-8")

        process = await asyncio.create_subprocess_exec(
            self.python_bin,
            "-u",
            str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(tmp_dir),
        )

        timed_out = False
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            timed_out = True
            process.kill()
            stdout_bytes, stderr_bytes = await process.communicate()

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        stdout = stdout[: self.max_output_chars]
        stderr = stderr[: self.max_output_chars]

        return_code = process.returncode if not timed_out else -1
        success = (return_code == 0) and not timed_out

        metadata = {
            "status": "success" if success else "failed",
            "timeout": timed_out,
            "return_code": return_code,
            "stdout": stdout,
            "stderr": stderr,
            "working_dir": str(tmp_dir),
        }

        if timed_out:
            response = "[timeout]"
        elif success:
            response = stdout or "[no stdout]"
        else:
            response = stderr or "[execution error]"

        # Clean up temporary directory best-effort.
        try:
            for child in tmp_dir.iterdir():
                child.unlink(missing_ok=True)
            tmp_dir.rmdir()
        except OSError as exc:
            logger.debug(f"Failed to clean up temp dir {tmp_dir}: {exc}")

        return response, metadata
