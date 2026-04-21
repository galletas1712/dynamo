# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import List, Optional


class ServerManager:
    """Manages the lifecycle of a serving backend launched via a bash script.

    Uses ``setsid`` so the server gets its own process group, allowing clean
    shutdown without killing the orchestrator.
    """

    def __init__(self, port: int = 8000, timeout: int = 600) -> None:
        self.port = port
        self.timeout = timeout
        self._process: Optional[subprocess.Popen] = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(
        self,
        workflow_script: str,
        model: str,
        extra_args: Optional[List[str]] = None,
        env_overrides: Optional[dict] = None,
    ) -> None:
        """Launch the workflow script and block until the model is served."""
        if self.is_running:
            raise RuntimeError("Server is already running. Call stop() first.")

        script = Path(workflow_script)
        if not script.is_file():
            raise FileNotFoundError(f"Workflow script not found: {script}")

        model_flag = "--model-path" if "trtllm" in str(script) else "--model"
        cmd = ["bash", str(script), model_flag, model]
        if extra_args:
            cmd.extend(extra_args)

        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)

        print(f"Launching: {' '.join(cmd)}", flush=True)
        self._process = subprocess.Popen(
            cmd,
            start_new_session=True,
            env=env,
        )

        self.wait_for_ready(model)

    def wait_for_ready(self, model: str) -> None:
        """Poll /v1/models until the expected model name appears."""
        import urllib.error
        import urllib.request

        url = f"http://localhost:{self.port}/v1/models"
        deadline = time.monotonic() + self.timeout

        print(
            f"Waiting for server at {url} to list model '{model}' "
            f"(timeout: {self.timeout}s)...",
            flush=True,
        )

        while time.monotonic() < deadline:
            if not self.is_running:
                raise RuntimeError(
                    "Server process exited unexpectedly during startup "
                    f"(exit code {self._process.returncode})."
                )
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=5) as resp:
                    body = resp.read().decode()
                    if model in body:
                        print("Server is ready (model registered).", flush=True)
                        return
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
            time.sleep(5)

        self.stop()
        raise TimeoutError(f"Server did not become ready within {self.timeout}s")

    def stop(self) -> None:
        """Stop the server by killing its process group."""
        if self._process is None:
            return

        pid = self._process.pid
        print(f"Stopping server (PID {pid})...", flush=True)

        # SIGINT to the top-level process only (not the whole process group):
        # - nsys 2026.2.1 ignores SIGTERM; SIGINT triggers its graceful
        #   stop-and-finalize path (produces a proper .nsys-rep).
        # - killpg(SIGINT) also signals every vLLM worker simultaneously,
        #   which crashes them abruptly before nsys can finalize. Sending
        #   SIGINT to only the root (nsys, via exec) lets nsys tear down
        #   its children cleanly.
        try:
            self._process.send_signal(signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            pass

        # Wait long enough for nsys to kill vllm, finalize its .nsys-rep, and
        # exit. Finalize can take 60-90s for large traces; tolerate up to 180s
        # before escalating to SIGKILL, else we lose the trace (nsys only emits
        # .nsys-rep on graceful exit, not SIGKILL).
        try:
            self._process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            self._process.wait(timeout=5)

        print(f"Server stopped (PID {pid}).", flush=True)
        self._process = None
        time.sleep(5)
