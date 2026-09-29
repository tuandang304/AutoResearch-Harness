import base64
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch
import uuid

import requests

from ai_scientist.remote.workspace import extract_workspace
from ai_scientist.treesearch.remote_interpreter import (
    RemoteInterpreter,
    load_remote_config,
    RemoteExecutorUnavailable,
)


def archive_entry(name, kind=tarfile.REGTYPE, linkname=""):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo(name)
        member.type, member.linkname = kind, linkname
        member.size = 1 if kind == tarfile.REGTYPE else 0
        archive.addfile(member, io.BytesIO(b"x") if member.size else None)
    return buffer.getvalue()


class ArchiveTests(unittest.TestCase):
    def test_lost_job_resubmits_with_a_fresh_id(self):
        submitted = []

        def request(method, path, **kwargs):
            response = requests.Response()
            response.status_code = 200
            if method == "POST":
                submitted.append(kwargs["json"]["job_id"])
                body = {"job_id": submitted[-1]}
            elif path.endswith("/workspace"):
                response._content = archive_entry("result.txt")
                return response
            elif len(submitted) == 1:
                response.status_code = 404
                body = {"error": "unknown job"}
            else:
                body = {
                    "status": "done",
                    "result": {
                        "term_out": ["ok"],
                        "exec_time": 0.1,
                        "exc_type": None,
                        "exc_info": None,
                        "exc_stack": None,
                    },
                }
            response._content = json.dumps(body).encode()
            return response

        with tempfile.TemporaryDirectory() as directory, patch(
            "ai_scientist.treesearch.remote_interpreter.POLL_SECONDS", 0
        ):
            interpreter = RemoteInterpreter(directory)
            with patch.object(
                interpreter, "_request", side_effect=request
            ), patch.object(interpreter, "cleanup_session"):
                self.assertIsNone(interpreter.run("print('ok')").exc_type)
            self.assertEqual(len(submitted), 2)
            self.assertNotEqual(submitted[0], submitted[1])

    def test_paths_links_and_devices_rejected(self):
        for name, kind in (
            ("../escape", tarfile.REGTYPE),
            ("/tmp/escape", tarfile.REGTYPE),
            ("link", tarfile.SYMTYPE),
            ("link", tarfile.LNKTYPE),
            ("fifo", tarfile.FIFOTYPE),
        ):
            with self.subTest(
                name=name, kind=kind
            ), tempfile.TemporaryDirectory() as root:
                with self.assertRaises(ValueError):
                    extract_workspace(archive_entry(name, kind, "/tmp"), root)

    def test_existing_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as outside:
            (Path(root) / "link").symlink_to(outside)
            with self.assertRaises(ValueError):
                extract_workspace(archive_entry("link/file"), root)
            self.assertEqual(list(Path(outside).iterdir()), [])

    def test_expansion_limit(self):
        with tempfile.TemporaryDirectory() as root, patch(
            "ai_scientist.remote.workspace.MAX_WORKSPACE_BYTES", 0
        ):
            with self.assertRaises(ValueError):
                extract_workspace(archive_entry("file"), root)
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_config_rejects_remote_plaintext(self):
        with patch.dict(
            os.environ,
            AI_SCIENTIST_REMOTE_URL="http://example.com",
            AI_SCIENTIST_REMOTE_TOKEN="token",
        ):
            with self.assertRaises(RemoteExecutorUnavailable):
                load_remote_config()


class RemoteIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.TemporaryDirectory()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        cls.url = f"http://127.0.0.1:{port}"
        cls.headers = {"Authorization": "Bearer test-secret"}
        cls.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "ai_scientist.remote.colab_server",
                "--port",
                str(port),
            ],
            env={
                **os.environ,
                "AISCI_JOBS_ROOT": cls.root.name,
                "AISCI_TOKEN": "test-secret",
                "AISCI_MAX_CONCURRENT": "1",
            },
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                if requests.get(
                    cls.url + "/health", headers=cls.headers, timeout=0.5
                ).ok:
                    return
            except requests.RequestException:
                time.sleep(0.05)
        cls.proc.terminate()
        cls.proc.wait(timeout=5)
        cls.root.cleanup()
        raise RuntimeError("Test server failed to start")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(timeout=10)
        cls.root.cleanup()

    def request(self, method, path, **kwargs):
        return requests.request(
            method, self.url + path, headers=self.headers, timeout=10, **kwargs
        )

    def test_auth_and_validation(self):
        self.assertEqual(requests.get(self.url + "/health", timeout=2).status_code, 401)
        for spec in (
            {},
            {"code": "", "timeout": -1},
            {"code": "", "agent_file_name": "../escape.py"},
            {"code": "", "env_vars": []},
        ):
            self.assertEqual(self.request("POST", "/jobs", json=spec).status_code, 400)
        malicious = base64.b64encode(archive_entry("../escape")).decode()
        self.assertEqual(
            self.request(
                "POST", "/jobs", json={"code": "", "workspace_tgz_b64": malicious}
            ).status_code,
            400,
        )

    def test_retry_deduplication_and_conflict(self):
        spec = {"job_id": uuid.uuid4().hex, "code": "print('once')"}
        first = self.request("POST", "/jobs", json=spec)
        second = self.request("POST", "/jobs", json=spec)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json(), second.json())
        self.assertEqual(
            self.request(
                "POST", "/jobs", json={**spec, "code": "print('different')"}
            ).status_code,
            409,
        )
        self.request("DELETE", "/jobs/" + first.json()["job_id"])

    def test_cancel_running_and_queued_jobs(self):
        jobs = [
            self.request(
                "POST", "/jobs", json={"code": "import time; time.sleep(30)"}
            ).json()["job_id"]
            for _ in range(2)
        ]
        for job_id in reversed(jobs):
            self.assertEqual(self.request("DELETE", "/jobs/" + job_id).status_code, 200)
            self.assertEqual(self.request("GET", "/jobs/" + job_id).status_code, 404)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and any(
            (Path(self.root.name) / job_id).exists() for job_id in jobs
        ):
            time.sleep(0.02)
        self.assertFalse(
            any((Path(self.root.name) / job_id).exists() for job_id in jobs)
        )

    def test_execution_sync_exceptions_timeout_and_retention(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(
            os.environ,
            AI_SCIENTIST_REMOTE_URL=self.url,
            AI_SCIENTIST_REMOTE_TOKEN="test-secret",
        ), patch("ai_scientist.treesearch.remote_interpreter.POLL_SECONDS", 0.03):
            working = Path(root) / "working"
            working.mkdir()
            (working / "stale").write_text("old")
            interpreter = RemoteInterpreter(root, timeout=2, wait_minutes=0.1)
            result = interpreter.run(
                "import os\nfrom pathlib import Path\nprint(os.getenv('AISCI_TOKEN', 'absent'))\nPath('working/stale').unlink()\nPath('working/result').write_text('ok')\n"
            )
            self.assertIsNone(result.exc_type)
            self.assertIn("absent", "".join(result.term_out))
            self.assertFalse((working / "stale").exists())
            self.assertEqual((working / "result").read_text(), "ok")
            result = interpreter.run(
                "from concurrent.futures import ProcessPoolExecutor\n"
                "def square(x):\n    return x * x\n"
                "with ProcessPoolExecutor(2) as pool:\n"
                "    print('squares', sum(pool.map(square, range(4))))\n"
            )
            self.assertIsNone(result.exc_type, "".join(result.term_out))
            self.assertIn("squares 14", "".join(result.term_out))
            result = interpreter.run("raise ValueError('expected')")
            self.assertEqual(result.exc_type, "ValueError")
            self.assertIsNone(interpreter.run("import sys; sys.exit(0)").exc_type)
            interpreter.timeout = 0.2
            result = interpreter.run("import time\ntime.sleep(20)")
            self.assertEqual(result.exc_type, "TimeoutError")
            interpreter.timeout = 2
            interpreter.max_file_mb = 0.001
            result = interpreter.run(
                "from pathlib import Path\nPath('working/large.bin').write_bytes(b'x' * 4096)"
            )
            self.assertIsNone(result.exc_type)
            self.assertFalse((working / "large.bin").exists())
            # Skipped checkpoint is retained on the executor instead of silently deleted.
            retained = list(Path(self.root.name).glob("*/ws/working/large.bin"))
            self.assertEqual(len(retained), 1)
            self.request("DELETE", "/jobs/" + retained[0].parents[2].name)


if __name__ == "__main__":
    unittest.main()
