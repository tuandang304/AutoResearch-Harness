import ast
import json
import os
import signal
from pathlib import Path
import stat
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from autoresearch import colab_runtime
from autoresearch.colab_runtime import (
    ColabPool, bootstrap_script, candidates, compute_instructions, is_oom, next_tier,
    overloaded, parse_compute, redact, validate_colab_settings,
)
from autoresearch.config import validate_config
from ai_scientist.treesearch.interpreter import ExecutionResult
from ai_scientist.treesearch.remote_interpreter import (
    PoolClosed,
    RemoteExecutorUnavailable,
    RemoteInterpreter,
    SessionStatusUnknown,
)

FAKE = Path(__file__).resolve().parent / "fixtures" / "fake_colab.py"
# Fails with a CUDA OOM on the tiers listed in OOM_ON, prints its tier otherwise.
OOM_CODE = (
    "# COMPUTE: {tier}  # test workload\n"
    "import os\n"
    "if os.environ['FAKE_TIER'] in {oom!r}:\n"
    "    raise RuntimeError('CUDA out of memory. Tried to allocate 2.00 GiB')\n"
    "print('finished on', os.environ['FAKE_TIER'])\n"
)


class PolicyTests(unittest.TestCase):
    def test_defaults_and_ordering(self):
        settings = validate_colab_settings(None)
        self.assertEqual(settings["gpu"], ["T4"])
        self.assertEqual(list(settings["tiers"]), ["cpu", "T4", "L4", "A100"])
        shuffled = validate_colab_settings({"tiers": {"A100": {"rate": 5}, "T4": {"rate": 1}}})
        self.assertEqual(list(shuffled["tiers"]), ["T4", "A100"])

    def test_invalid_settings_are_rejected_before_any_cli_call(self):
        # An unknown --gpu value silently becomes an A100 in the real CLI.
        for bad in ({"gpu": "a100"}, {"gpu": []}, {"idle": 3}, {"max_compute_units": -1},
                    {"auto_provision": "yes"}, {"max_concurrent": 0}, {"max_sessions": 0},
                    {"selection": "auto"}, {"tiers": {"V100": {"rate": 1}}},
                    {"tiers": {"T4": {"rate": 0}}}, {"tiers": {"T4": {"rate": 1, "vram": 16}}},
                    {"gpu": "H100"},  # default tier missing from the menu
                    {"max_upgrade_ratio": 0.5},
                    {"packages": ["x; rm -rf /"]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_colab_settings(bad)
        self.assertEqual(validate_colab_settings({"selection": "fixed", "gpu": "H100"})["gpu"],
                         ["H100"])

    def test_run_config_validation_covers_exec_colab(self):
        config = {
            "exp_name": "run",
            "exec": {"backend": "colab", "colab": {"gpu": "V100"}},
            "report": {"model": "m"},
            "agent": {"type": "parallel", "num_workers": 1, "steps": 1,
                      "multi_seed_eval": {"num_seeds": 1}, "stages": {"stage1_max_iters": 1},
                      "search": {"num_drafts": 1, "max_debug_depth": 1, "debug_prob": 0.5},
                      **{role: {"model": "m"} for role in ("code", "feedback", "vlm_feedback")}},
        }
        with self.assertRaisesRegex(ValueError, "exec.colab.gpu"):
            validate_config(config)

    def test_compute_declaration_parsing(self):
        self.assertEqual(parse_compute("# COMPUTE: a100  # ViT-B fine-tune\nx=1"),
                         ("A100", "ViT-B fine-tune"))
        self.assertEqual(parse_compute("import os\n#COMPUTE:CPU\n"), ("cpu", ""))
        self.assertEqual(parse_compute("# COMPUTE: V100\n"), (None, ""))
        self.assertEqual(parse_compute("print('no header')"), (None, ""))

    def test_fallbacks_never_cost_more_and_escalation_is_one_step(self):
        settings = validate_colab_settings(None)
        self.assertEqual(candidates(settings, "A100"), ["A100", "L4", "T4"])
        self.assertEqual(candidates(settings, "cpu"), ["cpu"])
        # L4 is ~1.44x a T4 (within 1.5x); A100 is ~3.2x an L4 and never a fallback.
        self.assertEqual(candidates(settings, None), ["T4", "L4"])
        self.assertEqual(candidates(settings, "L4"), ["L4", "T4"])
        self.assertEqual(candidates(settings, "H100"), ["T4", "L4"])  # not on the menu
        strict = {**settings, "max_upgrade_ratio": 1}
        self.assertEqual(candidates(strict, "T4"), ["T4"])
        self.assertEqual(next_tier(settings, "T4"), "L4")
        self.assertIsNone(next_tier(settings, "A100"))
        self.assertIsNone(next_tier(settings, "cpu"))
        self.assertIsNone(next_tier({**settings, "escalate_on_oom": False}, "T4"))
        fixed = validate_colab_settings({"selection": "fixed", "gpu": ["L4", "T4"]})
        self.assertEqual(candidates(fixed, "A100"), ["L4", "T4"])
        self.assertIsNone(next_tier(fixed, "L4"))

    def test_overload_detection(self):
        self.assertTrue(overloaded({"running": 1, "queued": 0, "max_concurrent": 1}))
        self.assertFalse(overloaded({"running": 1, "queued": 0, "max_concurrent": 2,
                                     "gpu": "Tesla T4, 4000 MiB, 15360 MiB"}))
        self.assertTrue(overloaded({"running": 1, "queued": 0, "max_concurrent": 2,
                                    "gpu": "Tesla T4, 14000 MiB, 15360 MiB"}))
        self.assertFalse(overloaded({"running": 0, "queued": 0, "max_concurrent": 1,
                                     "gpu": "Tesla T4, 14000 MiB, 15360 MiB"}))  # leftover cache

    def test_oom_detection(self):
        oom = ExecutionResult(["RuntimeError: CUDA out of memory. Tried"], 1.0, "RuntimeError")
        self.assertTrue(is_oom(oom))
        self.assertFalse(is_oom(ExecutionResult(["ValueError: bad"], 1.0, "ValueError")))

    def test_bootstrap_parses_and_redaction_hides_proxy_tokens(self):
        ast.parse(bootstrap_script(2, ["timm"]))
        self.assertNotIn("SECRET", redact("url?colab-runtime-proxy-token=SECRET&x=1"))


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        wrapper = self.dir / "colab"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
        wrapper.chmod(0o755)
        self.env = patch.dict(os.environ, {
            "FAKE_COLAB_DIR": str(self.dir), "AUTORESEARCH_COLAB_CLI": str(wrapper),
            "AUTORESEARCH_COMPUTE_LOG": str(self.dir / "compute.jsonl"),
        })
        self.env.start()
        for key in ("AI_SCIENTIST_REMOTE_URL", "AI_SCIENTIST_REMOTE_TOKEN"):
            os.environ.pop(key, None)
        for patcher in (patch.object(colab_runtime, "STATE_DIR", self.dir / "state"),
                        patch("ai_scientist.treesearch.remote_interpreter.POLL_SECONDS", 0.05)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        state = self.dir / "state.json"
        if state.exists():
            for name in json.loads(state.read_text())["sessions"]:
                colab_runtime.ColabCLI().stop(name)
        self.env.stop()
        self.tmp.cleanup()

    def fake(self, **values):
        path = self.dir / "state.json"
        state = json.loads(path.read_text()) if path.exists() else {
            "sessions": {}, "balance": 100.0, "rate": 0.0}
        state.update(values)
        path.write_text(json.dumps(state))

    def vm_sessions(self):
        path = self.dir / "state.json"
        return sorted(json.loads(path.read_text())["sessions"]) if path.exists() else []

    def calls(self):
        path = self.dir / "calls.log"
        return path.read_text().splitlines() if path.exists() else []

    def managed(self, **settings):
        pool = ColabPool.create({"packages": [], **settings}, owner_pid=None)
        os.environ[colab_runtime.STATE_ENV] = str(pool.state_path)
        return pool

    def interpreter(self):
        workspace = self.dir / f"ws{len(list(self.dir.glob('ws*')))}"
        (workspace / "working").mkdir(parents=True)
        return RemoteInterpreter(workspace, timeout=30, wait_minutes=0.2)

    def run_job(self, code="print('hello from fake colab')", interpreter=None):
        return (interpreter or self.interpreter()).run(code)

    def records(self):
        path = self.dir / "compute.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_first_job_provisions_lazily_and_keeps_token_out_of_logs(self):
        pool = self.managed()
        self.assertEqual(self.calls(), [])  # creating state allocates nothing
        result = self.run_job()
        self.assertIsNone(result.exc_type)
        self.assertIn("hello from fake colab", "".join(result.term_out))
        self.assertTrue(result.term_out[0].startswith("[compute] Requested no tier; ran on T4"))
        entry = pool.load()["sessions"]["T4"]
        self.assertEqual((entry["up"], entry["provisions"]), (True, 1))
        self.assertEqual(entry["measured_rate"], 1.19)
        self.assertTrue(any(c.startswith(f"new -s {entry['session']} --gpu T4")
                            for c in self.calls()))
        executor = Path(entry["executor_path"])
        self.assertEqual(stat.S_IMODE(executor.stat().st_mode), 0o600)
        token = json.loads(executor.read_text())["token"]
        for path in (self.dir / "calls.log", pool.state_path, self.dir / "compute.jsonl"):
            self.assertNotIn(token, path.read_text())
        self.assertEqual(self.records()[0]["tier"], "T4")

    def test_declared_tiers_route_to_separate_sessions_and_stick_for_follow_up_code(self):
        pool = self.managed()
        light = self.interpreter()
        result = self.run_job("# COMPUTE: cpu  # sklearn baseline\nimport os\n"
                              "print('tier', os.environ['FAKE_TIER'])", light)
        self.assertIn("tier cpu", "".join(result.term_out))
        follow_up = self.run_job("import os\nprint('plot on', os.environ['FAKE_TIER'])", light)
        self.assertIn("plot on cpu", "".join(follow_up.term_out))
        heavy = self.run_job("# COMPUTE: L4 # larger batch\nimport os\n"
                             "print('tier', os.environ['FAKE_TIER'])")
        self.assertIn("tier L4", "".join(heavy.term_out))
        state = pool.load()["sessions"]
        self.assertEqual(sorted(t for t, e in state.items() if e["up"]), ["L4", "cpu"])
        self.assertEqual([r["declared"] for r in self.records()], ["cpu", None, "L4"])
        self.assertEqual(self.records()[0]["reason"], "sklearn baseline")

    def test_max_sessions_stops_an_idle_session_to_make_room(self):
        pool = self.managed(max_sessions=1)
        self.run_job("# COMPUTE: cpu\nprint(1)")
        self.run_job("# COMPUTE: T4\nprint(2)")
        sessions = pool.load()["sessions"]
        self.assertEqual((sessions["cpu"]["up"], sessions["cpu"]["stop_reason"]),
                         (False, "make-room"))
        self.assertEqual(self.vm_sessions(), [sessions["T4"]["session"]])

    def test_unavailable_tier_falls_back_to_a_cheaper_one(self):
        os.environ["FAKE_COLAB_FAIL_GPU"] = "A100"
        pool = self.managed()
        result = self.run_job("# COMPUTE: A100 # big\nimport os\nprint(os.environ['FAKE_TIER'])")
        self.assertIn("ran on L4", result.term_out[0])
        self.assertIn("A100", result.term_out[0])
        self.assertNotIn("SECRETA100", pool.state_path.read_text() + result.term_out[0])
        self.assertEqual(pool.load()["sessions"]["A100"]["stop_reason"], "allocation-failed")

    def test_capacity_refusal_is_retried_once(self):
        os.environ["FAKE_COLAB_BUSY_GPU"] = "T4:1"
        pool = self.managed()
        with patch.object(colab_runtime.time, "sleep"):
            self.assertIsNone(self.run_job("# COMPUTE: T4\nprint(1)").exc_type)
        self.assertEqual(pool.load()["sessions"]["T4"]["provisions"], 1)
        self.assertEqual(sum(c.startswith("new") for c in self.calls()), 2)

    def test_scarce_t4_upgrades_to_l4_within_the_ratio(self):
        os.environ["FAKE_COLAB_BUSY_GPU"] = "T4"
        pool = self.managed()
        with patch.object(colab_runtime.time, "sleep"):
            result = self.run_job("# COMPUTE: T4\nimport os\nprint(os.environ['FAKE_TIER'])")
        self.assertIn("ran on L4", result.term_out[0])
        self.assertIn("Service Unavailable", result.term_out[0])
        self.assertFalse(any("--gpu A100" in c for c in self.calls()))

    def test_persistent_capacity_refusal_waits_then_fails_without_pricier_tiers(self):
        os.environ["FAKE_COLAB_BUSY_GPU"] = "T4"
        pool = self.managed(max_upgrade_ratio=1)
        interpreter = self.interpreter()
        interpreter.wait_minutes = 0
        with patch.object(colab_runtime.time, "sleep"), \
                self.assertRaises(colab_runtime.NoTierAvailable) as caught:
            interpreter.run("# COMPUTE: T4\nprint(1)")
        self.assertIn("Service Unavailable", str(caught.exception))
        entry = pool.load()["sessions"]["T4"]
        self.assertEqual(entry["stop_reason"], "capacity")
        self.assertNotIn("\n", str(caught.exception))
        self.assertLess(entry["unavailable_until"] - colab_runtime.time.time(),
                        colab_runtime.TRANSIENT_SECONDS + 1)
        self.assertFalse(any("--gpu L4" in c or "--gpu A100" in c for c in self.calls()))

    def test_unrequested_accelerator_is_stopped_and_treated_as_unavailable(self):
        os.environ["FAKE_COLAB_GPU_NAME_A100"] = "Tesla T4, 15360 MiB"  # Colab can hand out a T4
        pool = self.managed()
        result = self.run_job("# COMPUTE: A100\nprint('ok')")
        self.assertIn("ran on L4", result.term_out[0])
        self.assertEqual(pool.load()["sessions"]["A100"]["stop_reason"], "wrong-gpu")
        self.assertNotIn(pool.load()["sessions"]["A100"]["session"], self.vm_sessions())

    def test_cuda_oom_reruns_once_on_the_next_tier(self):
        pool = self.managed()
        result = self.run_job(OOM_CODE.format(tier="T4", oom=["T4"]))
        self.assertIsNone(result.exc_type)
        self.assertIn("finished on L4", "".join(result.term_out))
        self.assertIn("T4 ran out of GPU memory; re-ran on L4", result.term_out[0])
        self.assertEqual(self.records()[-1]["escalated_from"], "T4")
        # A second OOM is reported to the model instead of climbing further.
        again = self.run_job(OOM_CODE.format(tier="T4", oom=["T4", "L4"]))
        self.assertEqual(again.exc_type, "RuntimeError")
        self.assertIn("ran on L4", again.term_out[0])
        self.assertFalse(pool.load()["sessions"].get("A100", {}).get("up"))

    def test_prompt_lists_tiers_with_budget(self):
        self.assertIsNone(compute_instructions())
        pool = self.managed(max_compute_units=12)
        lines = compute_instructions()["Compute selection"]
        text = "\n".join(lines)
        for fragment in ("# COMPUTE: <tier>", "cpu: no GPU", "T4: 15 GB", "A100: 40 GB",
                         "Remaining run budget: 12.0 of 12", "runs on T4",
                         "another VM of the same tier (up to 2)"):
            self.assertIn(fragment, text)
        self.run_job()
        self.assertIn("T4: 15 GB GPU memory, ~1.19", "\n".join(compute_instructions()["Compute selection"]))
        fixed = ColabPool.create({"selection": "fixed", "packages": []})
        os.environ[colab_runtime.STATE_ENV] = str(fixed.state_path)
        self.assertIsNone(compute_instructions())

    def test_idle_stop_is_per_tier_and_next_job_reprovisions(self):
        pool = self.managed(idle_stop_minutes=60,
                            tiers={"cpu": {"rate": 0.08}, "T4": {"rate": 1.2, "idle_stop_minutes": 0}})
        self.run_job("# COMPUTE: cpu\nprint(1)")
        self.run_job("# COMPUTE: T4\nprint(2)")
        pool.watch(once=True)
        sessions = pool.load()["sessions"]
        self.assertEqual((sessions["cpu"]["up"], sessions["T4"]["up"]), (True, False))
        self.assertEqual(sessions["T4"]["stop_reason"], "idle")
        self.assertIsNone(self.run_job("# COMPUTE: T4\nprint(3)").exc_type)
        self.assertEqual(pool.load()["sessions"]["T4"]["provisions"], 2)

    def test_budget_stop_is_terminal(self):
        pool = self.managed(max_compute_units=10, idle_stop_minutes=None)
        self.assertIsNone(self.run_job().exc_type)
        self.fake(balance=85.0)
        pool.watch(once=True)
        state = pool.load()
        self.assertEqual(state["stop_reason"], "budget")
        self.assertEqual(state["spent_units"], 15.0)
        self.assertEqual(self.vm_sessions(), [])
        with self.assertRaisesRegex(RemoteExecutorUnavailable, "budget"):
            self.run_job()

    def test_release_prevents_reprovisioning_and_summary_has_no_token(self):
        pool = self.managed()
        self.run_job("# COMPUTE: cpu\nprint(1)")
        self.run_job("# COMPUTE: T4\nprint(1)")
        colab_runtime.release_from_env("experiments-complete")
        self.assertTrue(pool.load()["released"])
        self.assertEqual(self.vm_sessions(), [])
        with self.assertRaisesRegex(RemoteExecutorUnavailable, "released"):
            pool.acquire("T4")
        summary = pool.summary()
        self.assertEqual(sorted(summary["sessions"]), ["T4", "cpu"])
        self.assertNotIn("token", json.dumps(summary))

    def test_unreachable_tunnel_is_restarted_once_then_the_vm_is_stopped(self):
        os.environ["FAKE_COLAB_BAD_URL"] = "1"
        pool = self.managed()
        with patch.object(colab_runtime, "HEALTH_SECONDS", 0.2), \
                patch.object(colab_runtime.time, "sleep"), \
                self.assertRaisesRegex(RemoteExecutorUnavailable, "VM stopped.*quic"):
            pool.acquire("T4")
        self.assertEqual(sum(" -f " in c for c in self.calls() if c.startswith("exec")), 2)
        self.assertEqual(pool.load()["sessions"]["T4"]["stop_reason"], "tunnel-unreachable")
        self.assertEqual(self.vm_sessions(), [])

    def test_unknown_session_status_never_replaces_a_running_vm(self):
        pool = self.managed()
        self.assertIsNone(self.run_job("print('first')").exc_type)
        server = json.loads((self.dir / "state.json").read_text())["sessions"]
        os.killpg(next(iter(server.values()))["server_pid"], signal.SIGTERM)  # tunnel gone
        (self.dir / "status_fails").touch()
        with self.assertRaises(SessionStatusUnknown):
            pool.ensure("T4")
        self.assertEqual(sum(c.startswith("new") for c in self.calls()), 1)
        (self.dir / "status_fails").unlink()
        pool.ensure("T4")  # the VM still exists: only the server/tunnel restarts
        self.assertEqual(sum(c.startswith("new") for c in self.calls()), 1)
        self.assertIsNone(self.run_job("print('again')").exc_type)

    def test_failed_reconnect_keeps_an_existing_vm(self):
        pool = self.managed()
        self.assertIsNone(self.run_job("print('first')").exc_type)
        server = json.loads((self.dir / "state.json").read_text())["sessions"]
        os.killpg(next(iter(server.values()))["server_pid"], signal.SIGTERM)
        os.environ["FAKE_COLAB_BAD_URL"] = "1"  # the restarted tunnel stays unreachable
        with patch.object(colab_runtime, "HEALTH_SECONDS", 0.2), \
                patch.object(colab_runtime.time, "sleep"), \
                self.assertRaisesRegex(RemoteExecutorUnavailable, "VM kept"):
            pool.ensure("T4")
        self.assertEqual(len(self.vm_sessions()), 1)  # not stopped
        self.assertFalse(any(c.startswith("stop") for c in self.calls()))
        del os.environ["FAKE_COLAB_BAD_URL"]
        pool.ensure("T4")
        self.assertEqual(sum(c.startswith("new") for c in self.calls()), 1)

    def test_reconnect_retries_transient_errors_but_not_a_closed_pool(self):
        interpreter = self.interpreter()
        with patch.object(RemoteInterpreter, "_recover",
                          side_effect=colab_runtime.TierUnavailable("L4: ReadTimeoutError")):
            self.assertFalse(interpreter._try_recover())
        with patch.object(RemoteInterpreter, "_recover", side_effect=PoolClosed("budget spent")), \
                self.assertRaises(PoolClosed):
            interpreter._try_recover()

    def background_job(self, code, results, name):
        thread = threading.Thread(target=lambda: results.__setitem__(name, self.run_job(code)))
        thread.start()
        return thread

    def wait_until_running(self, pool, key="T4"):
        for _ in range(200):
            health = pool.health(pool.load()["sessions"].get(key, {"executor_path": "/none"}))
            if health and health["running"]:
                return
            colab_runtime.time.sleep(0.05)
        self.fail("background job never started")

    def test_busy_tier_scales_out_to_another_vm_and_back_in_when_idle(self):
        pool = self.managed(idle_stop_minutes=None, max_sessions=3)
        results = {}
        self.run_job("# COMPUTE: T4\nprint('warm')")
        thread = self.background_job("# COMPUTE: T4\nimport time\ntime.sleep(2)", results, "long")
        self.wait_until_running(pool)
        second = self.run_job("# COMPUTE: T4\nimport os\nprint('on', os.environ['FAKE_SESSION'])")
        thread.join()
        self.assertIn("-t4-2", "".join(second.term_out))
        self.assertIn("started T4 replica 2", second.term_out[0])
        self.assertIsNone(results["long"].exc_type)
        sessions = pool.load()["sessions"]
        self.assertEqual(sorted(k for k, e in sessions.items() if e["up"]), ["T4", "T4.2"])
        self.assertEqual(sorted(r["replica"] for r in self.records()), ["T4", "T4", "T4.2"])
        # Both idle again: the next job reuses a VM instead of starting a third.
        self.run_job("# COMPUTE: T4\nprint(1)")
        self.assertEqual(sum("--gpu T4" in c for c in self.calls()), 2)
        # Scale-in is the per-replica idle stop.
        state = pool.load()
        state["settings"]["idle_stop_minutes"] = 0
        pool.save(state)
        pool.watch(once=True)
        self.assertEqual(self.vm_sessions(), [])

    def test_scale_out_limits_queue_the_job_instead(self):
        for limits, reason in (({"max_replicas": 1}, "max_replicas=1"),
                               ({"max_sessions": 1}, "max_sessions=1 VMs are all busy"),
                               ({"max_compute_units": 0.3}, "scale-out reserve")):
            with self.subTest(limits=limits):
                pool = self.managed(idle_stop_minutes=None, **limits)
                results = {}
                thread = self.background_job("# COMPUTE: T4\nimport time\ntime.sleep(1.5)",
                                             results, "long")
                self.wait_until_running(pool)
                queued = self.run_job("# COMPUTE: T4\nprint('after')")
                thread.join()
                self.assertIn(reason, queued.term_out[0])
                self.assertIn("queued on T4", queued.term_out[0])
                self.assertNotIn("T4.2", pool.load()["sessions"])
                colab_runtime.release_from_env("next")

    def test_oom_on_a_shared_gpu_reruns_alone_on_the_same_tier(self):
        os.environ["FAKE_COLAB_SLOTS"] = "2"
        flag = self.dir / "contention"
        pool = self.managed(idle_stop_minutes=None, max_sessions=3)
        results = {}
        thread = self.background_job(
            f"# COMPUTE: T4\nimport time, pathlib\np = pathlib.Path({str(flag)!r})\n"
            "p.touch()\ntime.sleep(2.5)\np.unlink()", results, "neighbour")
        self.wait_until_running(pool)
        victim = self.run_job(
            f"# COMPUTE: T4\nimport os, pathlib\n"
            f"if pathlib.Path({str(flag)!r}).exists() and not os.environ['FAKE_SESSION'].endswith('-2'):\n"
            "    raise RuntimeError('CUDA out of memory. Tried to allocate 1 GiB')\n"
            "print('ran on', os.environ['FAKE_SESSION'])")
        thread.join()
        self.assertIsNone(victim.exc_type)
        self.assertIn("re-ran alone", victim.term_out[0])
        self.assertIn("-t4-2", "".join(victim.term_out))
        self.assertFalse(any("--gpu L4" in c for c in self.calls()))  # no tier escalation
        record = next(r for r in self.records() if r["replica"] == "T4.2")
        self.assertEqual((record["shared"], record["escalated_from"]), (False, None))
        self.assertIn("re-ran alone", " ".join(record["notes"]))

    def test_bootstrap_failure_detected_despite_zero_exit(self):
        os.environ["FAKE_COLAB_BOOT_FAIL"] = "1"
        pool = self.managed()
        with self.assertRaisesRegex(RemoteExecutorUnavailable, "bootstrap failed"):
            pool.acquire()
        self.assertEqual(self.vm_sessions(), [])

    def test_owner_exit_releases_every_session(self):
        pool = self.managed()
        pool.acquire("cpu")
        pool.acquire("T4")
        state = pool.load()
        state["owner_pid"] = 2 ** 22 + 12345  # beyond pid_max defaults; never alive
        pool.save(state)
        self.assertEqual(pool.watch(once=True), "owner-exited")
        self.assertTrue(pool.load()["released"])
        self.assertEqual(self.vm_sessions(), [])

    def test_low_balance_refuses_to_start(self):
        self.fake(balance=1.0)
        pool = self.managed(min_balance=5)
        with self.assertRaisesRegex(RemoteExecutorUnavailable, "below"):
            pool.preflight()
        self.assertFalse(any(c.startswith("new") for c in self.calls()))

    def test_fixed_mode_uses_the_pool_name_and_gpu_fallback_list(self):
        os.environ["FAKE_COLAB_FAIL_GPU"] = "L4"
        executor = self.dir / "remote_executor.json"
        pool = ColabPool.create({"selection": "fixed", "gpu": ["L4", "T4"], "packages": []},
                                executor_path=executor, session="manual-exec")
        tier, _, path, notes, _ = pool.acquire("A100")  # declarations are ignored in fixed mode
        self.assertEqual((tier, Path(path)), ("T4", executor.resolve()))
        self.assertEqual(self.vm_sessions(), ["manual-exec"])
        self.assertTrue(notes and notes[0].startswith("L4:"))



class AgentPromptTests(unittest.TestCase):
    def test_code_prompts_include_compute_menu_only_for_managed_colab(self):
        from types import SimpleNamespace
        from ai_scientist.treesearch.parallel_agent import MinimalAgent

        agent = MinimalAgent.__new__(MinimalAgent)
        agent.cfg = SimpleNamespace(exec=SimpleNamespace(backend="local"))
        self.assertEqual(agent._prompt_compute, {})
        agent.cfg.exec.backend = "colab"
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(colab_runtime, "STATE_DIR", Path(directory)):
            pool = ColabPool.create({"packages": []})
            with patch.dict(os.environ, {colab_runtime.STATE_ENV: str(pool.state_path)}):
                self.assertIn("Compute selection", agent._prompt_compute)
            self.assertEqual(agent._prompt_compute, {})  # unmanaged colab: manual executor


if __name__ == "__main__":
    unittest.main()
