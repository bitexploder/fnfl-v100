"""agents: a short, light agent run. --users lanes in parallel, one seeded easy task each (agents_fixture.TASKS, in
order, wrapping), each on a clean copy of the fixture project, driven by pi (the agent harness DeepSWE uses) in a
bwrap sandbox against the server. The task's tests decide pass or fail. The page shows each lane's result, turns,
tokens, wall time and its longest wait for the model. A lane past --timeout is killed by its exact PID."""

import html
import json
import os
import shutil
import signal
import subprocess
import threading
import time

from . import agents_fixture as fx
from . import core

LATP = os.path.expanduser("~/src/flashnext-opt/work/latp-tasks")
PI_MODEL = ("fnfl", "fnfl")


def add_args(p):
    p.add_argument("--users", type=int, default=2, help="lanes (default 2); lane k takes task k (six tasks, wrapping)")
    p.add_argument("--timeout", type=float, default=300, help="seconds before a lane's agent is killed (default 300)")
    p.add_argument("--node", default=f"{LATP}/node", help="the node install with pi (DeepSWE's)")
    p.add_argument("--tasks", default=None, help="comma-separated task ids instead of the default order")


def models_json(url):
    return {"providers": {PI_MODEL[0]: {
        "baseUrl": url.rstrip("/") + "/v1", "api": "openai-completions", "apiKey": "dummy",
        "models": [{"id": PI_MODEL[1], "name": "FNFL (bench)", "reasoning": True, "input": ["text"],
                    "contextWindow": 131072, "maxTokens": 32768,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False, "supportsStore": False,
                               "supportsUsageInStreaming": True, "maxTokensField": "max_tokens"}}]}}}


class Lane:
    def __init__(self, k, task, root, args):
        self.k, self.task, self.root, self.args = k, task, root, args
        self.work = os.path.join(root, "work")
        self.agent_dir = os.path.join(root, "pi-agent")
        self.sessions = os.path.join(root, "sessions")
        self.turns, self.tool_calls, self.out_tok, self.in_tok = 0, 0, 0, 0
        self.replies = []      # seconds from a model call's start to its end, a turn each
        self.state = "starting"
        self.passed, self.test_out, self.status, self.wall, self.pid = None, "", "", 0.0, None
        self._t_msg = None

    def setup(self):
        os.makedirs(self.work)
        os.makedirs(self.agent_dir)
        os.makedirs(self.sessions)
        fx.make_copy(self.work, self.task)
        with open(os.path.join(self.agent_dir, "models.json"), "w") as f:
            json.dump(models_json(self.args.url), f, indent=1)
        for name, v in (("auth.json", {}), ("models-store.json", {}), ("settings.json", {"compaction": {"enabled": False}})):
            with open(os.path.join(self.agent_dir, name), "w") as f:
                json.dump(v, f)

    def sandbox(self):
        node = os.path.realpath(self.args.node)
        path = f"{node}/current/bin:{node}/prefix/bin:/usr/local/bin:/usr/bin:/bin"
        home = os.environ["HOME"]
        # DeepSWE's sandbox (latp-tasks/bin/sandbox.sh): the filesystem read-only, an empty HOME and /tmp, the lane's
        # copy and pi's directories writable, node read-only, its own PID namespace; the network shared (the server)
        box = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--tmpfs", home,
               "--tmpfs", f"/run/user/{os.getuid()}", "--ro-bind", node, node]
        for d in (self.work, self.agent_dir, self.sessions):
            box += ["--bind", d, d]
        for s in ("/run/docker.sock", "/run/containerd/containerd.sock"):
            if os.path.exists(s):
                box += ["--ro-bind", "/dev/null", s]
        box += ["--unshare-pid", "--unshare-ipc", "--unshare-uts", "--new-session", "--die-with-parent",
                "--setenv", "HOME", home, "--chdir", self.work]
        return box, path, home

    def command(self):
        box, path, home = self.sandbox()
        pi = ["pi", "--mode", "json", "--provider", PI_MODEL[0], "--model", PI_MODEL[1], "--no-approve", "--no-skills",
              "--no-extensions", "--no-prompt-templates", "--no-themes", "--session-dir", self.sessions, "--",
              self.task["prompt"]]
        env = {"PATH": path, "HOME": home, "LANG": "C.UTF-8", "TERM": "dumb", "PI_CODING_AGENT_DIR": self.agent_dir,
               "PI_OFFLINE": "1", "PI_SKIP_VERSION_CHECK": "1", "PI_TELEMETRY": "0", "NO_COLOR": "1",
               "PYTHONDONTWRITEBYTECODE": "1"}
        return box + pi, env

    def handle(self, e):
        t = e.get("type")
        if t == "turn_start":
            self.turns += 1
        elif t == "message_start" and (e.get("message") or {}).get("role") == "assistant":
            self._t_msg = time.time()
            self.state = "waiting for the model"
        elif t == "message_end" and (e.get("message") or {}).get("role") == "assistant":
            u = e["message"].get("usage") or {}
            self.out_tok += u.get("output", 0) or 0
            self.in_tok += (u.get("input", 0) or 0) + (u.get("cacheRead", 0) or 0)
            if self._t_msg:
                self.replies.append(time.time() - self._t_msg)
            self._t_msg = None
            self.state = "tools"
        elif t == "tool_execution_end":
            self.tool_calls += 1

    def run(self):
        cmd, env = self.command()
        ev = open(os.path.join(self.root, "events.jsonl"), "wb")
        err = open(os.path.join(self.root, "stderr.txt"), "wb")
        t0 = time.time()
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err, env=env)
        self.pid = p.pid

        def reader():
            for raw in p.stdout:
                ev.write(raw)
                ev.flush()
                try:
                    self.handle(json.loads(raw))
                except (ValueError, KeyError, TypeError):
                    pass

        th = threading.Thread(target=reader, daemon=True)
        th.start()
        self.status = "done"
        while p.poll() is None:
            time.sleep(0.5)
            if time.time() - t0 > self.args.timeout:
                self.status = "killed (timeout)"
                core.say(f"  lane {self.k}: past {self.args.timeout:g} s; killing pid {p.pid}")
                os.kill(p.pid, signal.SIGKILL)  # the sandbox's --die-with-parent ends pi with it
                break
        p.wait()
        th.join(5)
        self.wall = time.time() - t0
        if self.status == "done" and p.returncode != 0:
            self.status = f"exit {p.returncode}"
        self.state = "grading"
        self.passed, self.test_out = fx.grade(self.work, self.task, python="/usr/bin/python3", prefix=self.sandbox()[0])
        self.state = "PASS" if self.passed else "FAIL"
        core.say(f"  lane {self.k} ({self.task['id']}): {self.state}, {self.status}, {self.turns} turns, "
                 f"{self.out_tok} tokens out, {self.wall:.0f} s, longest model wait "
                 f"{max(self.replies, default=0):.1f} s")


def run(args, run_dir, info):
    if shutil.which("bwrap") is None:
        raise SystemExit("agents: bwrap is not installed")
    if not os.path.exists(os.path.join(os.path.realpath(args.node), "prefix", "bin", "pi")):
        raise SystemExit(f"agents: no pi under {args.node}/prefix/bin")
    tasks = fx.TASKS
    if args.tasks:
        by = {t["id"]: t for t in fx.TASKS}
        tasks = [by[x] for x in args.tasks.split(",")]
    lanes = [Lane(k, tasks[k % len(tasks)], run_dir.path(f"lane{k}"), args) for k in range(args.users)]
    for ln in lanes:
        ln.setup()
    core.say(f"[{run_dir.name}] {len(lanes)} lanes: " + ", ".join(f"{ln.k}:{ln.task['id']}" for ln in lanes))
    t0 = time.time()
    ths = [threading.Thread(target=ln.run, daemon=True) for ln in lanes]
    for th in ths:
        th.start()
    while any(th.is_alive() for th in ths):
        time.sleep(5)
        core.say(f"  t={time.time() - t0:5.0f}s  " + "  ".join(
            f"lane {ln.k}: {ln.state}, turn {ln.turns}, {ln.out_tok} tok" for ln in lanes))
    wall = time.time() - t0
    passed = sum(1 for ln in lanes if ln.passed)
    rows = []
    for ln in lanes:
        rows.append([str(ln.k), html.escape(ln.task["id"]),
                     f'<span class="{"ok" if ln.passed else "bad"}">{"pass" if ln.passed else "fail"}</span>',
                     str(ln.turns), str(ln.tool_calls), f"{ln.out_tok:,}", f"{ln.in_tok:,}", f"{ln.wall:.0f}",
                     f"{max(ln.replies, default=0):.1f}", html.escape(ln.status)])
    fails = "".join(f"<h2>Lane {ln.k} ({html.escape(ln.task['id'])}): test output</h2><pre>{html.escape(ln.test_out)}</pre>"
                    for ln in lanes if not ln.passed)
    body = (core.stats_block([(f"{passed} / {len(lanes)}", "tasks passed"), (f"{wall:.0f} s", "whole run"),
                              (f"{sum(ln.out_tok for ln in lanes):,}", "tokens generated"),
                              (f"{max((max(ln.replies, default=0) for ln in lanes), default=0):.1f} s", "longest model wait")])
            + core.table(["Lane", "Task", "Result", "Turns", "Tool calls", "Tokens out", "Prompt tokens", "Wall (s)",
                          "Longest wait (s)", "Agent"], rows, num_cols=(3, 4, 5, 6, 7, 8))
            + fails
            + '<p class="foot">Longest wait: the longest single model call of the lane (from the reply\'s start to its '
              "end). Prompt tokens add each call's prompt (the server's usage). Tests are rewritten from the bench before "
              "grading, so an agent cannot pass by editing them.</p>")
    headline = f"{passed}/{len(lanes)} tasks passed, {wall:.0f} s"
    sub = (f"{len(lanes)} pi lanes in parallel (DeepSWE's harness, bwrap sandbox), one seeded task each on a clean copy "
           f"of the fixture project; timeout {args.timeout:g} s a lane.<br>{core.server_line(info)}")
    data = {"server": info, "users": len(lanes), "wall_s": wall, "passed": passed,
            "lanes": [{"lane": ln.k, "task": ln.task["id"], "passed": ln.passed, "status": ln.status, "turns": ln.turns,
                       "tool_calls": ln.tool_calls, "output_tokens": ln.out_tok, "prompt_tokens": ln.in_tok,
                       "wall_s": ln.wall, "replies_s": ln.replies, "pid": ln.pid} for ln in lanes]}
    return headline, data, body, "Agents: six easy tasks", sub
