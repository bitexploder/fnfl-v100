"""serve: load, run, down. Starts FNFL's server through fn/tools/fnfl-serve.sh (its locks, card checks and production
defaults: `start --split` unless --serve-flags says otherwise), waits for /health, runs the synthetic bench (its
presets and flags), then stops the server it started. It never stops another server: when something already answers
on the port, or fnfl-serve.sh refuses (busy cards or held locks), it ends without running."""

import os
import shlex
import subprocess
import time
import urllib.parse

from . import core, synth

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # fn/tools
REPO = os.path.dirname(os.path.dirname(TOOLS))


def add_args(p):
    synth.add_args(p)
    p.add_argument("--build", default=os.path.join(REPO, "build-fnfl"),
                   help="the build folder whose bin/llama-server (and bin/fnfl-drive, the same path on the follower) "
                        "runs (default this checkout's build-fnfl)")
    p.add_argument("--model", default=None, help="the target's first shard (default: fnfl-serve.sh's)")
    p.add_argument("--mmproj", default=None, help="the vision projector (default none)")
    p.add_argument("--serve-flags", default="--split", help="fnfl-serve.sh start's flags (default --split, production)")


def serve(cmd, env, log):
    core.say(f"$ {' '.join(f'{k}={v}' for k, v in env.items())} fnfl-serve.sh {cmd}")
    p = subprocess.Popen([os.path.join(TOOLS, "fnfl-serve.sh")] + shlex.split(cmd), env=dict(os.environ, **env),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in p.stdout:
        core.say("  " + line.rstrip())
        lines.append(line.rstrip())
    p.wait()
    with open(log, "a") as f:
        f.write(f"$ fnfl-serve.sh {cmd}\n" + "\n".join(lines) + f"\nrc {p.returncode}\n")
    return p.returncode, lines


def run(args, run_dir, _info):
    if core.health(args.url):
        raise SystemExit(f"a server already answers at {args.url}: fnfl-bench serve starts its own; stop that one "
                         "first (or run `fnfl-bench vllm` against it)")
    port = urllib.parse.urlparse(args.url).port or 8190
    sdir = run_dir.path("server")
    os.makedirs(sdir, exist_ok=True)
    env = {"BIN": os.path.join(os.path.abspath(args.build), "bin", "llama-server"), "SERVE_DIR": sdir, "PORT": str(port)}
    if args.model:
        env["MODEL"] = args.model
    if args.mmproj:
        env["MMPROJ"] = args.mmproj
    if not os.path.exists(env["BIN"]):
        raise SystemExit(f"no llama-server at {env['BIN']} (give --build)")
    log = run_dir.path("serve.log")
    t0 = time.time()
    rc, _ = serve("start " + args.serve_flags, env, log)
    t_up = time.time()
    loaded = rc == 0 and core.health(args.url)
    try:
        if not loaded:
            raise SystemExit(f"fnfl-serve.sh start failed (rc {rc}); see {log}")
        info = core.server_info(args.url)
        info["serve_dir"] = sdir
        args.server_log = args.server_log or os.path.join(sdir, "server.log")
        headline, data, body, title, sub = synth.run(args, run_dir, info)
    finally:
        rc_stop, stop_lines = serve("stop", env, log)
        core.say(f"server stopped (rc {rc_stop})")
    data.update(load={"start_s": t_up - t0, "started_rc": rc, "stop_rc": rc_stop, "stop": stop_lines,
                      "build": args.build, "serve_flags": args.serve_flags, "model": args.model, "env": env})
    sub += f"<br>Loaded by fnfl-bench serve (fnfl-serve.sh start {args.serve_flags}) in {t_up - t0:.0f} s, stopped after."
    return headline, data, body, title, sub
