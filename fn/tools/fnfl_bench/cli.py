"""fnfl-bench's command line: one subcommand a bench module.

A bench module has HELP, add_args(parser) and run(args, run, info) -> (headline, data, body_html, title, subtitle);
cli writes the run folder and refreshes the index. A new bench (for example the standard multi-user points) is one
more module and one more SUBCOMMANDS entry."""

import argparse
import importlib
import sys

from . import core

# name: (module, needs numpy, help)
SUBCOMMANDS = {
    "serve": ("launch", True, "load the model (fn/tools/fnfl-serve.sh), run the synthetic bench, stop the server"),
    "vllm": ("synth", True, "vLLM's synthetic serving bench (random dataset) against the running server"),
    "pelicans": ("pelicans", False, "N users each draw an SVG pelican on a bicycle at once; a gallery page"),
    "storm": ("storm", True, "writers stream poetry; a long read drops in beside them"),
    "agents": ("agents", False, "pi agent lanes on six seeded easy tasks; tests decide pass or fail"),
}


def needs_numpy(argv):
    return len(argv) > 0 and argv[0] in SUBCOMMANDS and SUBCOMMANDS[argv[0]][1]


def parser():
    ap = argparse.ArgumentParser(prog="fnfl-bench", description="FNFL's bench runner. Every run writes "
                                 "RUNS/NAME-<UTC time>/ (index.html and result.json) and refreshes RUNS/index.html.")
    sub = ap.add_subparsers(dest="bench", required=True, metavar="BENCH")
    for name, (mod, _, hlp) in SUBCOMMANDS.items():
        p = sub.add_parser(name, help=hlp, description=hlp)
        p.add_argument("--url", default="http://127.0.0.1:8190", help="the server (default http://127.0.0.1:8190)")
        p.add_argument("--out", dest="out_raw", default=None,
                       help="a number: generated tokens a request (the synthetic bench); a path: the runs folder")
        p.add_argument("--runs", default=None, help="the runs folder (default ./bench-runs)")
        p.add_argument("--name", default=None, help="the run's name (default the bench's)")
        p.add_argument("--server-log", default=None, help="the server's log (default: found from its process)")
        importlib.import_module(f"fnfl_bench.{mod}").add_args(p)
    return ap


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = parser().parse_args(argv)
    args.url = args.url.rstrip("/")
    if args.out_raw is not None:
        if args.out_raw.isdigit():
            if not hasattr(args, "output_len"):
                raise SystemExit(f"--out {args.out_raw}: {args.bench} has no output length; give --runs DIR for the folder")
            args.output_len = int(args.out_raw)
        else:
            args.runs = args.runs or args.out_raw
    mod = importlib.import_module(f"fnfl_bench.{SUBCOMMANDS[args.bench][0]}")
    if (args.bench == "serve") == core.health(args.url):
        raise SystemExit(f"a server already answers at {args.url}: serve starts its own (stop that one, or run vllm)"
                         if args.bench == "serve" else f"no server answering /health at {args.url}")
    run = core.Run(args.runs or "bench-runs", args.name or args.bench)
    core.say(f"fnfl-bench {args.bench}: run folder {run.dir}")
    if args.bench == "serve":
        headline, data, body, title, sub = mod.run(args, run, None)
    else:
        info = core.server_info(args.url)
        headline, data, body, title, sub = mod.run(args, run, info)
    run.finish(args.bench, headline, data, body, title, sub)
    core.say(f"headline: {headline}")
    return 0
