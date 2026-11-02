"""pelicans: N users each ask for an SVG of a pelican riding a bicycle at once. A gallery page of the drawings with
each one's tokens, decode speed and time. The server's own sampling and thinking; no token cap unless asked.
Replaces work/pelicans/ (pelicans.py, pelicans2.py, pelican_gallery.py, pelican_stats.py)."""

import base64
import html
import re
import threading
import time

from . import core

PROMPT = "Generate an SVG of a pelican riding a bicycle"


def add_args(p):
    p.add_argument("--users", type=int, default=4, help="pelicans drawn at once (default 4)")
    p.add_argument("--max-tokens", type=int, default=None, help="token cap a pelican (default none)")
    p.add_argument("--prompt", default=PROMPT)


def run(args, run_dir, info):
    url, n = args.url, args.users
    core.say(f"[{run_dir.name}] {n} pelicans at once on {info.get('build')}: {args.prompt!r}")
    res = [None] * n
    live = [None] * n
    slog = core.ServerLog(url, args.server_log)
    log_start = slog.offset()
    t0 = time.perf_counter()

    def one(i):
        body = {"model": info["model_id"], "messages": [{"role": "user", "content": args.prompt}]}
        if args.max_tokens:
            body["max_tokens"] = args.max_tokens
        live[i] = core.StreamResult()
        r = core.stream(url, "/v1/chat/completions", body, on_chunk=lambda s: live.__setitem__(i, s))
        res[i] = r
        m = re.search(r"<svg\b.*?</svg>", r.text, re.S | re.I)
        r.svg = m.group(0) if m else None
        s = r.summary()
        core.say(f"  pelican {i + 1}: {'SVG' if r.svg else 'NO SVG'}, {r.output_tokens} tokens, "
                 f"{core.fmt(s['decode_tps'], '.1f')} tok/s, {r.latency:.1f} s, finish {r.finish}"
                 f"{'  ERROR ' + r.error[:100] if r.error else ''}")

    threads = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(n)]
    for th in threads:
        th.start()
    while any(th.is_alive() for th in threads):
        time.sleep(5)
        el = time.perf_counter() - t0
        core.say(f"  t={el:6.0f}s  chunks so far: " + " ".join(
            f"#{i + 1}:{len(x.chunk_times) if x else 0}{'' if res[i] is None else ' done'}" for i, x in enumerate(live)))
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0
    sums = [r.summary() for r in res]
    toks = [r.output_tokens for r in res if r.success]
    total = sum(toks)
    drawn = sum(1 for r in res if r.svg)
    ls = slog.stats(log_start)
    acc = core.acceptance(sums)
    acc = acc if acc is not None else ls.get("acceptance")
    dec = [s["decode_tps"] for s in sums if s["decode_tps"]]
    cards = []
    for i, r in enumerate(res):
        s = sums[i]
        img = (f'<img alt="pelican {i + 1}" src="data:image/svg+xml;base64,{base64.b64encode(r.svg.encode()).decode()}">'
               if r.svg else '<div class="none">no SVG</div>')
        cards.append(f'<figure>{img}<figcaption><b>pelican {i + 1}</b>'
                     f'<dl><dt>Tokens</dt><dd>{r.output_tokens:,}</dd>'
                     f'<dt>Speed</dt><dd>{core.fmt(s["decode_tps"], ".1f")} tok/s</dd>'
                     f'<dt>Time</dt><dd>{r.latency:.1f} s</dd>'
                     f'<dt>Thinking</dt><dd>{len(r.reasoning):,} chars</dd>'
                                          f'<dt>Finish</dt><dd>{html.escape(str(r.finish or r.error[:40]))}</dd></dl></figcaption></figure>')
    body = (core.stats_block([(f"{drawn} / {n}", "pelicans drawn"), (f"{total:,}", "tokens generated"),
                              (f"{core.median(toks):,.0f}" if toks else "n/a", "median tokens a pelican"),
                              (core.fmt(core.median(dec) if dec else None, ".1f") + " tok/s", "median speed a pelican"),
                              (f"{total / wall:.1f} tok/s", "all pelicans together"),
                              (f"{wall:.0f} s", "whole run"),
                              (core.fmt(None if acc is None else 100 * acc, ".1f") + "%", "draft acceptance")])
            + f'<div class="grid">{"".join(cards)}</div>'
            + '<p class="foot">Speed is the server\'s decode rate for that request. "All together" is every token over '
              "the whole run, first request sent to last token.</p>")
    headline = f"{drawn}/{n} pelicans, {total:,} tokens, {total / wall:.1f} tok/s together, {wall:.0f} s"
    sub = (f'Prompt: "{html.escape(args.prompt)}". {n} requests at once, the server\'s default sampling and thinking, '
           f'{"no token cap" if not args.max_tokens else f"cap {args.max_tokens} tokens"}.<br>{core.server_line(info)}')
    data = {"server": info, "users": n, "prompt": args.prompt, "wall_s": wall, "drawn": drawn, "total_tokens": total,
            "acceptance": acc, "server_log": ls, "requests": [dict(s, svg=r.svg, thinking_chars=len(r.reasoning)) for s, r in zip(sums, res)]}
    for i, r in enumerate(res):
        if r.svg:
            with open(run_dir.path(f"pelican-{i + 1:02d}.svg"), "w") as f:
                f.write(r.svg + "\n")
    return headline, data, body, "Pelicans on a bicycle", sub
