"""storm: --users writers stream poetry; after --delay seconds one --in-token read (a prompt of exactly that many tokens,
built as the synthetic bench builds its prompts, one token out) drops in beside them. The page charts each writer's
tokens over time with the read marked, and gives the read's time and each writer's rate before, during and after it."""

import html
import threading
import time

from . import core, synth

POEM = "Write a long poem about the sea, many stanzas, and keep going until you are told to stop."


def add_args(p):
    p.add_argument("--users", type=int, default=4, help="writers (default 4)")
    p.add_argument("--in", dest="input_len", type=int, default=65536, help="the read's prompt tokens (default 65536)")
    p.add_argument("--delay", type=float, default=5.0, help="seconds before the read drops in (default 5)")
    p.add_argument("--tail", type=float, default=5.0, help="seconds the writers go on after the read (default 5)")
    p.add_argument("--seed", type=int, default=0)


def rate(times, a, b):
    n = sum(1 for t in times if a <= t < b)
    return n / (b - a) if b > a else None


def longest_gap(times, a, b):
    ts = [t for t in times if a <= t <= b]
    pts = [a] + ts + [b]
    return max((y - x for x, y in zip(pts, pts[1:])), default=None)


def chart(writers, read_a, read_b, t_end):
    W, H, L, B, T, R = 900, 340, 56, 36, 14, 14
    ymax = max([len(w.chunk_times) for w in writers] + [1])
    sx = lambda t: L + (W - L - R) * t / t_end  # noqa: E731
    sy = lambda y: H - B - (H - B - T) * y / ymax  # noqa: E731
    parts = [f'<rect x="{sx(read_a):.1f}" y="{T}" width="{max(sx(read_b) - sx(read_a), 1):.1f}" height="{H - B - T}" '
             f'fill="var(--band)"/>',
             f'<text x="{sx(read_a) + 4:.1f}" y="{T + 14}" font-size="12" fill="var(--acc)">read</text>']
    for k in range(6):
        y = ymax * k / 5
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="var(--line)"/>'
                     f'<text x="{L - 6}" y="{sy(y) + 4:.1f}" font-size="11" text-anchor="end" fill="var(--mut)">{y:.0f}</text>')
    for k in range(7):
        t = t_end * k / 6
        parts.append(f'<text x="{sx(t):.1f}" y="{H - B + 16}" font-size="11" text-anchor="middle" fill="var(--mut)">{t:.0f}s</text>')
    for i, w in enumerate(writers):
        pts = " ".join(f"{sx(t):.1f},{sy(j + 1):.1f}" for j, t in enumerate(w.rel))
        parts.append(f'<polyline points="{sx(0):.1f},{sy(0):.1f} {pts}" fill="none" stroke="var(--s{i % 8 + 1})" stroke-width="1.8"/>')
    parts.append(f'<text x="{L}" y="{H - 4}" font-size="11" fill="var(--mut)">seconds from the first writer\'s start; '
                 f'y: streamed tokens</text>')
    legend = " ".join(f'<span style="color:var(--s{i % 8 + 1})">&#9632;</span> writer {i + 1}' for i in range(len(writers)))
    return (f'<div class="chart"><svg viewBox="0 0 {W} {H}" role="img" aria-label="writer tokens over time">'
            f'{"".join(parts)}</svg><div class="legend">{legend}</div></div>')


def run(args, run_dir, info):
    url = args.url
    tok = synth.ServerTokenizer(url)
    tok.find_vocab()
    core.say(f"[{run_dir.name}] building the {args.input_len}-token read prompt")
    read_prompt = synth.random_prompts(tok, 1, args.input_len, 1, seed=args.seed)[0]
    stop = threading.Event()
    writers = [None] * args.users
    live = [core.StreamResult() for _ in range(args.users)]
    t0 = time.perf_counter()

    def writer(i):
        body = dict(core.PRODUCTION, model=info["model_id"], max_tokens=32768,
                    messages=[{"role": "user", "content": POEM + f" (Poem {i + 1}.)"}])
        writers[i] = core.stream(url, "/v1/chat/completions", body, stop_event=stop,
                                 on_chunk=lambda s: live.__setitem__(i, s))

    ths = [threading.Thread(target=writer, args=(i,), daemon=True) for i in range(args.users)]
    for th in ths:
        th.start()
    read = {}

    def reader():
        body = {"model": info["model_id"], "prompt": read_prompt[0], "max_tokens": 1, "temperature": 0.0}
        read["r"] = core.stream(url, "/v1/completions", body)

    rt = None
    while True:
        time.sleep(1)
        el = time.perf_counter() - t0
        state = "waiting" if rt is None else ("reading" if rt.is_alive() else "read done")
        core.say(f"  t={el:6.1f}s  [{state}]  writer tokens: " + " ".join(f"#{i + 1}:{len(x.chunk_times)}" for i, x in enumerate(live)))
        if rt is None and el >= args.delay:
            core.say(f"  t={el:6.1f}s  the {read_prompt[1]}-token read drops in")
            rt = threading.Thread(target=reader, daemon=True)
            rt.start()
        if rt is not None and not rt.is_alive():
            if "end" not in read:
                read["end"] = time.perf_counter()
                r = read["r"]
                core.say(f"  read done: {r.latency:.2f} s (first token {r.ttft:.2f} s)"
                         f"{'  ERROR ' + r.error[:120] if r.error else ''}")
            if time.perf_counter() - read["end"] >= args.tail:
                break
        if not any(th.is_alive() for th in ths) and rt is not None and not rt.is_alive():
            break
    stop.set()
    for th in ths:
        th.join(30)
    r = read["r"]
    ra, rb = r.start - t0, r.start - t0 + r.latency
    t_end = time.perf_counter() - t0
    rows, wdata = [], []
    for i, w in enumerate(writers):
        w = w or live[i]
        w.rel = [t - t0 for t in w.chunk_times]
        first = w.rel[0] if w.rel else ra
        before, during, after = rate(w.rel, first, ra), rate(w.rel, ra, rb), rate(w.rel, rb, t_end)
        gap = longest_gap(w.rel, ra, rb)
        rows.append([f"writer {i + 1}", f"{len(w.rel):,}", core.fmt(before, ".1f"), core.fmt(during, ".1f"),
                     core.fmt(after, ".1f"), core.fmt(gap, ".2f")])
        wdata.append({"tokens": len(w.rel), "rate_before": before, "rate_during": during, "rate_after": after,
                      "longest_gap_in_read_s": gap, "chunk_times": w.rel, "error": w.error if w.error != "stopped by the bench" else ""})
        writers[i] = w
    rs = r.summary()
    body = (core.stats_block([(f"{r.latency:.2f} s", f"read time ({read_prompt[1]:,} tokens)"),
                              (f"{r.ttft:.2f} s", "read's first token"),
                              (core.fmt(rs["prompt_tps"], ".0f"), "server prompt tok/s (the read)"),
                              (core.fmt(max((d["longest_gap_in_read_s"] or 0) for d in wdata), ".2f") + " s",
                               "longest writer stall in the read")])
            + chart(writers, ra, rb, t_end)
            + "<h2>Writers</h2>" + core.table(["", "Tokens", "tok/s before", "tok/s during read", "tok/s after",
                                                "Longest gap in read (s)"], rows, num_cols=(1, 2, 3, 4, 5))
            + '<p class="foot">Tokens are streamed chunks as the client saw them. "Before" counts from the writer\'s '
              "first token to the read's start.</p>")
    headline = (f"{args.users} writers + {read_prompt[1]:,}-token read: read {r.latency:.2f} s, writers "
                f"{sum(d['rate_during'] or 0 for d in wdata):.0f} tok/s during it vs {sum(d['rate_before'] or 0 for d in wdata):.0f} before")
    sub = (f"{args.users} writers stream poetry (production sampling); after {args.delay:g} s one read of "
           f"{read_prompt[1]:,} tokens (one token out) arrives; the writers stop {args.tail:g} s after it ends.<br>"
           f"{core.server_line(info)}")
    data = {"server": info, "users": args.users, "read_tokens": read_prompt[1], "delay": args.delay, "tail": args.tail,
            "read": dict(rs, start_s=ra, end_s=rb), "writers": wdata}
    return headline, data, body, "Storm: writers and one long read", sub
