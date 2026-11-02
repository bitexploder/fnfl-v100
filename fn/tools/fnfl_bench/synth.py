"""The synthetic serving bench: vLLM's `vllm bench serve` random dataset and metrics, ported from vLLM c722f3a5.

Sources: vllm/benchmarks/datasets/datasets.py (RandomDataset, gen_prompt_decode_to_target_len), datasets/utils.py
(get_sampling_params), lib/endpoint_request_func.py (async_request_openai_completions) and serve.py (benchmark,
calculate_metrics). Prompts, the request rate (inf), the client concurrency limit and every metric follow vLLM.
Changes, forced by the server: the tokenizer is the server's /tokenize and /detokenize; the client is threads over
http.client instead of aiohttp; usage and timings are read from the final chunk, where llama-server puts them."""

import html
import math
import threading
import time

from . import core

PRESETS = {"prefill": (2048, 128), "write": (128, 2048), "short": (128, 128), "long": (2048, 2048)}


class ServerTokenizer:
    """vLLM's TokenizerLike, backed by the server. vocab_size is the count of the model's normal tokens: an HF
    tokenizer's vocab_size likewise leaves out the added (special) tokens, which all come after them."""

    def __init__(self, url):
        self.url = url
        self.vocab_size = None
        self.all_special_ids = []

    def encode(self, text, add_special_tokens=False):
        return core.http_json(self.url + "/tokenize", {"content": text, "add_special": add_special_tokens})["tokens"]

    def decode(self, tokens):
        return core.http_json(self.url + "/detokenize", {"tokens": [int(t) for t in tokens]})["content"]

    def num_special_tokens_to_add(self):
        # FNFL's target adds no BOS (tokenizer.ggml.add_bos_token = false)
        return len(core.http_json(self.url + "/tokenize", {"content": "a", "add_special": True})["tokens"]) - \
            len(self.encode("a"))

    def find_vocab(self):
        """The first special token ends the normal vocabulary. /detokenize shows specials as their text."""
        n_vocab = core.http_json(self.url + "/v1/models")["data"][0].get("meta", {}).get("n_vocab")
        eos = self.encode("<|endoftext|>")
        if n_vocab and len(eos) == 1:
            self.vocab_size = eos[0]
            self.all_special_ids = list(range(eos[0], n_vocab))
        else:
            raise SystemExit("cannot find the vocabulary size from the server (/v1/models meta.n_vocab, <|endoftext|>)")


def get_sampling_params(rng, num_requests, range_ratio, input_len, output_len, tokenizer):
    num_special_tokens = int(tokenizer.num_special_tokens_to_add())
    real_input_len = max(0, int(input_len) - num_special_tokens)
    input_low = math.floor(real_input_len * (1 - range_ratio))
    input_high = math.ceil(real_input_len * (1 + range_ratio))
    output_low = max(math.floor(output_len * (1 - range_ratio)), 1)
    output_high = max(math.ceil(output_len * (1 + range_ratio)), 1)
    input_lens = rng.integers(input_low, input_high + 1, size=num_requests)
    output_lens = rng.integers(output_low, output_high + 1, size=num_requests)
    offsets = rng.integers(0, tokenizer.vocab_size, size=num_requests)
    return input_lens, output_lens, offsets


def gen_prompt_decode_to_target_len(tokenizer, token_sequence, target_token_len, max_retry=10, rng=None):
    remain_num_try = max_retry
    token_mismatch = 0
    while True:
        prompt = tokenizer.decode(token_sequence)
        token_sequence = tokenizer.encode(prompt, add_special_tokens=False)
        if remain_num_try <= 0:
            if len(token_sequence) != target_token_len:
                token_mismatch = len(token_sequence) - target_token_len
            break
        if len(token_sequence) == target_token_len:
            break
        elif len(token_sequence) < target_token_len:
            token_sequence.extend(rng.integers(0, tokenizer.vocab_size,
                                               size=target_token_len - len(token_sequence)).tolist())
        else:
            token_sequence = token_sequence[:target_token_len]
        remain_num_try -= 1
    return prompt, token_sequence, token_mismatch


def random_prompts(tokenizer, num_requests, input_len, output_len, seed=0, range_ratio=0.0):
    """RandomDataset.sample with prefix_len 0: [(prompt, prompt_len, output_len)]."""
    import numpy as np

    rng = np.random.default_rng(seed)
    input_lens, output_lens, offsets = get_sampling_params(rng, num_requests, range_ratio, input_len, output_len,
                                                           tokenizer)
    all_tokens = np.arange(tokenizer.vocab_size)
    allowed_tokens = np.array(list(set(all_tokens) - set(tokenizer.all_special_ids)))
    out, mismatch = [], 0
    for i in range(num_requests):
        seq = allowed_tokens[(int(offsets[i]) + i + np.arange(int(input_lens[i]))) % len(allowed_tokens)].tolist()
        prompt, adjusted, mm = gen_prompt_decode_to_target_len(tokenizer, seq, int(input_lens[i]), rng=rng)
        mismatch += mm
        out.append((prompt, len(adjusted), int(output_lens[i])))
        if (i + 1) % 50 == 0 or i + 1 == num_requests:
            core.say(f"  prompts built: {i + 1}/{num_requests}")
    if mismatch:
        core.say(f"WARNING: across all prompts, {abs(mismatch)} {'more' if mismatch > 0 else 'fewer'} tokens than asked")
    return out


def calculate_metrics(outs, dur_s, percentiles=(99,)):
    """serve.py calculate_metrics, for streamed completions."""
    import numpy as np

    lens, total_input, completed = [], 0, 0
    itls, tpots, ttfts, e2els = [], [], [], []
    for o, plen in outs:
        if o.success:
            n = o.output_tokens
            lens.append(n)
            total_input += (o.usage or {}).get("prompt_tokens") or plen
            if n > 1:
                tpots.append((o.latency - o.ttft) / (n - 1))
            itls += o.itl
            ttfts.append(o.ttft)
            e2els.append(o.latency)
            completed += 1
        else:
            lens.append(0)
    peak_tps, peak_conc = 0.0, 0
    ok = [o for o, _ in outs if o.success]
    if ok:
        t0 = min(o.start for o in ok)
        nsec = int(np.ceil(max(o.start + o.latency for o in ok) - t0)) + 1
        tps, conc = np.zeros(nsec), np.zeros(nsec)
        for o in ok:
            for t in o.chunk_times:
                b = int(t - t0)
                if 0 <= b < nsec:
                    tps[b] += 1
            for s in range(int(o.start - t0), int(o.start + o.latency - t0) + 1):
                conc[s] += 1
        peak_tps, peak_conc = float(np.max(tps)), int(np.max(conc))
    m = {"completed": completed, "failed": len(outs) - completed, "total_input": total_input, "total_output": sum(lens),
         "request_throughput": completed / dur_s, "output_throughput": sum(lens) / dur_s,
         "total_token_throughput": (total_input + sum(lens)) / dur_s, "max_output_tokens_per_s": peak_tps,
         "max_concurrent_requests": peak_conc}
    for name, v in (("ttft", ttfts), ("tpot", tpots), ("itl", itls), ("e2el", e2els)):
        m[f"mean_{name}_ms"] = float(np.mean(v or 0) * 1000)
        m[f"median_{name}_ms"] = float(np.median(v or 0) * 1000)
        m[f"std_{name}_ms"] = float(np.std(v or 0) * 1000)
        m[f"percentiles_{name}_ms"] = [(p, float(np.percentile(v or 0, p) * 1000)) for p in percentiles]
    return m, lens


def vllm_text(m, users, dur, extra):
    """The result block as serve.py prints it, plus the server's lines."""
    L = ["{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="=")]
    row = lambda k, v, f="": L.append(("{:<40} {:<10" + f + "}").format(k, v))  # noqa: E731
    row("Successful requests:", m["completed"])
    row("Failed requests:", m["failed"])
    row("Maximum request concurrency:", users)
    row("Benchmark duration (s):", dur, ".2f")
    row("Total input tokens:", m["total_input"])
    row("Total generated tokens:", m["total_output"])
    row("Request throughput (req/s):", m["request_throughput"], ".2f")
    row("Output token throughput (tok/s):", m["output_throughput"], ".2f")
    row("Peak output token throughput (tok/s):", m["max_output_tokens_per_s"], ".2f")
    row("Peak concurrent requests:", m["max_concurrent_requests"], ".2f")
    row("Total token throughput (tok/s):", m["total_token_throughput"], ".2f")
    for a, nm, hdr in (("ttft", "TTFT", "Time to First Token"), ("tpot", "TPOT", "Time per Output Token (excl. 1st token)"),
                       ("itl", "ITL", "Inter-token Latency"), ("e2el", "E2EL", "End-to-end Latency")):
        L.append("{s:{c}^{n}}".format(s=hdr, n=50, c="-"))
        row(f"Mean {nm} (ms):", m[f"mean_{a}_ms"], ".2f")
        row(f"Median {nm} (ms):", m[f"median_{a}_ms"], ".2f")
        for p, v in m[f"percentiles_{a}_ms"]:
            row(f"P{int(p) if int(p) == p else p} {nm} (ms):", v, ".2f")
    L.append("{s:{c}^{n}}".format(s="Server (llama-server log)", n=50, c="-"))
    row("Sampling mode:", extra["sampling"])
    row("ignore_eos:", extra["ignore_eos"])
    row("Server prompt processing (tok/s):", core.fmt(extra["prompt_tps"]))
    row("Draft acceptance (%):", core.fmt(None if extra["acceptance"] is None else 100 * extra["acceptance"]))
    L.append("=" * 50)
    return L


def add_args(p):
    p.add_argument("--preset", choices=sorted(PRESETS), help="prefill 2048/128, write 128/2048, short 128/128, long 2048/2048")
    p.add_argument("--in", dest="input_len", type=int, help="prompt tokens (default 1024, or the preset's)")
    p.add_argument("--out-tokens", "--out-len", dest="output_len", type=int,
                   help="generated tokens a request (default 128, or the preset's)")
    p.add_argument("--users", type=int, default=8, help="max concurrent requests (vLLM --max-concurrency); default 8")
    p.add_argument("--prompts", type=int, default=None, help="number of requests (default 200, vLLM's CI)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sampling", choices=sorted(core.SAMPLING), default="vllm",
                   help="vllm: T 0 + ignore_eos (vLLM's CI, default); production: T 1.0 top-k 20 top-p 0.95 min-p 0.05 + ignore_eos")
    p.add_argument("--percentiles", default="99", help="metric percentiles, as vLLM's --metric-percentiles")


def shape(args):
    i, o = PRESETS[args.preset] if args.preset else (1024, 128)
    return (args.input_len or i), (args.output_len or o), (args.prompts or 200)


def run(args, run_dir, info):
    url = args.url
    input_len, output_len, n = shape(args)
    users = args.users
    pcts = [float(p) for p in args.percentiles.split(",")]
    tok = ServerTokenizer(url)
    tok.find_vocab()
    label = args.preset or f"{input_len}/{output_len}"
    core.say(f"[{run_dir.name}] random {input_len}/{output_len}, {n} prompts, {users} users, seed {args.seed}, "
             f"sampling {args.sampling}, server {info.get('build')}")
    t = time.time()
    prompts = random_prompts(tok, n, input_len, output_len, seed=args.seed)
    core.say(f"[{run_dir.name}] prompts ready in {time.time() - t:.1f} s; vocab {tok.vocab_size}")
    body, eos_note = core.sampling_body(args.sampling, url, info["model_id"])
    payload = lambda p: dict(body, model=info["model_id"], prompt=p[0], max_tokens=p[2], logprobs=None,  # noqa: E731
                             repetition_penalty=1.0)
    core.say("Starting initial single prompt test run...")
    test = core.stream(url, "/v1/completions", payload(prompts[0]))
    if not test.success:
        raise SystemExit(f"Initial test run failed: {test.error}")
    core.say("Initial test run completed. Starting main benchmark run...")
    core.say(f"Traffic request rate: inf\nMaximum request concurrency: {users}")
    slog = core.ServerLog(url, args.server_log)
    log_start = slog.offset()

    sem = threading.Semaphore(users)
    outs = [None] * n
    lock = threading.Lock()
    done = {"n": 0, "tok": 0}
    t0 = time.perf_counter()

    def one(i):
        with sem:
            r = core.stream(url, "/v1/completions", payload(prompts[i]))
        outs[i] = (r, prompts[i][1])
        with lock:
            done["n"] += 1
            done["tok"] += r.output_tokens if r.success else 0
            el = time.perf_counter() - t0
            core.say(f"  [{done['n']:>4}/{n}] t={el:7.1f}s req {i:>4}: ttft {r.ttft * 1000:8.1f} ms, "
                     f"out {r.output_tokens:>5} tok, e2el {r.latency:7.2f} s"
                     f"{'' if r.success else '  FAILED ' + r.error[:100]}  | output tok/s {done['tok'] / el:8.1f}")

    threads = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(n)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    dur = time.perf_counter() - t0
    m, lens = calculate_metrics(outs, dur, pcts)
    sums = [o.summary() for o, _ in outs]
    ls = slog.stats(log_start)
    acc = core.acceptance(sums)
    extra = {"sampling": args.sampling, "ignore_eos": eos_note,
             "acceptance": acc if acc is not None else ls.get("acceptance"),
             "prompt_tps": ls.get("prompt_tps") or core.prompt_rate(sums)}
    lines = vllm_text(m, users, dur, extra)
    for ln in lines:
        core.say(ln)
    short = sum(1 for x in lens if 0 < x < output_len)
    headline = (f"{label} x{n} @{users}: {m['output_throughput']:.1f} out tok/s, TTFT {m['median_ttft_ms']:.0f} ms, "
                f"TPOT {m['median_tpot_ms']:.1f} ms, accept {core.fmt(None if extra['acceptance'] is None else 100 * extra['acceptance'], '.1f')}%")
    sub = (f"vLLM random dataset {input_len}/{output_len}, {n} prompts, {users} concurrent users (request rate inf), "
           f"seed {args.seed}, sampling <b>{args.sampling}</b>, ignore_eos {html.escape(eos_note)}.<br>"
           f"{core.server_line(info)}")
    body_html = (core.stats_block([(f"{m['output_throughput']:.1f}", "output tok/s"),
                                   (f"{m['total_token_throughput']:.1f}", "total tok/s"),
                                   (f"{m['median_ttft_ms']:.0f} ms", "median TTFT"),
                                   (f"{m['median_tpot_ms']:.1f} ms", "median TPOT"),
                                   (core.fmt(None if extra['acceptance'] is None else 100 * extra['acceptance'], '.1f') + "%", "draft acceptance"),
                                   (core.fmt(extra["prompt_tps"], ".0f"), "server prompt tok/s")])
                 + "<h2>vLLM output</h2><pre>" + html.escape("\n".join(lines)) + "</pre>"
                 + (f'<p class="sub">{short} of {n} requests ended before {output_len} tokens (end-of-sequence; '
                    "ignore_eos was not in effect).</p>" if short else ""))
    data = {"server": info, "input_len": input_len, "output_len": output_len, "num_prompts": n, "users": users,
            "seed": args.seed, "sampling": args.sampling, "sampling_body": body, "ignore_eos": eos_note,
            "duration": dur, "metrics": m, "output_lens": lens, "requests": sums, "vllm_text": lines,
            "acceptance": extra["acceptance"], "server_prompt_tps": extra["prompt_tps"], "ended_short": short,
            "server_log": ls}
    return headline, data, body_html, f"Synthetic serving bench {label}", sub
