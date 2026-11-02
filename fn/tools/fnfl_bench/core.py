"""Shared pieces of fnfl-bench: HTTP and streaming, sampling modes, run folders, pages and the index.

Host-side only, against an OpenAI-compatible llama-server. Standard library only here (numpy is needed only by the
synthetic bench's prompt generator)."""

import datetime
import html
import http.client
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request

# --------------------------------------------------------------------------------------------- terminal

_print_lock = threading.Lock()


def say(*parts):
    """One progress line, flushed at once so the run can be watched live."""
    with _print_lock:
        print(*parts, flush=True)


def utc_stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_iso():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


# --------------------------------------------------------------------------------------------- HTTP


def http_json(url, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def health(url, timeout=5):
    try:
        return http_json(url + "/health", timeout=timeout).get("status") == "ok"
    except Exception:  # noqa: BLE001 - down or loading
        return False


def server_info(url):
    """What the bench records about the server: build, model, slots."""
    info = {"url": url}
    try:
        p = http_json(url + "/props")
        info.update(build=p.get("build_info"), model=os.path.basename(p.get("model_path") or ""),
                    slots=p.get("total_slots"))
    except Exception as e:  # noqa: BLE001
        info["props_error"] = repr(e)
    try:
        info["model_id"] = http_json(url + "/v1/models")["data"][0]["id"]
    except Exception:  # noqa: BLE001
        info["model_id"] = info.get("model") or "model"
    return info


class StreamResult:
    """One streamed request, timed on the client (time.perf_counter), plus what the server reports at its end."""

    def __init__(self):
        self.start = 0.0          # perf_counter when the request was sent
        self.ttft = 0.0           # first content chunk
        self.latency = 0.0        # last chunk
        self.itl = []             # gaps between content chunks
        self.chunk_times = []     # perf_counter of every content chunk
        self.text = ""
        self.reasoning = ""
        self.usage = None
        self.timings = None
        self.finish = None
        self.error = ""
        self.success = False

    @property
    def output_tokens(self):
        if self.usage and self.usage.get("completion_tokens") is not None:
            return int(self.usage["completion_tokens"])
        return len(self.chunk_times)

    def summary(self):
        t = self.timings or {}
        return {"ttft": self.ttft, "latency": self.latency, "output_tokens": self.output_tokens,
                "finish": self.finish, "error": self.error, "success": self.success,
                "prompt_tokens": (self.usage or {}).get("prompt_tokens"),
                "decode_tps": t.get("predicted_per_second"), "prompt_tps": t.get("prompt_per_second"),
                "prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"),
                "draft_n": t.get("draft_n"), "draft_accepted": t.get("draft_n_accepted")}


def stream(url, path, payload, stop_event=None, on_chunk=None, timeout=6 * 3600):
    """POST a streaming request and time it. Content chunks are any chunk whose first choice carries text, a content
    delta or a reasoning delta (vLLM counts a chunk with choices as a token event; so does this). usage and timings
    are read from whichever chunk carries them (llama-server puts them on the final chunk, beside its choices).
    stop_event: closing the connection when it is set ends the request on the server."""
    r = StreamResult()
    u = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    body = json.dumps(dict(payload, stream=True, stream_options={"include_usage": True})).encode()
    r.start = time.perf_counter()
    last = r.start
    try:
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            r.error = f"HTTP {resp.status}: {resp.read()[:400].decode('utf-8', 'replace')}"
            return r
        first = False
        for raw in resp:
            if stop_event is not None and stop_event.is_set():
                r.error = "stopped by the bench"
                break
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                data = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if "error" in data and not data.get("choices"):
                r.error = json.dumps(data["error"])[:400]
                break
            if data.get("usage"):
                r.usage = data["usage"]
            if data.get("timings"):
                r.timings = data["timings"]
            choices = data.get("choices")
            if choices:
                c = choices[0]
                if c.get("finish_reason"):
                    r.finish = c["finish_reason"]
                delta = c.get("delta") or {}
                text = c.get("text")
                if text is None:
                    text = delta.get("content")
                rtext = delta.get("reasoning_content")
                now = time.perf_counter()
                if not first:
                    first = True
                    r.ttft = now - r.start
                else:
                    r.itl.append(now - last)
                last = now
                r.chunk_times.append(now)
                r.text += text or ""
                r.reasoning += rtext or ""
                if on_chunk:
                    on_chunk(r)
        r.success = first and not r.error
        if not first and not r.error:
            r.error = "Never received a valid chunk to calculate TTFT."
        r.latency = last - r.start
    except Exception as e:  # noqa: BLE001 - recorded on the request; the others go on
        r.error = repr(e)
        r.latency = last - r.start
    finally:
        conn.close()
    return r


# --------------------------------------------------------------------------------------------- sampling

PRODUCTION = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.05}
SAMPLING = {
    "vllm": {"temperature": 0.0},     # vLLM's CI: T 0 (greedy) with ignore_eos
    "production": dict(PRODUCTION),   # the launch sampling, with ignore_eos
}


def ignore_eos_supported(url, model):
    """Ask the server once (one token) whether it takes a request's ignore_eos. FNFL refuses it with a 400."""
    try:
        http_json(url + "/v1/completions", {"model": model, "prompt": "hi", "max_tokens": 1, "temperature": 0.0,
                                            "ignore_eos": True}, timeout=120)
        return True, ""
    except urllib.error.HTTPError as e:
        return False, e.read().decode("utf-8", "replace")[:300]
    except Exception as e:  # noqa: BLE001
        return False, repr(e)


def sampling_body(mode, url, model, want_ignore_eos=True):
    """The request fields for a sampling mode, and a note on ignore_eos for the report."""
    body = dict(SAMPLING[mode])
    note = "not asked"
    if want_ignore_eos:
        ok, why = ignore_eos_supported(url, model)
        if ok:
            body["ignore_eos"] = True
            note = "sent"
        else:
            note = "refused by the server, not sent: output lengths are not fixed"
            say(f"WARNING: the server refuses ignore_eos ({why.strip()[:160]}); sending without it, so a request can "
                "end early at end-of-sequence. Generated-token totals show the effect.")
    return body, note


# --------------------------------------------------------------------------------------------- numbers


def percentile(vals, p):
    """numpy.percentile's default (linear) method, for the pieces that do not import numpy."""
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def median(vals):
    return percentile(vals, 50)


def acceptance(summaries):
    """Draft acceptance over requests from the server's own timings: accepted / proposed draft tokens."""
    dn = sum(s.get("draft_n") or 0 for s in summaries)
    da = sum(s.get("draft_accepted") or 0 for s in summaries)
    return (da / dn) if dn else None


def prompt_rate(summaries):
    """The server's prompt processing rate over requests: prompt tokens processed / prompt ms (its timings)."""
    n = sum(s.get("prompt_n") or 0 for s in summaries)
    ms = sum(s.get("prompt_ms") or 0 for s in summaries)
    return (n / (ms / 1000.0)) if ms else None


def fmt(v, spec=".2f", none="n/a"):
    return none if v is None else format(v, spec)


# --------------------------------------------------------------------------------------------- run folders


class Run:
    """runs/NAME-<UTC time>/: result.json (the numbers) and index.html (the page)."""

    def __init__(self, out_dir, name):
        self.out_dir = os.path.abspath(out_dir)
        self.name = name
        self.stamp = utc_stamp()
        self.dir = os.path.join(self.out_dir, f"{name}-{self.stamp}")
        os.makedirs(self.dir, exist_ok=True)
        self.started = utc_iso()

    def path(self, *p):
        return os.path.join(self.dir, *p)

    def finish(self, bench, headline, data, body_html, title, subtitle):
        data = dict(data, bench=bench, name=self.name, started=self.started, finished=utc_iso(), headline=headline,
                    command=" ".join(sys.argv))
        with open(self.path("result.json"), "w") as f:
            json.dump(data, f, indent=1, default=str)
        with open(self.path("index.html"), "w") as f:
            f.write(page(title, subtitle, body_html))
        write_index(self.out_dir)
        say(f"page: {self.path('index.html')}")
        say(f"index: {os.path.join(self.out_dir, 'index.html')}")


def write_index(out_dir):
    rows = []
    for d in os.listdir(out_dir):
        rp = os.path.join(out_dir, d, "result.json")
        if not os.path.isfile(rp):
            continue
        try:
            r = json.load(open(rp))
        except (OSError, ValueError):
            continue
        rows.append((d.rsplit("-", 1)[-1], d, r))
    rows.sort(key=lambda x: x[0], reverse=True)
    trs = "".join(
        f'<tr><td><a href="{html.escape(d)}/index.html">{html.escape(r.get("name", d))}</a></td>'
        f'<td>{html.escape(r.get("bench", ""))}</td><td class="num">{html.escape(r.get("started", ""))}</td>'
        f'<td>{html.escape(str((r.get("server") or {}).get("build") or ""))}</td>'
        f'<td>{html.escape(r.get("headline", ""))}</td></tr>' for _, d, r in rows)
    body = (f'<div class="wrap"><table><thead><tr><th>Run</th><th>Bench</th><th>Started (UTC)</th><th>Build</th>'
            f'<th>Headline</th></tr></thead><tbody>{trs}</tbody></table></div>')
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(page("Bench runs", f"{len(rows)} runs, newest first.", body))


# --------------------------------------------------------------------------------------------- pages

CSS = """
:root{--bg:#f6f5f2;--fg:#1c1b19;--mut:#6b6760;--card:#fff;--line:#e3dfd7;--acc:#c75a1a;--ok:#2f7d4f;--bad:#b3261e;
--s1:#2f6db5;--s2:#c75a1a;--s3:#2f7d4f;--s4:#8a4fb5;--s5:#b58a1f;--s6:#1f9aa0;--s7:#b53f6d;--s8:#5a6b2f;--band:#c75a1a22}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#151514;--fg:#ecebe7;--mut:#a29e96;
--card:#211f1d;--line:#36332f;--acc:#f08a45;--ok:#5fbf86;--bad:#ef6b62;--s1:#6ea6ec;--s2:#f08a45;--s3:#5fbf86;
--s4:#b98ae6;--s5:#e3bd55;--s6:#55cdd2;--s7:#ea7aa6;--s8:#a8bd6a;--band:#f08a4526}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,-apple-system,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 48px}h1{margin:0 0 4px;font-size:26px}h2{margin:28px 0 10px;font-size:19px}
.sub{color:var(--mut);margin:0 0 18px;max-width:880px}a{color:var(--acc)}
.wrap{overflow-x:auto}table{border-collapse:collapse;background:var(--card);border:1px solid var(--line);width:100%;
font-variant-numeric:tabular-nums;margin-bottom:12px}th,td{padding:7px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
thead th{font-weight:650}td.num,th.num{text-align:right}tr:last-child td{border-bottom:0}
pre{background:var(--card);border:1px solid var(--line);padding:12px;overflow-x:auto;font-size:13px;line-height:1.35}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;background:var(--line);
border:1px solid var(--line);border-radius:10px;overflow:hidden;margin-bottom:16px}
.stat{background:var(--card);padding:12px 14px}.stat .v{font-size:21px;font-weight:650;font-variant-numeric:tabular-nums}
.stat .l{color:var(--mut);font-size:13px}.ok{color:var(--ok);font-weight:650}.bad{color:var(--bad);font-weight:650}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
figure img,.none{display:block;width:100%;aspect-ratio:4/3;object-fit:contain;background:#fff;border-bottom:1px solid var(--line)}
.none{display:grid;place-items:center;color:#888}figcaption{padding:10px 12px}
dl{display:grid;grid-template-columns:auto 1fr;gap:2px 12px;margin:0;font-size:13px}dt{color:var(--mut)}
dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}.chart{background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:8px}.chart svg{width:100%;height:auto;display:block}.legend{font-size:13px;color:var(--mut)}
.foot{color:var(--mut);font-size:13px;margin-top:28px}
"""


def page(title, subtitle, body):
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>'
            f'<style>{CSS}</style></head><body><main><h1>{html.escape(title)}</h1>'
            f'<p class="sub">{subtitle}</p>{body}</main></body></html>')


def stats_block(items):
    return '<div class="stats">' + "".join(
        f'<div class="stat"><div class="v">{html.escape(str(v))}</div><div class="l">{html.escape(l)}</div></div>'
        for v, l in items) + "</div>"


def table(headers, rows, num_cols=()):
    th = "".join(f'<th class="{"num" if i in num_cols else ""}">{html.escape(h)}</th>' for i, h in enumerate(headers))
    trs = "".join("<tr>" + "".join(f'<td class="{"num" if i in num_cols else ""}">{c}</td>' for i, c in enumerate(r))
                  + "</tr>" for r in rows)
    return f'<div class="wrap"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


def server_line(info):
    return html.escape(f"{info.get('url')} | build {info.get('build') or '?'} | {info.get('model') or '?'} | "
                       f"{info.get('slots') or '?'} slots")


# --------------------------------------------------------------------------------------------- the server's log


class ServerLog:
    """The server's log (its stderr), found from the llama-server process serving the URL's port, or given. FNFL's
    per-request timings carry no draft counts, so draft acceptance comes from its "FNFL request:" lines, and the
    prompt rate from "prompt eval time" lines, over the byte window of a run."""

    def __init__(self, url, path=None):
        self.path, self.spec_d = path, None
        port = urllib.parse.urlparse(url).port or 80
        for pid in (p for p in os.listdir("/proc") if p.isdigit()):
            try:
                args = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
            except OSError:
                continue
            args = [a.decode("utf-8", "replace") for a in args if a]
            if not args or not os.path.basename(args[0]).startswith("llama-server"):
                continue
            if ("--port" in args and args[args.index("--port") + 1] != str(port)) or ("--port" not in args and port != 8080):
                continue
            if "--fnfl-spec-d" in args:
                self.spec_d = int(args[args.index("--fnfl-spec-d") + 1])
            if self.path is None:
                try:
                    self.path = os.readlink(f"/proc/{pid}/fd/2")
                except OSError:
                    pass
            break
        if self.path and not os.path.isfile(self.path):
            self.path = None

    def offset(self):
        return os.path.getsize(self.path) if self.path else 0

    def stats(self, start):
        if not self.path:
            return {"server_log": None}
        import re

        time.sleep(0.5)  # the per-request lines follow the response
        with open(self.path, "rb") as f:
            f.seek(start)
            text = f.read().decode("utf-8", "replace")
        p_tok = p_ms = acc = steps = 0.0
        for line in text.splitlines():
            m = re.search(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens", line)
            if m:
                p_ms += float(m.group(1))
                p_tok += int(m.group(2))
            if line.startswith("FNFL request:"):
                ms = re.search(r"speculative (\d+),", line)
                ma = re.search(r"accepted drafts a speculative step ([\d.]+)", line)
                if ms and ma:
                    steps += int(ms.group(1))
                    acc += float(ma.group(1)) * int(ms.group(1))
        d = self.spec_d or 0
        return {"server_log": self.path, "spec_d": self.spec_d,
                "prompt_tps": (p_tok / (p_ms / 1000.0)) if p_ms else None,
                "acceptance": (acc / (d * steps)) if steps and d else None,
                "accepted_per_step": (acc / steps) if steps else None}
