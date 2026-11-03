#!/usr/bin/env python3
"""The held-out history (README step 72): every point of results/heldout-history/points.tsv with
its per-prompt generation speed, as a table (history.tsv) and a page (history.html: the mean by
step, each prompt's change from the previous point, the table).

    python3 scripts/heldout_history.py

A point's data: runs/<step>-<commit>/ (written by scripts/heldout-rerun.sh) if complete, else
its `source` directory (measured at the time); a point with neither is pending. Only the standard library.
"""
import html
import json
import os
import re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
HIST = os.path.join(ROOT, "results", "heldout-history")
PROMPTS = [l.split("|")[0] for l in open(os.path.join(ROOT, "results/spec-heldout/prompts.txt")) if l.strip()]
LINE = re.compile(r"^([a-z-]+): ([0-9.?]+) t/s, ([0-9.?-]+) tokens per step.*output (\S+)")


def read_summary(path):
    rows, header = {}, ""
    for l in open(path):
        if l.startswith("#") and not header:
            header = l[1:].strip()
        m = LINE.match(l)
        if m and m.group(1) in PROMPTS and m.group(2) != "?":
            tps = m.group(3)
            rows[m.group(1)] = {"tps": float(m.group(2)), "tok": float(tps) if tps not in ("-", "?") else None,
                                "same": m.group(4)}
    return rows, header


def main():
    points = []
    for l in open(os.path.join(HIST, "points.tsv")):
        if l.startswith("#") or not l.strip():
            continue
        step, commit, status, flags, source, summary = l.rstrip("\n").split("\t")
        p = {"step": step, "commit": commit, "status": status, "flags": flags, "summary": summary, "prompts": {},
             "where": "", "header": ""}
        cands = [os.path.join(HIST, "runs", f"{step}-{commit}", "summary.txt")]  # a rerun wins
        if source:
            cands.append(os.path.join(ROOT, source, "summary.txt"))
        for c in cands:
            if os.path.exists(c):
                rows, header = read_summary(c)
                if len(rows) == len(PROMPTS):
                    p["prompts"], p["header"] = rows, header
                    p["where"] = os.path.relpath(os.path.dirname(c), ROOT)
                    break
        p["mean"] = round(sum(r["tps"] for r in p["prompts"].values()) / len(PROMPTS), 2) if p["prompts"] else None
        points.append(p)

    with open(os.path.join(HIST, "history.tsv"), "w") as f:
        f.write("# scripts/heldout_history.py: held-out generation t/s per point (8 prompts, 300 tokens, sclk 6); "
                "empty: pending\n")
        f.write("\t".join(["step", "commit", "mean"] + PROMPTS + ["data"]) + "\n")
        for p in points:
            vals = [f"{p['prompts'][n]['tps']:.2f}" if p["prompts"] else "" for n in PROMPTS]
            f.write("\t".join([p["step"], p["commit"], f"{p['mean']:.2f}" if p["mean"] else ""] + vals + [p["where"]]) + "\n")

    data = json.dumps({"prompts": PROMPTS, "points": points})
    page = TEMPLATE.replace("/*DATA*/null", data.replace("</", "<\\/"))
    with open(os.path.join(HIST, "history.html"), "w") as f:
        f.write(page)
    have = [p for p in points if p["mean"]]
    print(f"{len(have)} of {len(points)} points with data; history.tsv and history.html in {os.path.relpath(HIST)}")
    for p in have:
        print(f"  {p['step']:>4} {p['commit']:9s} {p['mean']:6.2f}")


TEMPLATE = r"""<title>Held-out Decode History</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>
/* layout: one reading column; the chart, then the per-prompt grid, then the table, each scrolling sideways on its own */
:root {
  --bg: #f7f7f4; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --line: #2a78d6; --pending: #c3c2b7; --ring: rgba(11,11,11,0.10);
  --neutral: #f0efec; --up: #2a78d6; --down: #e34948;
  --sans: system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --line: #3987e5; --pending: #52514e; --ring: rgba(255,255,255,0.10);
  --neutral: #383835; --up: #3987e5; --down: #e66767; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --line: #3987e5; --pending: #52514e; --ring: rgba(255,255,255,0.10);
  --neutral: #383835; --up: #3987e5; --down: #e66767; color-scheme: dark }
body { background: var(--bg); color: var(--ink); font-family: var(--sans); }
main { max-width: 1100px; margin: 0 auto; padding-block: 28px 48px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: flex; flex-direction: column; gap: 28px; }
h1 { font-size: 1.6rem; margin: 0; text-wrap: balance; letter-spacing: -0.01em; }
h2 { font-size: 1.05rem; margin: 0; }
.lede { color: var(--ink-2); margin: 6px 0 0; max-width: 70ch; line-height: 1.5; }
.facts { display: flex; flex-wrap: wrap; gap: 8px 24px; margin-top: 14px; color: var(--ink-2); font-size: 0.85rem; }
.facts b { color: var(--ink); font-variant-numeric: tabular-nums; font-size: 1.05rem; }
section { background: var(--surface); border: 1px solid var(--ring); border-radius: 8px; padding: 18px; display: flex; flex-direction: column; gap: 10px; min-width: 0; }
.note { color: var(--ink-2); font-size: 0.85rem; margin: 0; line-height: 1.45; max-width: 80ch; }
.scroll { overflow-x: auto; }
svg text { font-family: var(--sans); }
.mono { font-family: var(--mono); font-size: 0.82rem; }
table { border-collapse: collapse; font-size: 0.82rem; font-variant-numeric: tabular-nums; width: 100%; }
th, td { padding: 6px 8px; border-bottom: 1px solid var(--grid); text-align: right; white-space: nowrap; }
th { color: var(--ink-2); font-weight: 600; position: sticky; top: 0; background: var(--surface); }
td.l, th.l { text-align: left; }
td.what { white-space: normal; min-width: 260px; max-width: 420px; color: var(--ink-2); text-align: left; }
tr.pending td { color: var(--muted); }
.chip { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 0.72rem; border: 1px solid var(--axis); color: var(--ink-2); }
.legend { display: flex; align-items: center; gap: 8px; font-size: 0.78rem; color: var(--ink-2); flex-wrap: wrap; }
.ramp { display: inline-flex; border-radius: 3px; overflow: hidden; border: 1px solid var(--ring); }
.ramp span { width: 22px; height: 12px; }
#tip { position: fixed; pointer-events: none; background: var(--surface); color: var(--ink); border: 1px solid var(--axis);
  border-radius: 6px; padding: 8px 10px; font-size: 0.8rem; max-width: 320px; line-height: 1.4; box-shadow: 0 4px 14px rgba(0,0,0,0.12); }
#tip b { font-variant-numeric: tabular-nums; }
</style>
<main>
  <header>
    <h1>Held-out decode history</h1>
    <p class="lede">DeepSeek-V4.1-Flash-1017 on a Mac Pro 2019 (4 Vega II dies, CPU experts on a 16-core Xeon). Each point is one commit run on the same 8 held-out prompts: 300 generated tokens each, GPU clock level 6. Points marked pending are queued for the overnight rerun of old commits.</p>
    <div class="facts" id="facts"></div>
  </header>
  <section>
    <h2>Mean generation speed by step</h2>
    <div class="scroll"><svg id="line" role="img" aria-label="Mean generation speed by step"></svg></div>
    <p class="note">Mean of the 8 prompts, tokens per second. Hollow marks on the axis are pending points. Steps 20 onward use speculative decoding; steps 32 onward the adaptive expert cache with byte-identical outputs.</p>
  </section>
  <section>
    <h2>Change per prompt from the previous point</h2>
    <div class="legend"><span>slower</span><span class="ramp" id="ramp"></span><span>faster</span><span>&middot; clamped at &plusmn;10%; prompts ordered by tokens per step at the latest point (most speculation-friendly first)</span></div>
    <div class="scroll"><svg id="heat" role="img" aria-label="Per-prompt change from the previous point"></svg></div>
    <p class="note">A column of one color is a change every prompt feels. A column that splits by row is prompt-specific: rows near the top accept the most drafted tokens per step, so speculation-side changes land there first.</p>
  </section>
  <section>
    <h2>Points</h2>
    <div class="scroll"><table id="tbl"></table></div>
  </section>
</main>
<div id="tip" hidden></div>
<script>
const DATA = /*DATA*/null;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
const pts = DATA.points, P = DATA.prompts;
const have = pts.filter(p => p.mean);
const last = have[have.length - 1], first = have[0];
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

// facts
document.getElementById("facts").innerHTML =
  `<span>first measured: <b>${first.mean.toFixed(2)}</b> t/s (step ${first.step})</span>` +
  `<span>latest: <b>${last.mean.toFixed(2)}</b> t/s (step ${last.step})</span>` +
  `<span>points: <b>${have.length}</b> with data, <b>${pts.length - have.length}</b> pending</span>`;

// tooltip
const tip = document.getElementById("tip");
function showTip(ev, htmlStr) { tip.innerHTML = htmlStr; tip.hidden = false; moveTip(ev); }
function moveTip(ev) { const w = tip.offsetWidth, h = tip.offsetHeight; let x = ev.clientX + 14, y = ev.clientY + 14;
  if (x + w > innerWidth - 8) x = ev.clientX - w - 14; if (y + h > innerHeight - 8) y = ev.clientY - h - 14; tip.style.left = x + "px"; tip.style.top = y + "px"; }
function hideTip() { tip.hidden = true; }
function bindTip(node, fn) { node.addEventListener("mouseenter", e => showTip(e, fn())); node.addEventListener("mousemove", moveTip); node.addEventListener("mouseleave", hideTip); }
const pointTip = p => `<div><b>Step ${esc(p.step)}</b> <span class="mono">${esc(p.commit)}</span></div><div>${esc(p.summary)}</div>` +
  (p.mean ? `<div style="margin-top:4px">mean <b>${p.mean.toFixed(2)}</b> t/s</div>` : `<div style="margin-top:4px">pending: queued for the rerun</div>`);

// line chart: every point gets a slot, so the layout holds when pending points fill in
function drawLine() {
  const svg = document.getElementById("line"); svg.innerHTML = "";
  const slot = 40, L = 44, R = 16, T = 14, B = 46, H = 230;
  const W = Math.max(640, L + R + slot * pts.length);
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", W); svg.setAttribute("height", H);
  const maxV = Math.max(...have.map(p => p.mean)); const top = Math.ceil(maxV / 10) * 10;
  const y = v => T + (H - T - B) * (1 - v / top);
  const x = i => L + slot * i + slot / 2;
  for (let v = 0; v <= top; v += 10) {
    el("line", {x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: v ? css("--grid") : css("--axis"), "stroke-width": 1}, svg);
    el("text", {x: L - 8, y: y(v) + 4, "text-anchor": "end", "font-size": 11, fill: css("--muted")}, svg).textContent = v;
  }
  el("text", {x: 4, y: T + 2, "font-size": 11, fill: css("--muted")}, svg).textContent = "t/s";
  let d = "";
  pts.forEach((p, i) => { if (p.mean) d += (d ? "L" : "M") + x(i) + "," + y(p.mean); });
  el("path", {d, fill: "none", stroke: css("--line"), "stroke-width": 2, "stroke-linejoin": "round"}, svg);
  pts.forEach((p, i) => {
    el("text", {x: x(i), y: H - B + 18, "text-anchor": "middle", "font-size": 11, fill: css("--ink-2")}, svg).textContent = p.step;
    const g = el("g", {}, svg);
    if (p.mean) {
      el("circle", {cx: x(i), cy: y(p.mean), r: 4.5, fill: css("--line"), stroke: css("--surface"), "stroke-width": 2}, g);
      if (p === first || p === last)
        el("text", {x: x(i), y: y(p.mean) - 10, "text-anchor": "middle", "font-size": 12, "font-weight": 600, fill: css("--ink")}, g).textContent = p.mean.toFixed(1);
    } else {
      el("circle", {cx: x(i), cy: y(0), r: 4.5, fill: css("--surface"), stroke: css("--pending"), "stroke-width": 2}, g);
    }
    el("rect", {x: x(i) - slot / 2, y: T, width: slot, height: H - T - B + 24, fill: "transparent"}, g);
    bindTip(g, () => pointTip(p));
  });
  el("text", {x: L, y: H - 8, "font-size": 11, fill: css("--muted")}, svg).textContent = "README step";
}

// diverging scale: --down .. --neutral .. --up, clamped at +-10%
function mix(a, b, t) { const pa = [1,3,5].map(i => parseInt(a.slice(i, i + 2), 16)), pb = [1,3,5].map(i => parseInt(b.slice(i, i + 2), 16));
  return "#" + pa.map((v, k) => Math.round(v + (pb[k] - v) * t).toString(16).padStart(2, "0")).join(""); }
function divColor(pct) { const t = Math.max(-1, Math.min(1, pct / 10)); return t >= 0 ? mix(css("--neutral"), css("--up"), t) : mix(css("--neutral"), css("--down"), -t); }
function inkOn(hex) { const [r, g, b] = [1,3,5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255); const l = 0.2126*r + 0.7152*g + 0.0722*b; return l < 0.5 ? "#ffffff" : "#0b0b0b"; }

function drawHeat() {
  const ramp = document.getElementById("ramp"); ramp.innerHTML = "";
  [-10, -6, -3, 0, 3, 6, 10].forEach(v => { const s = document.createElement("span"); s.style.background = divColor(v); ramp.appendChild(s); });
  const svg = document.getElementById("heat"); svg.innerHTML = "";
  const cols = have.slice(1), order = [...P].sort((a, b) => (last.prompts[b].tok || 0) - (last.prompts[a].tok || 0));
  const cw = 52, ch = 26, gap = 2, L = 150, T = 22, B = 8;
  const W = Math.max(600, L + cols.length * cw + 8), H = T + order.length * ch + B;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", W); svg.setAttribute("height", H);
  cols.forEach((p, j) => el("text", {x: L + j * cw + cw / 2, y: 14, "text-anchor": "middle", "font-size": 11, fill: css("--ink-2")}, svg).textContent = p.step);
  order.forEach((n, i) => {
    const tok = last.prompts[n].tok;
    el("text", {x: L - 10, y: T + i * ch + ch / 2 + 4, "text-anchor": "end", "font-size": 12, fill: css("--ink")}, svg).textContent = n + (tok ? `  ${tok.toFixed(1)} tok/step` : "");
    cols.forEach((p, j) => {
      const prev = have[have.indexOf(p) - 1], pct = (p.prompts[n].tps / prev.prompts[n].tps - 1) * 100, fill = divColor(pct);
      const g = el("g", {}, svg);
      el("rect", {x: L + j * cw + gap / 2, y: T + i * ch + gap / 2, width: cw - gap, height: ch - gap, rx: 3, fill}, g);
      el("text", {x: L + j * cw + cw / 2, y: T + i * ch + ch / 2 + 4, "text-anchor": "middle", "font-size": 11, fill: inkOn(fill), "font-variant-numeric": "tabular-nums"}, g).textContent = (pct >= 0 ? "+" : "") + pct.toFixed(1);
      bindTip(g, () => `<div><b>${esc(n)}</b>, step ${esc(p.step)} vs ${esc(prev.step)}</div><div><b>${prev.prompts[n].tps.toFixed(2)} &rarr; ${p.prompts[n].tps.toFixed(2)}</b> t/s (${pct >= 0 ? "+" : ""}${pct.toFixed(1)}%)</div><div style="color:var(--ink-2)">${esc(p.summary)}</div>`);
    });
  });
}

function drawTable() {
  const t = document.getElementById("tbl");
  let h = `<thead><tr><th class="l">step</th><th class="l">commit</th><th class="l">change</th><th>mean</th>${P.map(n => `<th>${esc(n)}</th>`).join("")}</tr></thead><tbody>`;
  [...pts].reverse().forEach(p => {
    const cells = P.map(n => `<td>${p.mean ? p.prompts[n].tps.toFixed(2) : ""}</td>`).join("");
    h += `<tr class="${p.mean ? "" : "pending"}"><td class="l">${esc(p.step)}</td><td class="l mono">${esc(p.commit)}</td><td class="what">${esc(p.summary)}${p.mean ? "" : ` <span class="chip">${esc(p.status === "optional" ? "optional rerun" : "pending")}</span>`}</td><td><b>${p.mean ? p.mean.toFixed(2) : ""}</b></td>${cells}</tr>`;
  });
  t.innerHTML = h + "</tbody>";
}
function drawAll() { drawLine(); drawHeat(); }
drawAll(); drawTable();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawAll);
new MutationObserver(drawAll).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
</script>
"""

if __name__ == "__main__":
    main()
