# -*- coding: utf-8 -*-
"""Stylesheet (inlined into the single-file HTML)."""

CSS = r"""
@page { size: A4; margin: 0; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: "Noto Sans CJK SC", "Source Han Sans SC", sans-serif;
  color: #3c4a5a; background: #fff;
  -webkit-font-smoothing: antialiased;
}
:root {
  --ink: #16202b; --body: #3c4a5a; --muted: #7d8b9c;
  --line: #e7ecf3; --soft: #f6f8fb;
  --red: #d64545; --orange: #e08a2e; --green: #2f9e6e; --blue: #2b6cb0;
}

.page {
  width: 794px; min-height: 1123px; padding: 34px 44px 30px;
  position: relative; background: #fff; overflow: hidden;
}
.pfoot {
  position: absolute; left: 44px; right: 44px; bottom: 14px;
  display: flex; justify-content: space-between;
  font-size: 9px; color: #b3bfcc; letter-spacing: .3px;
  border-top: 1px solid #f0f3f7; padding-top: 5px;
}

/* ---------------- Cover header ---------------- */
.cover { border-bottom: 2px solid var(--ink); padding-bottom: 10px; margin-bottom: 11px; }
.cover h1 { font-size: 24px; line-height: 1.35; color: var(--ink); margin: 0; font-weight: 700; letter-spacing: .3px; }
.cover .sub { font-size: 12px; color: var(--muted); margin-top: 5px; letter-spacing: 1.5px; }
.cover .meta { display: flex; gap: 7px; margin-top: 11px; flex-wrap: wrap; }
.cover .meta .meta-label { background: none; border: none; padding: 2px 0 2px 0; color: var(--muted); font-size: 10px; }
.cover .meta span {
  font-size: 10.5px; color: #4a5a6c; background: var(--soft);
  border: 1px solid var(--line); border-radius: 20px; padding: 2px 9px;
}

/* ---------------- Section header ---------------- */
.sec-head {
  display: flex; align-items: center; gap: 9px;
  margin: 0 0 12px; padding-bottom: 7px; border-bottom: 1px solid var(--line);
}
.sec-num {
  font-size: 10px; font-weight: 700; letter-spacing: .5px; color: #fff;
  background: var(--ink); padding: 2px 6px; border-radius: 3px;
}
.sec-ic { color: var(--blue); display: flex; }
.sec-head h2 { font-size: 18px; font-weight: 700; color: var(--ink); margin: 0; letter-spacing: .3px; }

.sub-head {
  font-size: 13.5px; font-weight: 700; color: var(--ink);
  margin: 12px 0 7px; padding-left: 9px; border-left: 3px solid var(--blue);
  line-height: 1.4;
}
.sub-head:first-child { margin-top: 0; }
.sub-head { break-after: avoid; page-break-after: avoid; }

/* ---------------- Body text ---------------- */
p.t { font-size: 12px; line-height: 1.85; margin: 0 0 9px; text-align: justify; }
p.t strong { color: var(--ink); font-weight: 700; }
.lead { font-size: 12.5px; line-height: 1.78; color: #2f3d4d; }

.callout {
  border-left: 3px solid var(--red); background: #fffafa;
  border-radius: 0 8px 8px 0; padding: 8px 11px; margin: 9px 0;
  font-size: 11.5px; line-height: 1.8;
}
.callout.info { border-left-color: var(--blue); background: #f7fbff; }
.callout.warn { border-left-color: var(--orange); background: #fffdf7; }
.callout.ok { border-left-color: #2f9e6e; background: #f7fdfa; }
.callout strong { color: var(--ink); }

ul.t { margin: 4px 0 9px; padding-left: 17px; }
ul.t li { font-size: 11.5px; line-height: 1.74; margin: 2px 0; }
ul.t li strong { color: var(--ink); }
ul.t ul { margin: 2px 0 2px; padding-left: 15px; }
ul.t ul li { font-size: 11px; color: #4d5c6d; }

ol.t { margin: 4px 0 9px; padding-left: 18px; }
ol.t li { font-size: 11.5px; line-height: 1.8; margin: 3px 0; }
ol.t li strong { color: var(--ink); }

/* ---------------- Metric strip ---------------- */
.metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 9px; margin: 0 0 12px; }
.metric { border: 1px solid var(--line); border-radius: 9px; padding: 8px 10px; background: #fff; }
.metric-label { font-size: 10px; color: var(--muted); letter-spacing: .2px; }
.metric-value { font-size: 19px; font-weight: 700; color: var(--ink); line-height: 1.25; margin-top: 1px; }
.metric-unit { font-size: 10px; font-weight: 500; color: var(--muted); margin-left: 2px; }
.metric-tag { font-size: 9.5px; margin-top: 2px; font-weight: 600; }
.metric.hi { border-color: #f3c9c9; background: #fffafa; }
.metric.hi .metric-tag { color: var(--red); }
.metric.mid { border-color: #f0dcb8; background: #fffdf7; }
.metric.mid .metric-tag { color: #a86a00; }
.metric.ok .metric-tag { color: var(--green); }

/* ---------------- Dashboard ---------------- */
.dash-sum { display: flex; gap: 8px; margin-bottom: 9px; }
.cover .dash-sum { margin-top: 14px; }
.sum-item {
  flex: 1; border: 1px solid var(--line); border-radius: 9px;
  padding: 7px 10px; background: #fff; display: flex; align-items: baseline; gap: 6px;
}
.sum-n { font-size: 17px; font-weight: 700; line-height: 1; }
.sum-l { font-size: 10px; color: var(--muted); }
.sum-item.hi .sum-n { color: var(--red); }
.sum-item.mid .sum-n { color: var(--orange); }
.sum-item.low .sum-n { color: var(--green); }
.sum-item.none .sum-n { color: #9aa8b8; }

.dz {
  border: 1px solid var(--line); border-radius: 11px;
  padding: 8px 12px 6px; margin-bottom: 8px; background: #fff;
}
.dz-high { border-color: #f2c6c6; background: linear-gradient(180deg, #fffafa 0%, #fff 60%); }
.dz-mid { border-color: #efd9b4; background: linear-gradient(180deg, #fffdf7 0%, #fff 60%); }
.dz-low { background: #fcfdfe; }
.dz-none { background: #fbfcfd; border-style: dashed; }
.dz-head { display: flex; align-items: center; gap: 7px; margin-bottom: 7px; }
.dz-ic { color: #6b7c8f; display: flex; }
.dz-name { font-size: 13.5px; font-weight: 700; color: var(--ink); }
.dz-vote { font-size: 9.5px; color: var(--muted); margin-left: auto; }
.dz-body { display: flex; gap: 14px; align-items: flex-start; }
.dz-gauges { display: flex; flex-wrap: wrap; gap: 8px; }

/* Band detail sidebar */
.glegend { flex: 1; min-width: 0; padding: 4px 0 0 2px; }
.glegend-h { font-size: 9.5px; color: var(--muted); margin-bottom: 5px; letter-spacing: .2px; }
.gl-row {
  display: flex; align-items: center; gap: 6px;
  font-size: 10px; color: #6b7c8f; padding: 2px 0;
}
.gl-row i { width: 9px; height: 9px; border-radius: 2px; flex: 0 0 auto; opacity: .55; }
.gl-row.on { color: var(--ink); font-weight: 700; }
.gl-row.on i { opacity: 1; box-shadow: 0 0 0 2px #fff, 0 0 0 3px currentColor; }
.glegend-note {
  font-size: 9px; color: #9aa8b8; margin-top: 5px;
  border-top: 1px dashed #e7ecf3; padding-top: 4px;
}

.gcard {
  border: 1px solid var(--line); border-radius: 9px;
  padding: 5px 8px 3px; background: #fff;
}
.gcard-head { display: flex; align-items: center; gap: 5px; margin-bottom: 1px; }
.gcard-name { font-size: 10px; color: #4d5c6d; font-weight: 600; white-space: nowrap; }

.chip { font-size: 9px; font-weight: 700; padding: 1px 5px; border-radius: 8px; white-space: nowrap; }
.c-high { background: #fdeaea; color: #b93a3a; }
.c-mid { background: #fdf3e2; color: #a86a00; }
.c-low { background: #e9f6ef; color: #1f7a4d; }
.c-none { background: #eef1f6; color: #6b7c8f; }

.gauge { display: block; }
.gauge-value { font-weight: 700; fill: #16202b; }
.gauge-tick { font-size: 8.5px; fill: #8a97a8; }

/* Low-risk compact grid */
.cover-meta { font-size: 10.5px; color: #4a5a6e; margin: 10px 0 14px; line-height: 1.8; }
.lowsec-h { break-after: avoid; page-break-after: avoid;
  font-size: 10.5px; font-weight: 700; color: #5b6b7f; letter-spacing: .5px;
  margin: 0 0 7px; display: flex; align-items: center; gap: 6px;
}
.lowsec-h::before { content: ''; width: 5px; height: 5px; border-radius: 50%; background: #30a46c; }
.lowgrid { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; }
.lowcell {
  border: 1px solid var(--line); border-radius: 9px; padding: 4px 7px 3px; background: #fcfdfe;
}
.lowcell-head {
  display: flex; align-items: center; gap: 5px; margin-bottom: 3px;
  font-size: 10.5px; font-weight: 700; color: #2f3d4d;
}
.lowcell-head .ic { color: #7d8b9c; flex: 0 0 auto; }
.lowcell-head > span:not(.chip) { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lowcell-gauges { display: flex; gap: 5px; flex-wrap: wrap; }
.lowcell .gcard { padding: 2px 3px 1px; border-radius: 7px; border-color: #f0f3f7; }
.lowcell .gcard-head { margin-bottom: 0; display: block; }
.lowcell .gcard-name {
  font-size: 8.5px; display: block; width: 100%; overflow: hidden;
  white-space: nowrap; text-overflow: ellipsis;
}

.g-nogauge { display: flex; flex-direction: column;
  align-items: center; justify-content: center; gap: 3px; background: #fafbfd; border-radius: 7px; }
.g-nogauge-val { font-size: 15px; font-weight: 700; color: #6b7c8f; }
.g-nogauge-note { font-size: 8px; color: #a3b0be; text-align: center; padding: 0 6px; line-height: 1.35; }

.dash-note {
  font-size: 10px; line-height: 1.75; color: #8a99ab;
  border-top: 1px dashed #dde5ef; padding-top: 5px; margin-top: 3px;
}
.dash-note b { color: #6b7c8f; }

/* ---------------- Abnormal finding cards ---------------- */
.fcard {
  display: flex; gap: 10px; border: 1px solid var(--line); border-radius: 9px;
  padding: 8px 11px; margin-bottom: 7px; background: #fff;
}
.fcard-ic { flex: 0 0 44px; display: flex; align-items: flex-start; padding-top: 1px; }
.pic { display: block; }
.fcard-main { flex: 1; min-width: 0; }
.fcard-top { display: flex; align-items: baseline; gap: 6px; }
.fcard-idx {
  font-size: 9px; font-weight: 700; color: #fff; background: var(--red);
  border-radius: 3px; padding: 1px 4px; flex: 0 0 auto;
}
.fcard-name { font-size: 12px; font-weight: 700; color: var(--ink); line-height: 1.45; }
.fcard-val { margin-left: auto; font-size: 14px; font-weight: 700; color: var(--red); white-space: nowrap; }
.fcard-val.norm { color: #6b7c8f; font-size: 11px; font-weight: 600; }
.fcard-ref { font-size: 9.5px; color: var(--muted); margin-top: 1px; }
.fcard-legend { font-size: 9px; color: #8a97a8; margin: 0 0 5px 2px; letter-spacing: .02em; }
.fcard-row { font-size: 10.5px; line-height: 1.65; color: #4d5c6d; margin-top: 3px; }
.fcard-row b { color: var(--ink); font-weight: 600; }

/* ---------------- Negative results ---------------- */
.neg-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; margin: 8px 0; }
.neg {
  border: 1px solid #e2f0e8; background: #fbfefc; border-radius: 8px;
  padding: 7px 9px; font-size: 10.5px; line-height: 1.65; color: #3f5a4c;
}
.neg b { color: #1f7a4d; font-weight: 700; }

/* ---------------- Mechanism chain ---------------- */
.mech { border: 1px solid var(--line); border-radius: 9px; padding: 9px 11px; margin-bottom: 8px; }
.mech-chain {
  font-size: 10.5px; font-weight: 700; color: var(--blue);
  letter-spacing: .2px; margin-bottom: 4px; display: flex; align-items: center; gap: 5px;
}
.mech-n {
  display: inline-block; width: 15px; height: 15px; border-radius: 50%;
  background: var(--blue); color: #fff; font-size: 9px; font-weight: 700;
  text-align: center; line-height: 15px; margin-right: 5px; flex: 0 0 auto;
}
.mech-chain::before {
  content: ''; display: inline-block; width: 5px; height: 5px;
  border-radius: 50%; background: var(--blue); margin-right: 6px;
  vertical-align: 2px;
}
.mech-body { font-size: 11px; line-height: 1.75; color: #4d5c6d; }
.mech-body strong { color: var(--ink); }

/* ---------------- Tables ---------------- */
table.hub { width: 100%; border-collapse: collapse; font-size: 10.5px; margin: 6px 0 10px; }
table.hub th {
  background: var(--soft); color: var(--ink); font-weight: 600;
  text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line);
  font-size: 10px;
}
table.hub td { padding: 6px 8px; border-bottom: 1px solid #f0f3f7; vertical-align: top; line-height: 1.6; }
table.hub td strong { color: var(--ink); }
table.hub tr:last-child td { border-bottom: none; }

/* ---------------- Strategy cards ---------------- */
.stg { border: 1px solid var(--line); border-radius: 9px; padding: 7px 11px; margin-bottom: 6px; }
.stg-title { font-size: 12px; font-weight: 700; color: var(--ink); margin-bottom: 5px; display: flex; gap: 6px; align-items: center; }
.stg-num {
  font-size: 9px; background: var(--blue); color: #fff;
  border-radius: 3px; padding: 1px 5px; font-weight: 700; flex: 0 0 auto;
}
.kv { display: grid; grid-template-columns: 52px 1fr; gap: 3px 7px; font-size: 10.5px; line-height: 1.7; }
.kv-k { color: var(--muted); }
.kv-v { color: #4d5c6d; }
.kv-v strong { color: var(--ink); }
.kv-v ul { margin: 1px 0; padding-left: 13px; }
.stg ul, .stg ol { margin: 1px 0; padding-left: 14px; }
.stg li { font-size: 10.5px; line-height: 1.7; margin: 1px 0; color: #4d5c6d; }
.stg li strong { color: var(--ink); }
.stg p { font-size: 10.5px; line-height: 1.7; margin: 2px 0; }

.kv-v li { margin: 1px 0; }

/* ---------------- Disease detail card ---------------- */
.dz5 { border: 1px solid var(--line); border-radius: 9px; padding: 7px 11px; margin-bottom: 7px; }
.dz5-head { display: flex; align-items: center; gap: 7px; margin-bottom: 6px; }
.dz5-name { font-size: 12.5px; font-weight: 700; color: var(--ink); }
.dz5-sub { font-size: 9.5px; color: var(--muted); margin-left: auto; }
.dz5-cols { display: grid; grid-template-columns: 0.86fr 1.14fr; gap: 12px; }
.dz5-col-h { font-size: 10px; font-weight: 700; color: var(--blue); margin-bottom: 3px; letter-spacing: .3px; }
.dz5 ul, .dz5 ol { margin: 0; padding-left: 14px; }
.dz5 li { font-size: 10.5px; line-height: 1.7; margin: 2px 0; color: #4d5c6d; }
.dz5 li strong { color: var(--ink); }

/* ---------------- Action items ---------------- */
.act { display: flex; gap: 9px; border: 1px solid var(--line); border-radius: 8px; padding: 5px 10px; margin-bottom: 4px; }
.act-pri {
  flex: 0 0 20px; height: 20px; border-radius: 50%; background: var(--red); color: #fff;
  font-size: 10px; font-weight: 700; display: flex; align-items: center; justify-content: center;
}
.act-pri.p2 { background: var(--orange); }
.act-pri.p3 { background: #c9a227; }
.act-pri.p4 { background: #7d9bb5; }
.act-pri.p5 { background: #9aa8b8; }
.act-pri.p6 { background: #b3bfcc; }
.act-main { flex: 1; min-width: 0; }
.act-title { font-size: 11.5px; font-weight: 700; color: var(--ink); line-height: 1.5; }
.act-why { font-size: 10px; line-height: 1.6; color: #5b6b7f; margin-top: 2px; }
.act-meta { display: flex; gap: 12px; margin-top: 3px; font-size: 9.5px; color: var(--muted); }
.act-meta b { color: #4d5c6d; font-weight: 600; }

/* ---------------- Miscellaneous ---------------- */
.rule { border: none; border-top: 1px solid var(--line); margin: 12px 0; }
.tiny { font-size: 9.5px; color: var(--muted); line-height: 1.7; }
.spacer { height: 8px; }

/* Quantitative targets */
.goal-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.goal {
  display: flex; gap: 7px; align-items: baseline;
  border: 1px solid var(--line); border-radius: 8px; padding: 6px 9px;
  font-size: 10.5px; line-height: 1.6; background: #fcfdfe;
}
.goal-k { flex: 0 0 46px; font-weight: 700; color: var(--ink); }
.goal-v { color: #4d5c6d; }

/* Warning signs */
.warn-box { border: 1px solid #f3d5d5; background: #fffafa; border-radius: 9px; padding: 8px 11px; }
.warn-row {
  display: flex; gap: 7px; align-items: flex-start;
  font-size: 10.5px; line-height: 1.7; color: #5a3f3f; padding: 3px 0;
}
.warn-row .ic { color: var(--red); flex: 0 0 auto; margin-top: 2px; }

/* Disclaimer */
.disc { border: 1px solid var(--line); border-radius: 9px; padding: 9px 12px; margin-top: 9px; background: #fafbfd; }
.disc.guide { border-color: #cfe0f0; background: #f7fbff; }
.disc-h {
  display: flex; align-items: center; gap: 6px;
  font-size: 11px; font-weight: 700; color: var(--ink); margin-bottom: 4px;
}
.disc-h .ic { color: var(--muted); }
.disc.guide .disc-h .ic { color: var(--blue); }
.disc-b { font-size: 10px; line-height: 1.8; color: #5b6b7f; }
.disc-b strong { color: var(--ink); }


/* ================= WeasyPrint compatibility layer ================= */
@page {
  size: A4;
  margin: 30px 40px 44px;
  @bottom-left {
    content: "Personal Multimorbidity Risk Interpretation and Health Management Report";
    font-size: 8.5px; color: #9aa8b8;
  }
  @bottom-right {
    content: "Page " counter(page) " of " counter(pages);
    font-size: 8.5px; color: #9aa8b8;
  }
}
html, body { margin: 0; padding: 0; background: #fff; }
body { width: auto; }
.page { min-height: 0; height: auto; padding: 0; margin: 0; box-shadow: none; page-break-after: auto; }
.cover { height: auto; page-break-after: auto; margin-bottom: 18px; }

/* grid → table / inline-block */
.metrics { display: table; width: 100%; table-layout: fixed; border-spacing: 5px 0; }
.metric { display: table-cell; }
.dash-sum { display: table; width: 100%; table-layout: fixed; border-spacing: 4px 0; }
.sum-item { display: table-cell; }
.lowgrid, .neg-grid { display: block; }
.lowcell, .neg { display: inline-block; vertical-align: top; width: 48%; margin: 0 2% 8px 0; }
.lowcell:nth-child(2n), .neg:nth-child(2n) { margin-right: 0; }

/* Two-column card table: auto-aligned row heights */
table.low-tbl, table.neg-tbl {
  width: 100%; border-collapse: separate; border-spacing: 0 7px; table-layout: fixed;
}
table.low-tbl td, table.neg-tbl td { vertical-align: top; padding: 0; width: 50%; }
table.low-tbl td:first-child, table.neg-tbl td:first-child { padding-right: 7px; }
table.low-tbl tr, table.neg-tbl tr { break-inside: avoid; page-break-inside: avoid; }
table.low-tbl .lowcell, table.neg-tbl .neg { display: block; width: auto; margin: 0; }
.lowcell-gauges { display: block; text-align: left; }
.lowcell-gauges .gcard {
  display: inline-block; vertical-align: top; width: 48%; margin: 0 2% 5px 0;
  border: 0; padding: 0; background: none;
}
.lowcell-gauges .gcard:nth-child(2n) { margin-right: 0; }
.lowcell-gauges .gauge { width: 100%; max-width: 150px; }
.mech-chain { display: block; }
.lowcell-more { font-size: 8.5px; color: #8a97a8; text-align: right; margin-top: 2px; }
.kv { display: table; width: 100%; }
.kv-k { display: table-cell; width: 1%; white-space: nowrap; padding-right: 8px; }
.kv-v { display: table-cell; }
/* Narrow two-column disease card: wrap labels so long English model names do not overflow */
.dz5-col .kv { display: block; }
.dz5-col .kv-k { display: block; width: auto; white-space: normal; padding-right: 0; }
.dz5-col .kv-v { display: block; word-break: break-word; overflow-wrap: anywhere; }
.dz5-col { overflow: hidden; }
.dz5-cols { display: table; width: 100%; table-layout: fixed; }
.dz5-col { display: table-cell; vertical-align: top; }
.dz5-col:first-child { width: 30%; padding-right: 12px; }
.dz5-col:last-child { width: 70%; }
.goal-grid { display: block; }
.goal { display: inline-table; vertical-align: top; width: 48.6%; margin: 0 1.4% 7px 0; }
.goal:nth-child(2n) { margin-right: 0; }
.goal-k { display: table-cell; width: 58px; padding-right: 6px; }
.goal-v { display: table-cell; }

/* flex → table */
.dz { break-inside: avoid; page-break-inside: avoid; }
.lowcell { break-inside: avoid; page-break-inside: avoid; }
.fcard, .mech, .stg, .act, .goal, .neg, .dz5 {
  break-inside: avoid; page-break-inside: avoid;
}
.dz-body { display: table; width: 100%; }
.dz-gauges { display: table-cell; vertical-align: top; }
.glegend { display: table-cell; vertical-align: top; width: 180px; }
.dz-body.solo .gcard { width: 49%; margin-right: 2%; }
.dz-body.solo .gcard:last-child { margin-right: 0; }
.dz-body.solo .gauge { width: 100%; }
.gl-row { display: block; }
.gl-row i { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 6px; }
.gcard { display: inline-block; vertical-align: top; width: 238px; margin: 0 7px 7px 0; }
.dz-body:not(.solo) .gcard { width: 300px; }
.dz-body:not(.solo) .gauge { width: 290px; }
.gauge { width: 230px; height: auto; }
.fcard { display: table; width: 100%; }
.fcard-top { display: table; width: 100%; }
.fcard-idx { display: table-cell; width: 20px; }
.fcard-name { display: table-cell; }
.fcard-val { display: table-cell; text-align: right; white-space: nowrap; }
.fcard-ic { display: table-cell; width: 46px; vertical-align: top; }
.fcard-main { display: table-cell; vertical-align: top; }
.pic { width: 34px; height: 34px; }
.uic { width: 15px; height: 15px; vertical-align: -2px; }
.ic { width: 15px; height: 15px; vertical-align: -2px; }

/* Additional subclasses */
.gcard-na { font-size: 9.5px; color: var(--muted); background: #f4f6f9; border-radius: 6px;
  padding: 10px 8px; text-align: center; margin-top: 4px; }
.gcard-note { font-size: 8.5px; color: var(--muted); margin-top: 2px; }
.dz.c-high { border-color: #f0c9c9; background: #fffafa; }
.dz.c-mid { border-color: #f0e0c0; background: #fffdf7; }
.dz.c-low { border-color: #d8e6d8; background: #fbfdfb; }
.dz.c-none { border-color: var(--line); background: #fbfcfd; }
.p { font-size: 10.5px; line-height: 1.85; color: #3d4a5c; margin: 0 0 6px; }
.ul, .ol { margin: 0 0 6px; padding-left: 16px; }
.ul li, .ol li { font-size: 10.5px; line-height: 1.8; color: #3d4a5c; margin-bottom: 2px; }
.tbl { width: 100%; border-collapse: collapse; margin: 4px 0 8px; }
.tbl th { background: #f2f5f9; font-size: 9.5px; color: #4a5a6e; text-align: left;
  padding: 5px 7px; border-bottom: 1px solid var(--line); font-weight: 700; }
.tbl td { font-size: 9.5px; color: #3d4a5c; padding: 5px 7px; border-bottom: 1px solid #eef1f5;
  line-height: 1.6; vertical-align: top; }
.sec { margin-bottom: 14px; }
.sec-h { display: table; width: 100%; margin: 0 0 8px; }
.sec-h .n { display: table-cell; width: 26px; }
.sec-h .t { display: table-cell; font-size: 14px; font-weight: 700; color: var(--ink); }
.sec-h .u { display: table-cell; width: 20px; text-align: right; }

.dz-ic { width: 16px; height: 16px; }
img.dz-ic { display: inline-block; vertical-align: -3px; margin-right: 6px; }
.uic { width: 15px; height: 15px; }
.ic { width: 14px; height: 14px; vertical-align: -2px; margin-right: 4px; }
.pic { width: 34px; height: 34px; }
.gcard-na { font-size: 9.5px; color: #8a97a8; background: #f4f6f9; border-radius: 6px;
  padding: 12px 8px; text-align: center; margin-top: 4px; }
.gcard-note { font-size: 8.5px; color: #8a97a8; margin-top: 2px; }
.chapter { margin-bottom: 16px; }
.act-group { font-size: 10px; font-weight: 700; color: #4a5a6e; margin: 10px 0 6px; padding-left: 2px; }
.sec { margin-bottom: 12px; }
.dz5-col-h { font-size: 9.5px; font-weight: 700; color: #6b7a8d; margin-bottom: 5px;
  letter-spacing: .04em; }
"""
