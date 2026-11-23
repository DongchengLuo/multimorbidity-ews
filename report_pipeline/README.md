# report_pipeline — Multimorbidity Risk Report Visualization Pipeline

Renders an "LLM-produced markdown report + batch scoring data" into a **single-file HTML + vector PDF**.
Layout is fully determined by code; **the rendering stage calls no LLM**, so the same input always yields the same output.

## Quick start

```bash
# Existing markdown: render directly (HTML + PDF + integrity check)
python -m report_pipeline --app-no 90000001 --pdf --check

# Run the scoring chain first (DB -> models -> batch), then render
python -m report_pipeline --app-no 90000001 --score --pdf --check

# Specify batch / markdown / output directory
python -m report_pipeline --app-no 90000001 \
    --batch batches/batch_200_xxx \
    --md debug_output/task_xxx/90000001/report_90000001.md \
    --out /tmp/out --pdf
```

Dependencies: `weasyprint` (PDF), `Pillow` (optional). **No browser required.**

## Modules

| File | Responsibility |
|---|---|
| `cohort.py` | Read batch data: patients, model results, `gauge_reference`, cohort distribution, lab items, demographics |
| `gauge.py` | Threshold parsing `parse_bands()`, marker placement `locate()`, color mapping, 180-degree half-ring SVG |
| `mdparse.py` | Markdown -> structured blocks (sections, tables, lists, paragraphs) |
| `components.py` | Inline icons, gauge card, disease card, fallback block rendering |
| `pages.py` | Cover, gauge page, body chapters (abnormal cards / negative cards / mechanism cards / hub table / strategy cards / action items / goal cards / disease detail) |
| `render.py` | Assemble HTML |
| `pdf.py` | HTML -> PDF (WeasyPrint) |
| `check.py` | Information integrity check (every number / sentence / table cell must be found) |
| `cli.py` | Command-line entry point |

## Gauge rules (important)

1. **Arc length is divided equally by "risk band"**; the marker maps linearly to the actual value within its band.
   Tick labels show the model's original thresholds so readers can verify the marker position.
2. **Threshold source priority**
   - `model`: the model ships its own `gauge_reference` (e.g. `0%-5%|5%-10%|10%-100%`)
   - `cohort`: when no threshold exists, cut points are inferred from the **same-batch cohort distribution** (n>=12); the report notes
     "thresholds divided by same-batch cohort percentiles (n=...)"
   - `level`: when the model only gives a level and no continuous value, the scale is "Low/Medium/High"
3. **Disease tier = the highest risk among the disease's models**, and **only Medium/High are shown**:
   - Any Medium or High -> hide low-risk models, disease is Medium/High
   - Only Low -> show all, disease is Low
   - This prevents the same disease from showing both "high risk" and "low risk" conclusions
4. **Continuous models** (`continuous risk` / no level) infer a level from cohort percentiles:
   above P80 -> High, P50-P80 -> Medium, otherwise -> Low; the card notes "includes cohort-percentile inference".
5. **Categorical levels** are auto-detected: `C-TR3`/`TI-RADS`/`BI-RADS`/`Lung-RADS`/`LI-RADS`
   -> 1-3 Low, 4 Medium, 5+ High.
6. **Stacking scores are not shown on the gauge** (imputation ratio, training-set basis, not comparable across diseases).

## Body rendering contract

Every specialized renderer goes through `walk(body, handler)`: it iterates blocks in `blocks()` order,
and the handler only takes over blocks it recognizes (e.g. `table` -> hub table, `ul` -> negative cards);
**all other blocks fall back to `render_blocks()`**. This way, if the LLM changes its phrasing (table to list,
list to paragraph), no content is lost.

| Section | Renderer | Trigger |
|---|---|---|
| Key abnormal findings list | `findings_cards()` | Title contains "abnormal findings" |
| Tested-normal items | `negatives()` | Title contains "tested normal / normal items" |
| Cross-specialty risk integration | `mech_cards()` | Chapter title contains "cross-specialty" |
| Shared risk factors | `hub_table()` | Chapter title contains "shared risk factors / risk hub" |
| 4.1 / 4.2 / 4.3 | `strategy_cards()` | Title contains "shared strategy / specific strategy / coordination" |
| 6.1 Action list | `action_items()` | Title contains "action list" |
| 6.2 Quantified goals | `goal_cards()` | Title contains "quantified goals" |
| 5.x Disease detail | `disease_detail()` | Title matches a disease name |
| Others | `render_block_content()` | Fallback |

Paragraph-level callouts are recognized by `components.callout_cls()`: `**Note**` / `Note:` -> blue,
`**Needs priority explanation...**` -> red, `**Significance of these negative results**` -> green.

## Icon selection

Abnormal-item icons use the `pages.FINDING_ICON` keyword table (order-sensitive; ratio items come before ALT/AST);
if nothing matches, a generic warning icon is used. Add one line here when adding a new lab item.

## Information integrity

`check.py` is a hard guarantee: every number, every sentence and every table cell in the markdown must be found in the HTML.
Any specialized renderer (abnormal cards, action items, etc.) must output non-table text verbatim,
otherwise the check fails. **No passing check, no delivery.**

## Known limitations

- 6 models are dropped by `format_results()` because their `code` is empty (data-layer issue, see P0).
- Cohort-percentile inference depends on same-batch sample size (n>=12); single-case batches degrade to no scale.
- Low-risk diseases draw the half-ring of every visible model under that disease (2 per row);
  diseases with many models (e.g. VTE with 4 items) make the grid row taller, leaving the other cell in the row blank.
