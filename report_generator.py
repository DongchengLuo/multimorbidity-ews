# -*- coding: utf-8 -*-
"""
Multimorbidity primary-prevention report generation (two-step)
================================================================================
Step 1: generate a plain-text Markdown report from prompt_multimorbidity_rag.txt
        [Key subject information] = full record (no denoising; each tested item labeled normal/abnormal)
                            + basic information + all computable model scores and tiers
                            + stacking (only when the imputation ratio <= 1/3)
        [Knowledge corpus]      = individualized knowledge background after LLM selection (selection, not compression)

Step 2: convert the Markdown report into a json-render spec (flat element tree),
        Keep all sections and key information of the original report; change only the presentation.

Design trade-off (why two steps):
  - The step-1 prompt is clinical-reasoning oriented; also demanding a constrained UI structure would harm both
    medical reasoning quality and structural validity;
  - The two steps can be rerun/debugged independently: the report can be reviewed manually and the spec validated separately;
  - Step 2 only performs a format conversion, with structural validation and key-information checks to avoid loss.

Pipeline usage:
    import report_generator
    records = report_generator.generate_reports(records, batch_results, progress_cb=...)
================================================================================
"""
import json
import os
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path

import requests

import jsonrender_catalog as catalog
from llm_config import LLM_API_URL as _CFG_URL, LLM_MODEL as _CFG_MODEL, require_llm_config

PROJECT_ROOT = Path(__file__).resolve().parent
PROMPT_FILE = PROJECT_ROOT / "prompt_multimorbidity_rag.txt"

# The report template is English-only; no language switch is needed.

# ===================== LLM configuration (same interface as llm_*.py) =====================
# Read uniformly from llm_config, which loads .env at import time to avoid import-order issues
# fall back to a wrong default model name.
LLM_URL = _CFG_URL
LLM_MODEL = _CFG_MODEL


def _env_int(name, default, minimum=1):
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_float(name, default, minimum=0.0):
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


# Read timeout per request (seconds).
# Note: per-request throughput drops under high concurrency (about 20 min per request at 16-way concurrency),
# Estimate the timeout from prefill + max output, otherwise retries will time out repeatedly.
LLM_TIMEOUT = _env_float("REPORT_LLM_TIMEOUT", 2400, 1)
# Connect timeout (seconds): fail fast when the server is unreachable to avoid hung threads
LLM_CONNECT_TIMEOUT = _env_float("REPORT_LLM_CONNECT_TIMEOUT", 10, 1)
# Retries per request (including the first attempt)
LLM_RETRIES = _env_int("REPORT_LLM_RETRIES", 2, 1)

# ===== Streaming receive (key: detect disconnects/hangs quickly instead of waiting for LLM_TIMEOUT) =====
# 1 = streaming (recommended); 0 = fall back to legacy non-streaming full-payload receive
REPORT_LLM_STREAM = os.getenv("REPORT_LLM_STREAM", "1") != "0"
# Time to wait for the first byte (seconds): when client concurrency exceeds the server's max_num_seqs, requests
# Server-side queuing; allow enough time or normal queuing is misread as a disconnect.
REPORT_LLM_QUEUE_TIMEOUT = _env_float("REPORT_LLM_QUEUE_TIMEOUT", 1800, 10)
# Mid-stream stall timeout (seconds): max gap between chunks after the first token.
# Exceeding it means the connection is dead; raise and retry. The request is already generating, so a short timeout detects disconnects quickly.
REPORT_LLM_STALL_TIMEOUT = _env_float("REPORT_LLM_STALL_TIMEOUT", 120, 10)

REPORT_MAX_TOKENS = _env_int("REPORT_MAX_TOKENS", 8192, 256)
JSONRENDER_MAX_TOKENS = _env_int("JSONRENDER_MAX_TOKENS", 16384, 256)
REPORT_ENABLED = os.getenv("REPORT_ENABLED", "1") != "0"
# Knowledge distillation: convert the subgraph into individualized knowledge points (auto-denoised), then write the report
KNOWLEDGE_STEP = os.getenv("REPORT_KNOWLEDGE_STEP", "1") != "0"
# Output cap for knowledge selection: selection is not compression; allow enough length or key thresholds/contraindications get summarized away
KNOWLEDGE_MAX_TOKENS = int(os.getenv("REPORT_KNOWLEDGE_MAX_TOKENS", "16384"))
# When the candidate pool is too large, chunk by the "### Evidence" boundary so truncation does not silently drop later evidence
KNOWLEDGE_CHUNK_CHARS = _env_int("REPORT_KNOWLEDGE_CHUNK_CHARS", 120000, 1000)
KNOWLEDGE_CHUNK_WORKERS = _env_int("REPORT_KNOWLEDGE_CHUNK_WORKERS", 3, 1)

# ===================== Concurrency configuration (downstream report requests) =====================
# Concurrency model: global task queue + fixed worker thread pool.
# The concurrency cap limits in-flight LLM tasks, not the number of patients:
#   - Each patient's prepare / knowledge chunking / report / structured conversion is an independent task;
#   - On completion, immediately requeue the next-stage task so any idle worker can pick it up,
#     No situation where one patient waits while other workers idle;
#   - Patients no longer block each other; a slow request occupies one worker while others proceed.
REPORT_LLM_MAX_INFLIGHT = _env_int("REPORT_LLM_MAX_INFLIGHT", 128, 1)
# Backward compatibility: REPORT_LLM_MAX_CONCURRENCY is kept only as an alias of INFLIGHT
# (If INFLIGHT is set explicitly it takes precedence; new code should use INFLIGHT directly.)
REPORT_LLM_MAX_CONCURRENCY = _env_int(
    "REPORT_LLM_MAX_CONCURRENCY", REPORT_LLM_MAX_INFLIGHT, 1
)
# Max wait for a concurrency slot (seconds): must exceed a single request's duration or long requests are killed.
REPORT_LLM_ACQUIRE_TIMEOUT = _env_float("REPORT_LLM_ACQUIRE_TIMEOUT", 3600, 1)
# Overall wait limit for the concurrent stage (seconds): unfinished patients are marked TIMEOUT and the pipeline continues.
# Allow ample time; default 3 hours.
REPORT_LLM_WAIT_TIMEOUT = _env_float("REPORT_LLM_WAIT_TIMEOUT", 10800, 1)
# Retries per task (including the first attempt) for network jitter / sporadic 5xx.
REPORT_TASK_RETRIES = _env_int("REPORT_TASK_RETRIES", 2, 1)
# Fallback after a knowledge-chunking task fails permanently: use the raw evidence block (truncated) as that knowledge,
# Ensure the patient still gets a report instead of failing the whole report.
REPORT_KNOWLEDGE_FALLBACK_CHARS = _env_int(
    "REPORT_KNOWLEDGE_FALLBACK_CHARS", 120000, 1000
)


class _SlotPool:
    """FIFO-fair concurrency slot pool.

    threading.Semaphore is not fair: when a single request takes far longer than the queue interval,
    early threads may repeatedly fail to get a slot and starve (manifesting as acquire timeouts).
    Use queue.Queue as a token pool: first come, first served, no starvation.
    """

    def __init__(self, size):
        self.size = max(1, int(size))
        self._q = queue.Queue()
        for _ in range(self.size):
            self._q.put(True)

    def acquire(self, timeout=None):
        try:
            self._q.get(timeout=timeout)
            return True
        except queue.Empty:
            return False

    def release(self):
        self._q.put(True)


# Global in-flight request gate; every _llm_chat call must acquire a slot first.
_LLM_SLOTS = _SlotPool(REPORT_LLM_MAX_INFLIGHT)
# Thread-local: record the last request's finish_reason / usage to detect truncated responses
_TLS = threading.local()

# stacking imputation-ratio cap: excluded from the report when exceeded
STACKING_MAX_IMPUTE_RATIO = float(os.getenv("STACKING_MAX_IMPUTE_RATIO", "0.3333"))

TIER_LABEL = {0: "No valid tier", 1: "Low risk", 2: "Medium risk", 3: "High risk", 4: "Very high risk"}
TIER_COLOR = {0: "green", 1: "green", 2: "orange", 3: "red", 4: "red"}


# ===================== LLM calls =====================
def last_finish_reason():
    """finish_reason of the current thread's latest LLM request (length means the response was truncated)."""
    return getattr(_TLS, "finish_reason", None)


def _blocking_chat(payload):
    """Non-streaming full-payload receive (compatibility mode)."""
    require_llm_config()
    resp = requests.post(
        LLM_URL, json=payload,
        headers={"Content-Type": "application/json"},
        timeout=(LLM_CONNECT_TIMEOUT, LLM_TIMEOUT),
    )
    try:
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        return content, choice.get("finish_reason"), data.get("usage")
    finally:
        resp.close()


def _set_read_timeout(resp, seconds):
    """Set the connected socket's read timeout to seconds (urllib3 internals; keep the original on failure)."""
    try:
        sock = resp.raw._fp.fp.raw._sock
        sock.settimeout(seconds)
        return True
    except Exception:
        return False


def _stream_chat(payload, started_at):
    """Streaming receive (SSE).

    Why streaming is required: a non-streaming response only returns after full generation, so on server restart/disconnect the client
    would otherwise wait for LLM_TIMEOUT (tens of minutes); with streaming, tokens keep arriving, so as long as
    If no data arrives within REPORT_LLM_STALL_TIMEOUT, treat the connection as dead and retry immediately.

    Two-stage timeout:
      - Before the first chunk: wait per REPORT_LLM_QUEUE_TIMEOUT (server-side queuing is normal);
      - After the first chunk: switch to REPORT_LLM_STALL_TIMEOUT to detect disconnects quickly.
    """
    require_llm_config()
    body = dict(payload)
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    resp = requests.post(
        LLM_URL, json=body,
        headers={"Content-Type": "application/json"},
        stream=True,
        timeout=(LLM_CONNECT_TIMEOUT, REPORT_LLM_QUEUE_TIMEOUT),
    )
    try:
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        parts = []
        finish = None
        usage = None
        first_chunk_seen = False
        for raw in resp.iter_lines(decode_unicode=False):
            if raw and not first_chunk_seen:
                first_chunk_seen = True
                _set_read_timeout(resp, REPORT_LLM_STALL_TIMEOUT)
            if time.time() - started_at > LLM_TIMEOUT:
                raise RuntimeError(
                    f"LLM total time exceeded {LLM_TIMEOUT:.0f}s (still streaming; forced abort)"
                )
            if not raw:
                continue
            line = raw.decode("utf-8", "ignore").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if choices:
                ch = choices[0]
                delta = ch.get("delta") or {}
                piece = delta.get("content")
                if piece:
                    parts.append(piece)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
        return "".join(parts), finish, usage
    finally:
        resp.close()


def _llm_chat(messages, max_tokens, temperature=0.2, json_mode=False):
    """Single LLM call (concurrency-safe).

    Concurrency and fault-tolerance notes:
      1. Each request opens its own connection (requests.post does not reuse a Session), so responses cannot interleave;
      2. Acquire a global in-flight slot first; error out after REPORT_LLM_ACQUIRE_TIMEOUT if none is available,
         No situation where threads wait forever for a slot and the pipeline stalls;
      3. Streaming by default: on server restart/disconnect, no data within REPORT_LLM_STALL_TIMEOUT
         retry as soon as new data arrives instead of waiting for LLM_TIMEOUT;
      4. Separate connect/read timeouts; slots are released in finally so exception paths do not leak.
    """
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    last_err = None
    for attempt in range(LLM_RETRIES):
        if not _LLM_SLOTS.acquire(timeout=REPORT_LLM_ACQUIRE_TIMEOUT):
            raise RuntimeError(
                f"LLM concurrency-slot wait timed out (>{REPORT_LLM_ACQUIRE_TIMEOUT:.0f}s, "
                f"cap {REPORT_LLM_MAX_INFLIGHT})"
            )
        t0 = time.time()
        try:
            if REPORT_LLM_STREAM:
                content, finish, usage = _stream_chat(payload, t0)
            else:
                content, finish, usage = _blocking_chat(payload)
            _TLS.finish_reason = finish
            _TLS.usage = usage
            if content.strip():
                u = usage or {}
                print(
                    f"[report] LLM finished in {time.time()-t0:.0f}s "
                    f"prompt={u.get('prompt_tokens')} "
                    f"completion={u.get('completion_tokens')} "
                    f"finish={finish}"
                )
                if finish == "length":
                    print(
                        f"[report] WARNING: LLM response truncated by max_tokens={max_tokens}"
                        f"（finish_reason=length）"
                    )
                return content
            last_err = "model returned empty content"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            print(
                f"[report] LLM request failed attempt={attempt + 1}/{LLM_RETRIES} "
                f"elapsed={time.time()-t0:.0f}s: {last_err}"
            )
        finally:
            _LLM_SLOTS.release()
        if attempt < LLM_RETRIES - 1:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"LLM call failed: {last_err}")


# ===================== Step 1: key subject information =====================
def _stacking_ratio(stacking_results):
    """stacking imputation ratio = imputed / (inputs + imputed)."""
    total = 0
    imputed = 0
    for s in stacking_results or []:
        inp = s.get("input_scores") or {}
        imp = s.get("imputed_scores") or {}
        total += len(inp) + len(imp)
        imputed += len(imp)
    return (imputed / total) if total else 1.0


# ===================== Abnormal-finding pre-scan (deterministic, no LLM) =====================
ABNORMAL_DESC_KEYWORDS = [
    "hypoechoic", "hyperechoic", "strong echo", "increased echogenicity", "heterogeneous echo", "uneven", "streaky", "patchy", "spotty",
    "nodule", "mass", "cyst", "stone", "plaque", "thickening", "dilatation", "calcification", "reflux", "effusion",
    "stenosis", "enlargement", "increased size", "atrophy", "absent", "ectopic", "space-occupying", "polyp",
    "erosion", "ulcer", "hyperemia", "exudate", "turbid", "cloudy", "irregular", "not smooth", "coarse", "widened", "poorly defined",
    "hemorrhoid", "turbinate hypertrophy", "valvular regurgitation", "intimal thickening",
]

# Absent/not-visualized organs are often misread as normal due to "not seen"; detect them separately.
# Example: "no kidney in the right renal fossa" lacks "no obvious kidney", so the old keyword table missed it.
ORGAN_ABSENCE_RE = re.compile(
    r"(?:not seen|not visualized|not detected|absent|missing|resected)\s*(?:obvious)?\s*"
    r"(?:kidney|right kidney|left kidney|gallbladder|spleen|thyroid|uterus|ovary|prostate|testis)"
    r"(?!abnormal|lesion|mass|enlarged|stone|hydro|echo)"
)
NORMAL_DESC_KEYWORDS = [
    "no obvious abnormality", "no abnormality", "no obvious mass", "no dilatation", "no enlargement", "no obvious mass lesion",
    "normal size and shape", "homogeneous echo", "well-defined contour", "intact capsule",
]


def _parse_range(exp):
    """Parse reference ranges: '82.0-100.0' / '3.90-6.10' / '<5.2' / '>1.0'."""
    text = str(exp or "").strip().replace("～", "-").replace("~", "-")
    m = re.match(r"^([0-9.]+)\s*-\s*([0-9.]+)$", text)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.match(r"^[<≤]\s*([0-9.]+)$", text)
    if m:
        return None, float(m.group(1))
    m = re.match(r"^[>≥]\s*([0-9.]+)$", text)
    if m:
        return float(m.group(1)), None
    return None, None


def extract_abnormal_findings(record):
    """Programmatically extract all abnormal findings (out-of-range labs + abnormal descriptive sentences)."""
    findings = []

    for it in (record.get("lab_items") or []):
        name = it.get("fieldName") or it.get("res_itm_name") or ""
        val = str(it.get("res_value") or "").strip()
        exp = str(it.get("res_exp") or "").strip()
        unit = str(it.get("res_unit") or "").strip()
        if not val:
            continue
        try:
            v = float(val)
        except ValueError:
            continue

        # Prefer res_refl/res_refu (numeric reference range), then parse res_exp
        lo = hi = None
        refl = str(it.get("res_refl") or "").strip()
        refu = str(it.get("res_refu") or "").strip()
        try:
            if refl:
                lo = float(refl)
            if refu:
                hi = float(refu)
        except ValueError:
            lo = hi = None
        if lo is None and hi is None:
            lo, hi = _parse_range(exp)
        if lo is None and hi is None:
            continue

        flag = None
        if lo is not None and v < lo:
            flag = "down, below reference"
        elif hi is not None and v > hi:
            flag = "up, above reference"
        if flag:
            ref_txt = exp or (f"{refl}-{refu}" if refl and refu else (f"<{refu}" if refu else f">{refl}"))
            findings.append({
                "source": "lab",
                "dept": it.get("det_dep_name") or "",
                "text": f"{name} {val} {unit} (ref {ref_txt}) {flag}",
            })

    for d in (record.get("desc_items") or []):
        dept = d.get("det_com_name") or ""
        text = d.get("res_result") or ""
        # Split by clause and judge each clause (so "left kidney normal" does not skew "right kidney not seen")
        # Note: do not split on the enumeration comma; "prominent and widened aortic knob" is one description,
        # Splitting leaves subject-less fragments such as "widened".
        for clause in re.split(r"[。；;，,\n]", text):
            s = clause.strip()
            if len(s) < 4:
                continue
            if _is_abnormal_clause(s):
                findings.append({"source": "imaging/ultrasound", "dept": dept, "text": s})

    # Deduplicate (keep first occurrence)
    seen = set()
    unique = []
    for f in findings:
        key = (f["dept"], f["text"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(f)
    return unique


def _is_abnormal_clause(clause):
    """Check whether a single clause is an abnormal description (handling negations such as "no enlargement / not enlarged")."""
    # Absent/not-visualized organs take priority: they are abnormal and must not be negated by "not seen"
    if ORGAN_ABSENCE_RE.search(clause):
        return True
    for kw in ABNORMAL_DESC_KEYWORDS:
        idx = clause.find(kw)
        if idx < 0:
            continue
        prefix = clause[max(0, idx - 8):idx]
        # Negations: none / not seen / not / without / no, optionally with "obvious/abnormal/marked" modifiers
        negated = bool(re.search(r"(no|not|without|absent)( obvious| abnormal| marked)?$", prefix, re.I))
        if negated:
            # Exception: absent/ectopic organs are themselves abnormal
            if kw in ("absent", "ectopic"):
                return True
            continue
        return True
    return False


def _lab_value(record, keywords):
    """Take the first parseable number by name keyword."""
    for it in (record.get("lab_items") or []):
        name = it.get("fieldName") or it.get("res_itm_name") or ""
        if any(k in name for k in keywords):
            try:
                return float(str(it.get("res_value") or "").strip())
            except (TypeError, ValueError):
                continue
    return None


def _lab_text(record, keywords):
    for it in (record.get("lab_items") or []):
        name = it.get("fieldName") or it.get("res_itm_name") or ""
        if any(k in name for k in keywords):
            return str(it.get("res_value") or "").strip()
    return ""


def build_clinical_interpretation(record):
    """Interpret key indicators by guideline standards (preferred over lab reference ranges)."""
    lines = []

    hba1c = _lab_value(record, ["HbA1c", "glycated hemoglobin", "GHb"])
    if hba1c is not None:
        if hba1c >= 6.5:
            txt = ("Reaches the diabetes diagnostic cut-off (>=6.5%); confirm on repeat (another HbA1c or fasting glucose/OGTT); "
                   "diagnosis if typical symptoms or two abnormal results")
        elif hba1c >= 5.7:
            txt = "Prediabetes (ADA 5.7-6.4%; lab reference ranges may be wider)"
        else:
            txt = "Normal"
        lines.append(f"- HbA1c {hba1c}%: {txt}")

    fpg = _lab_value(record, ["GLU", "fasting glucose", "glucose"])
    if fpg is not None:
        if fpg >= 7.0:
            txt = ("Reaches the diabetes diagnostic cut-off (>=7.0 mmol/L); confirm on repeat; "
                   "diagnosis if typical symptoms or two abnormal results")
        elif fpg >= 5.6:
            txt = "Impaired fasting glucose (5.6-6.9 mmol/L)"
        else:
            txt = "Normal"
        lines.append(f"- Fasting glucose {fpg} mmol/L: {txt}")

    height = _lab_value(record, ["height"])
    weight = _lab_value(record, ["weight"])
    if height and weight and height > 50:
        bmi = weight / ((height / 100.0) ** 2)
        if bmi < 18.5:
            txt = "Underweight"
        elif bmi < 24:
            txt = "Normal"
        elif bmi < 28:
            txt = "Overweight"
        else:
            txt = "Obese"
        lines.append(f"- BMI {bmi:.1f} kg/m2 ({weight} kg / {height} cm): {txt} (Chinese criteria)")

    sbp = dbp = None
    bp = _lab_text(record, ["blood pressure"])
    m = re.match(r"^\s*(\d{2,3})\s*/\s*(\d{2,3})", bp or "")
    if m:
        sbp, dbp = int(m.group(1)), int(m.group(2))
        if sbp >= 140 or dbp >= 90:
            txt = "Hypertension"
        elif sbp >= 120 or dbp >= 80:
            txt = "High-normal"
        else:
            txt = "Normal"
        lines.append(f"- Blood pressure {sbp}/{dbp} mmHg: {txt}")

    # LDL-C: state the measured value and lab reference range only; do not risk-stratify or set targets
    ldl = _lab_value(record, ["LDL", "low-density lipoprotein"])
    if ldl is not None:
        if ldl >= 3.4:
            lines.append(
                f"- LDL-C {ldl} mmol/L: above the lab upper limit (<3.4 mmol/L); "
                "the specific target must be set by the physician based on overall cardiovascular risk"
            )
        else:
            lines.append(f"- LDL-C {ldl} mmol/L: within the lab reference range (<3.4 mmol/L)")

    # Urine albumin: use UACR; a single concentration is affected by specific gravity
    uacr = _lab_value(record, ["UACR", "urine albumin/creatinine", "albumin/creatinine"])
    alb = _lab_value(record, ["urine albumin", "microalbumin"])
    if uacr is not None:
        if uacr < 3.0:
            lines.append(f"- UACR {uacr} mg/mmol: normal (<3.0); a single urine albumin concentration is affected by specific gravity, so UACR is preferred")
        else:
            lines.append(f"- UACR {uacr} mg/mmol: elevated (>=3.0), suggesting albuminuria; confirm on repeat")
    elif alb is not None:
        lines.append(f"- Urine albumin {alb} mg/L: a single concentration is affected by specific gravity; interpret with UACR and do not judge renal injury from it alone")

    # Transaminases: do not interpret the AST/ALT ratio alone when absolute values are normal
    alt = _lab_value(record, ["ALT"])
    ast = _lab_value(record, ["AST"])
    if alt is not None and ast is not None:
        if alt <= 50 and ast <= 40:
            lines.append(f"- ALT {alt} / AST {ast} U/L: both within reference; do not interpret the AST/ALT ratio alone")

    # Homocysteine: guidelines flag 10 umol/L while lab reference ranges are often wider
    hcy = _lab_value(record, ["homocysteine", "HCY"])
    if hcy is not None and hcy >= 10:
        lines.append(f"- Homocysteine {hcy} umol/L: >=10 umol/L is the guideline attention threshold (lab ranges are often wider); assess folate/B12")

    hp = _lab_text(record, ["helicobacter pylori", "H. pylori", "Hp"])
    if hp:
        if "positive" in hp.lower() or "+" in hp:
            lines.append(f"- Helicobacter pylori: {hp} (infection; a definite risk factor for gastric cancer and peptic ulcer)")
        else:
            lines.append(f"- Helicobacter pylori: {hp}")

    return lines


def build_key_info(archive_text, diseases, assessments):
    """Assemble [Key subject information]: full record + model scores and tiers + conditional stacking."""
    lines = [archive_text.strip(), "", "[Model scores and risk tiers]"]

    # Disease -> assessment (model scores and stacking)
    assess_map = {a.get("disease_name"): a for a in (assessments or [])}

    ordered = sorted(diseases or [], key=lambda d: -(d.get("risk_tier") or 0))
    if not ordered:
        lines.append("(No computable disease risk score was produced this time)")

    for d in ordered:
        name = d.get("disease_name") or ""
        tier = d.get("risk_tier") or 0
        tier_label = TIER_LABEL.get(tier, "Unknown")
        # Output only the unified tier, not a single model's absolute risk percentage:
        # Tiers come from multi-model voting while percentages come from a single model; the two are not on the same scale;
        # Emitting both would make the generator write contradictions such as "0.31% but high risk".
        lines.append(
            f"- {name} ({d.get('english_name') or 'no English name'}): "
            f"unified risk tier={tier_label}; "
            f"{d.get('stacking_models') or 0} models involved"
        )

        a = assess_map.get(name) or {}
        for m in (a.get("model_results") or []):
            lines.append(
                f"    - model {m.get('model')}: score={m.get('score')} ({m.get('risk_level')})"
            )

        stacking = a.get("stacking_results") or []
        if stacking:
            ratio = _stacking_ratio(stacking)
            if ratio <= STACKING_MAX_IMPUTE_RATIO:
                for s in stacking:
                    pct = s.get("risk_5y_percent")
                    lines.append(f"    - stacking 5-year risk={pct}% (included)")
            # Emit no stacking information when the imputation ratio exceeds the limit:
            # Engineering details (imputation ratio / exclusions) must not enter the report context.

    return "\n".join(lines)


def build_key_info_with_uncomputable(archive_text, diseases, assessments, uncomputable,
                                     findings=None, interpretation=None):
    text = build_key_info(archive_text, diseases, assessments)
    if findings:
        head = [
            "[Abnormal findings that must each be addressed (system pre-scan; do not omit)]",
            "The items below were deterministically extracted from the raw data; each must appear in the report with interpretation and advice:",
        ]
        for i, f in enumerate(findings, 1):
            head.append(f"{i}. [{f['source']}/{f['dept']}] {f['text']}")
        head.append("")
        text = "\n".join(head) + "\n" + text
    if interpretation:
        head = [
            "[Key-indicator clinical interpretation (guideline standards, preferred over lab reference ranges)]",
            "The interpretations below were computed from guideline thresholds for reference; if they conflict with the raw values or imaging descriptions, the record prevails and the report should say so:",
        ]
        head.extend(interpretation)
        head.append("")
        text = "\n".join(head) + "\n" + text
    if uncomputable:
        lines = [text, "", "[Diseases not computable (insufficient information)]"]
        for u in uncomputable:
            if isinstance(u, dict):
                lines.append(f"- {u.get('disease_name') or u.get('disease') or ''}: {u.get('reason') or ''}")
            else:
                lines.append(f"- {u}")
        text = "\n".join(lines)
    return text


# ===================== Step 1: knowledge corpus =====================
def build_corpus(edges_json_path):
    """Render compressed subgraph edges as structured text (all edges, all retained evidence, full summaries)."""
    with open(edges_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    edges = data.get("edge_details") or []
    # If decision scoring ran, drop edges blocked by gating; keep all when unscored (normal flow).
    if any(e.get("decision_gate") for e in edges):
        kept = [e for e in edges if str(e.get("decision_gate") or "").upper() != "BLOCK"]
        edges = kept or edges
    risk_vector = data.get("patient_risk_vector") or {}

    lines = [
        f"[Subgraph overview] {data.get('node_count')} nodes, {len(edges)} edges; "
        f"patient risk vector: {json.dumps(risk_vector, ensure_ascii=False)}",
        "",
    ]

    for i, e in enumerate(edges, 1):
        src = e.get("source_term") or ""
        dst = e.get("target_term") or ""
        sg = e.get("subgraph_edge_data") or {}
        etype = sg.get("type") or ""
        meta = []
        if sg.get("weight") is not None:
            meta.append(f"subgraph weight={sg.get('weight')}")
        if sg.get("nav_weight") is not None:
            meta.append(f"navigation weight={sg.get('nav_weight')}")
        if sg.get("document_count") is not None:
            meta.append(f"total documents={sg.get('document_count')}")

        # Use "Evidence N" rather than "Edge N" so the generator does not write graph IDs into the report
        lines.append(f"### Evidence {i}: ({src}) -[{etype}]-> ({dst})")
        if meta:
            lines.append("Attributes: " + " | ".join(meta))

        docs = (e.get("full_kg_edge_data") or {}).get("document_sources") or {}
        if isinstance(docs, dict):
            doc_items = list(docs.items())
        elif isinstance(docs, list):
            doc_items = [(str(k), v) for k, v in enumerate(docs)]
        else:
            doc_items = []

        for j, (doc_name, info) in enumerate(doc_items, 1):
            info = info or {}
            detail = info.get("relation_detail") or {}
            title = info.get("document_title") or doc_name
            rel_type = detail.get("relation_type") or info.get("relation_type") or ""
            strength = detail.get("strength_qualitative") or ""
            summary = detail.get("text_summary") or info.get("text_summary") or ""
            head = f"  Evidence {j} | document: {title}"
            if rel_type:
                head += f" | relation type: {rel_type}"
            if strength:
                head += f" | evidence strength: {strength}"
            lines.append(head)
            if summary:
                lines.append(f"    summary: {summary}")
        lines.append("")

    return "\n".join(lines)


# ===================== Knowledge selection (candidate pool -> individualized knowledge) =====================
# Note: this step is selection, not compression. The KG only supplies background for the model to judge,
# Clinical reasoning is done by the generating model, so selection must preserve thresholds, contraindications and screening intervals from the source,
# Do not condense into vague conclusions just to save space.
KNOWLEDGE_PROMPT = """You are a clinical knowledge-selection assistant. Below are one subject's key information and a knowledge subgraph retrieved via the risk vector (candidate evidence pool, possibly noisy).__PART_NOTE__

Your task is SELECTION, not compression: pick the content useful for writing this subject's health-management report, remove irrelevant or contradictory noise, but **do not summarise away useful content**.

# Selection principles

1. **Keep only evidence consistent with this subject's clinical features.** Discard evidence that contradicts or is irrelevant to the individual, e.g.:
   - no asthma history or symptoms -> discard asthma evidence;
   - BMI normal or low -> discard "obesity causes ..." evidence;
   - uric acid normal -> discard hyperuricemia/gout evidence;
   - male subject -> discard pregnancy-related evidence.
2. **Preserve key source information; do not over-generalise.** For every piece of evidence judged relevant, keep:
   - specific values, thresholds and cut points (e.g. "HbA1c >9.0%", "LDL-C <1.8 mmol/L", "FPG >11.1 mmol/L");
   - drug names and conditions for use/avoidance (e.g. "use SGLT2i cautiously when ketoacidosis risk is present");
   - screening intervals and starting age;
   - evidence strength (strong/moderate/weak) and population restrictions.
   Prefer longer over lossy. **Do not merge, abbreviate or rewrite into vague conclusions to save space.**
3. **Organise by disease / risk hub**, listing the selected evidence points under each theme. Keep the source wording where possible; light editing for readability is fine, but meaning and numbers must not change.
4. **Do not fabricate** associations absent from the corpus. For themes the corpus does not cover, write "the candidate evidence pool does not cover this theme"; do not fill in with general knowledge.
5. **No source labels, knowledge-graph IDs or internal terminology** (e.g. "edge 71", "corpus relation", "decision score", "subgraph weight"). Output the knowledge content only.
6. Output Markdown. No hard length cap; aim for "complete after selection"; typically 3000-12000 words.

# Key subject information
__KEY_INFO__

# Candidate evidence pool (knowledge subgraph)
__CORPUS__
"""


def _split_corpus(corpus, limit):
    """Split the candidate pool by the "### Evidence" boundary into chunks of at most limit characters."""
    if len(corpus) <= limit:
        return [corpus]
    blocks = re.split(r"(?=^### Evidence )", corpus, flags=re.M)
    chunks, cur = [], ""
    for b in blocks:
        if cur and len(cur) + len(b) > limit:
            chunks.append(cur)
            cur = b
        else:
            cur += b
    if cur:
        chunks.append(cur)
    return chunks


def _filter_one(key_info, corpus, part, total):
    note = ""
    if total > 1:
        note = (f"\n\n[Note] The candidate pool is long and was split into {total} parts; this is part {part}."
                f"Output only this part's selection; do not infer or repeat other parts;"
                f"**Do not mention chunking, part numbers or this instruction in the output**.")
    prompt = (KNOWLEDGE_PROMPT
              .replace("__PART_NOTE__", note)
              .replace("__KEY_INFO__", key_info)
              .replace("__CORPUS__", corpus))
    return _llm_chat(
        [{"role": "user", "content": prompt}],
        max_tokens=KNOWLEDGE_MAX_TOKENS,
        temperature=0.1,
    )


def filter_knowledge(key_info, corpus, max_workers=None):
    """Use the LLM to select the knowledge this individual report needs from the candidate pool (selection, not compression).

    When the candidate pool exceeds KNOWLEDGE_CHUNK_CHARS, chunk by evidence boundary and select in parallel,
    Avoid over-long single contexts truncating and dropping later evidence.

    max_workers: intra-patient chunk parallelism; callers lower it when patient-level concurrency is high,
                 Avoid "patients x chunks" far exceeding the global in-flight cap and starving threads.
    """
    chunks = _split_corpus(corpus, KNOWLEDGE_CHUNK_CHARS)
    if len(chunks) == 1:
        return _filter_one(key_info, corpus, 1, 1)

    workers = max(1, min(max_workers or KNOWLEDGE_CHUNK_WORKERS, len(chunks)))
    results = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_filter_one, key_info, c, i + 1, len(chunks)): i
            for i, c in enumerate(chunks)
        }
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    return "\n\n".join(r for r in results if r)


# Backward-compatible alias (external scripts may still reference distill_knowledge)
distill_knowledge = filter_knowledge


# ===================== Output fallback cleaning =====================
# The prompt no longer requires source citations; this only removes the wording (no source detection),
# Prevent the model from occasionally leaving internal terms or graph IDs.
_INTERNAL_TAG_RE = re.compile(
    r"[(]\s*(?:corpus relation|general knowledge supplement|corpus relation[:][^)]*|edge\s*\d+[^)]*)\s*[)]"
)
_EDGE_ID_RE = re.compile(r"(?:see|refer to)?\s*edge\s*\d+\s*", re.I)


def _sanitize_output(text):
    """Strip residual internal labels and graph IDs."""
    if not text:
        return text
    text = _INTERNAL_TAG_RE.sub("", text)
    text = _EDGE_ID_RE.sub("", text)
    return text


# ===================== Negative-consistency warnings (warn only, never rewrite) =====================
# Warn when the record is clearly negative but the report asserts presence, for batch QA sampling.
# Match only presence-asserting phrases to avoid false positives such as "prevent plaque formation".
_NEGATIVE_CONFLICTS = [
    ("plaque", ("no plaque", "no obvious plaque", "without plaque"),
     ("stable plaque", "plaque rupture", "plaque burden", "plaque present", "carotid plaque", "established plaque")),
    ("stenosis", ("no stenosis", "no obvious stenosis", "no luminal stenosis"),
     ("luminal stenosis", "stenosis present", "degree of stenosis", "stenosis rate")),
    ("mass", ("no mass", "no obvious mass"),
     ("mass present", "space-occupying lesion")),
]


def check_negative_consistency(md, record):
    """Check whether the report contradicts negative conclusions in the record; returns warnings."""
    desc = " ".join(str(d.get("res_result") or "") for d in (record.get("desc_items") or []))
    lab = " ".join(
        f"{it.get('fieldName')}: {it.get('res_value')}" for it in (record.get("lab_items") or [])
    )
    corpus = desc + " " + lab
    warns = []
    for term, negs, positives in _NEGATIVE_CONFLICTS:
        if not any(n in corpus for n in negs):
            continue
        hit = [p for p in positives if p in md]
        if hit:
            warns.append(f"Report asserts '{'/'.join(hit)}' but the record is explicitly negative ({negs[0]})")
    return warns


# ===================== Step 1: generate the Markdown report =====================
def generate_markdown(key_info, corpus):
    template = PROMPT_FILE.read_text(encoding="utf-8")
    prompt = template.replace(
        "(paste the full examination results + all risk-score records and tier labels here)", key_info
    ).replace(
        "(paste the knowledge for the corresponding strategies: the subgraph edges content here)", corpus
    )
    return _llm_chat(
        [{"role": "user", "content": prompt}],
        max_tokens=REPORT_MAX_TOKENS,
        temperature=0.2,
    )


# ===================== Step 2: convert to json-render spec =====================
JSONRENDER_PROMPT = """You are a "report structure converter". Convert the following "Personal Multimorbidity Risk Interpretation and Health Management Report" into a json-render UI spec.

# Conversion target
- Change only the presentation; **do not lose any key information**: all sections, each disease's specific risk factors and recommendations, all numeric indicators, risk tiers, evidence-source labels (corpus relation / general-knowledge supplement) and action items must be retained.
- Merging duplicated wording and trimming redundancy is allowed, but facts, numbers, conclusions and recommendations must not be removed.
- The language must match the source report (English report -> English output); keep technical terms as-is.

# Output format (strict)
Output a JSON object of the form:
{
  "root": "page",
  "elements": {
    "page": {"type": "Page", "props": {"title": "...", "subtitle": "...", "patient": "...", "generatedAt": "..."}, "children": ["sec-1", "sec-2"]},
    "sec-1": {"type": "Section", "props": {"title": "1. Individual Risk Profile", "icon": "dashboard", "level": 1}, "children": ["p-1"]},
    "p-1": {"type": "Paragraph", "props": {"text": "..."}}
  }
}
- Put all elements in elements and reference child ids via children (ids are yours to choose, must be unique).
- Use only the component types below; do not invent new ones.

# Available components
__CATALOG__

# Structural requirements
1. The root must be Page; its children correspond in order to the report's six top-level chapters (Section). Keep the source chapter titles (English titles for an English report).
2. Chapter 1 "Individual Risk Profile": write only a textual risk-profile summary (Paragraph/Callout); **do not list all scores** - the dashboard is injected separately by the system.
3. Chapters 2 "Cross-Specialty Integrated Interpretation", 3 "Shared Risk Factors and Risk-Hub Synergy Analysis" and 4 "Multi-Disease Primary Prevention Strategy": express with Paragraph/List/Callout/Evidence, keeping restrained wording such as "the corpus provides no evidence for this association".
4. Chapter 5 "Disease-Specific Risk Factors and Primary Prevention Recommendations": one SubSection per disease, containing Paragraph + List (recommendations) + Evidence (source).
5. Chapter 6 "Overall Actionable Plan": use an ActionItem list (priority High/Medium/Low) or a Timeline.
6. Key numbers (e.g. HbA1c 10.5%, BMI 17.1, risk percentages) must appear verbatim in props text.
7. Keep evidence-source labels: write corpus relations into the Evidence source field and general knowledge as "General knowledge supplement".

# Report to convert
__REPORT__

Output JSON only, with no explanation or code-fence markers."""


def _strip_code_fence(text):
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if m:
        return m.group(1).strip()
    return text


def convert_to_jsonrender(markdown_report):
    prompt = JSONRENDER_PROMPT.replace(
        "__CATALOG__", catalog.describe_for_prompt()
    ).replace("__REPORT__", markdown_report)
    raw = _llm_chat(
        [{"role": "user", "content": prompt}],
        max_tokens=JSONRENDER_MAX_TOKENS,
        temperature=0.1,
        json_mode=True,
    )
    spec = json.loads(_strip_code_fence(raw))
    errors = catalog.validate_spec(spec)
    if errors:
        # Retry once on error
        retry_prompt = (
            prompt
            + "\n\n# The previous output had these structural errors; fix them and output the complete JSON again:\n- "
            + "\n- ".join(errors[:10])
        )
        raw = _llm_chat(
            [{"role": "user", "content": retry_prompt}],
            max_tokens=JSONRENDER_MAX_TOKENS,
            temperature=0.1,
            json_mode=True,
        )
        spec = json.loads(_strip_code_fence(raw))
        errors = catalog.validate_spec(spec)
    return spec, errors


# ===================== Risk dashboard injection =====================
# Disease -> organ/topic icon (ui_icons/report_ui/*.svg)
DISEASE_ICON_RULES = [
    (("kidney", "CKD", "renal"), "kidney"),
    (("lung", "COPD", "asthma", "pulmonary"), "lung"),
    (("liver", "HCC", "fatty liver", "NAFLD", "hepat"), "liver"),
    (("heart", "coronary", "CHD", "ASCVD", "stroke", "brain"), "heart"),
    (("thyroid",), "thyroid"),
    (("prostate",), "prostate"),
    (("breast",), "breast"),
    (("colorectal", "colon"), "colon"),
    (("gastric", "stomach"), "stomach"),
    (("osteo", "bone"), "bone"),
    (("diabetes", "metabolic"), "screening"),
]


def disease_icon(name, english=""):
    text = f"{name} {english}"
    for keys, icon in DISEASE_ICON_RULES:
        if any(k in text for k in keys):
            return icon
    return "risk"


# Unified tier -> 0-100 gauge position (low 0-33 / medium 33-66 / high 66-100)
TIER_POS = {4: 95, 3: 82, 2: 50, 1: 15, 0: 5}


def _short_model_name(name):
    """Trim model names: drop suffixes such as _result/_scored for display."""
    text = str(name or "")
    for suffix in ("_result", "_scored", "_screening", "_model"):
        text = text.replace(suffix, "")
    return text


def build_dashboard_items(diseases, assessments=None):
    """Convert disease scores into dashboard entries.

    One entry per disease, containing:
      - tier/tierLabel/color/icon: unified tier and visual encoding
      - pos: position mapped to 0-100 (unified tier thresholds: low 0-33 / medium 33-66 / high 66-100)
      - models: scores and tiers of all raw models under this disease (display only, not used for ranking)
    """
    assess_map = {a.get("disease_name"): a for a in (assessments or [])}
    items = []
    for d in sorted(diseases or [], key=lambda x: -(x.get("risk_tier") or 0)):
        tier = d.get("risk_tier") or 0
        if tier <= 0:
            continue
        name = d.get("disease_name") or ""
        a = assess_map.get(name) or {}
        models = []
        for m in (a.get("model_results") or []):
            score = m.get("score")
            if isinstance(score, float):
                score_txt = f"{score:.2f}%"
            else:
                score_txt = str(score)
            models.append({
                "name": _short_model_name(m.get("model")),
                "score": score_txt,
                "level": m.get("risk_level") or "",
            })
        items.append({
            "disease": name,
            "english": d.get("english_name") or "",
            "tier": tier,
            "tierLabel": TIER_LABEL.get(tier, "Unknown"),
            "color": TIER_COLOR.get(tier, "green"),
            "icon": disease_icon(name, d.get("english_name") or ""),
            "pos": TIER_POS.get(tier, 50),
            "models": models,
        })
    return items


# Numeric abnormality -> clinical significance (only for the "clinical significance" column; no priority grading)
_NUMERIC_MEANING = [
    (("HbA1c", "glycated hemoglobin"), ">=", 10.0, "Long-term glucose markedly elevated; endocrinology assessment advised"),
    (("HbA1c", "glycated hemoglobin"), ">=", 6.5, "Suggests long-term glucose elevation; confirm on repeat"),
    (("fasting glucose", "GLU", "glucose"), ">=", 13.9, "Glucose markedly elevated; seek medical assessment promptly"),
    (("fasting glucose", "GLU", "glucose"), ">=", 7.0, "Glucose elevated; confirm on repeat"),
    (("triglyceride", "TG"), ">=", 5.6, "Severely elevated; prompt assessment advised"),
    (("triglyceride", "TG"), ">=", 2.3, "Elevated; lifestyle intervention and recheck advised"),
    (("LDL", "low-density lipoprotein"), ">=", 4.1, "Markedly elevated; assess lipid-lowering therapy"),
    (("total cholesterol", "CHOL", "TC"), ">=", 6.2, "Elevated; assess lipid-lowering therapy"),
    (("hemoglobin", "haemoglobin", "HGB"), "<=", 70, "Severely low; identify the cause promptly"),
    (("creatinine", "Cr", "CREA"), ">=", 707, "Markedly elevated; prompt nephrology assessment advised"),
    (("creatinine", "Cr", "CREA"), ">=", 115, "Elevated; assess renal function"),
    (("potassium", "K"), ">=", 6.0, "Elevated; prompt assessment advised"),
    (("creatine kinase", "CK"), ">=", 5000, "Markedly elevated; prompt assessment advised"),
    (("uric acid", "UA"), ">=", 540, "Elevated; intervention advised"),
    (("visual acuity",), "<=", 0.3, "Markedly reduced; ophthalmologic examination advised"),
]

# Keyword -> clinical significance (order is match priority; specific rules first)
FINDING_RULES = [
    (("right kidney", "renal fossa", "absent", "solitary kidney"), "May be renal agenesis / ectopic kidney / prior nephrectomy / bowel-gas obscuration; if a solitary kidney is confirmed, adjust BP targets, avoid NSAIDs, monitor creatinine and potassium on ACEI/ARB, and use contrast cautiously"),
    (("mass", "lump", "hypoechoic", "nodule"), "Imaging lesion requiring further characterisation; specialist assessment advised (contrast imaging if needed)"),
    (("breast",), "Breast specialist assessment advised (ultrasound BI-RADS grading); mammography if indicated"),
    (("carotid", "plaque"), "Early atherosclerosis; focus on blood pressure/lipid/glucose risk-factor control and periodic carotid ultrasound follow-up"),
    (("helicobacter", "h. pylori", "hp"), "Active infection and a definite risk factor for gastric cancer and peptic ulcer; gastroenterology assessment for eradication advised"),
    (("MCV", "MCH", "microcyt", "hypochrom"), "Microcytic hypochromic change; check ferritin/serum iron/TIBC and haemoglobin electrophoresis (in thalassaemia-endemic areas); do not give blind iron before iron deficiency is confirmed"),
    (("hemoglobin", "haemoglobin", "HGB", "red blood cell"), "Anaemia-related assessment; combine with iron studies and GI blood-loss screening"),
    (("transaminase", "ALT", "AST"), "Hepatocellular injury marker elevated; assess with viral hepatitis, alcohol, drugs and imaging"),
    (("AFP", "CEA", "CA-", "CA19", "CA72", "CYFRA", "NSE", "SCC", "tumor marker"), "Tumour-marker elevation; interpret with imaging and specialist opinion; a single mild elevation is not cancer"),
    (("TSH", "T3", "T4", "thyroglobulin antibody", "peroxidase antibody"), "Thyroid function or autoimmune abnormality; endocrinology assessment advised"),
    (("thyroid",), "Recommend TI-RADS grading; combine with TSH to set the follow-up interval"),
    (("aortic", "aorta"), "Interpret with echocardiography; if the echo diameter is normal this usually reflects a prominent/tortuous aortic knob; consider aortic CTA only if indicated"),
    (("fatty liver", "hepatic echogenicity", "hepatic fat"), "Fatty liver is an early manifestation of metabolic liver injury; focus on weight, lipids and glucose"),
    (("liver", "hepat"), "Liver-related indicator or imaging change; follow up with liver function and metabolic markers"),
    (("uric acid", "urate"), "Uric acid metabolic abnormality; attend to diet and follow-up"),
    (("bone mass", "bone density", "osteopenia"), "Reduced bone mass; advise calcium and vitamin D, weight-bearing exercise, and periodic recheck"),
    (("visual acuity", "intraocular pressure", "fundus", "eye"), "Ophthalmologic abnormality; specialist eye examination advised"),
]


def _negated(text, term, window=10):
    """Check whether term appears in a negated context in text, e.g. "no obvious sclerotic plaque seen"."""
    for m in re.finditer(re.escape(term), text):
        head = text[max(0, m.start() - window):m.start()]
        if re.search(r"(no|not seen|not found|without|absent|ruled out|denies)[^.;,]{0,12}$", head, re.I):
            return True
    return False


def _bp_meaning(text):
    """Blood pressure: add an advisory note based on the higher of systolic/diastolic."""
    m = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", text)
    if not m:
        return None
    s, d = int(m.group(1)), int(m.group(2))
    if s >= 180 or d >= 110:
        return "Blood pressure markedly elevated; seek medical assessment promptly"
    if s >= 140 or d >= 90:
        return "Blood pressure elevated; confirm on repeat and assess target-organ damage"
    if s >= 130 or d >= 85:
        return "Blood pressure high-normal; lifestyle intervention and regular monitoring advised"
    return None


def _finding_advice(text):
    """Return the clinical significance of the abnormal item (advisory wording, no priority grading)."""
    if "blood pressure" in text.lower():
        m = _bp_meaning(text)
        if m:
            return m

    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    if m:
        val = float(m.group(1))
        for keys, op, thr, meaning in _NUMERIC_MEANING:
            if not any(k in text for k in keys):
                continue
            # "HbA1c" must not match the "hemoglobin" rule
            if ("hba1c" in text.lower() or "glycated" in text.lower()) and "hemoglobin" in keys:
                continue
            if (op == ">=" and val >= thr) or (op == "<=" and val <= thr):
                return meaning

    for keys, meaning in FINDING_RULES:
        if not any(k in text for k in keys):
            continue
        if "plaque" in text.lower() and _negated(text, "plaque"):
            continue
        return meaning
    return "Requires further clinical assessment"


def build_findings_table(findings):
    """Convert pre-scanned abnormal findings into a table (item / measured value / reference range / clinical significance)."""
    rows = []
    for f in findings or []:
        text = f.get("text") or ""
        meaning = _finding_advice(text)
        # Split "name value unit (ref xxx) up/down"
        m = re.match(r"^(.*?)\s+([0-9.]+)\s*([^\s(]*)\s*[(]\s*ref(?:erence)?\s*([^)]*)[)]\s*(.*)$", text, re.I)
        if m:
            name, val, unit, ref, flag = m.groups()
            rows.append([name, f"{val} {unit} {flag}".strip(), ref, meaning])
        else:
            rows.append([f"[{f.get('dept') or ''}]", text, "-", meaning])
    return {
        "headers": ["Abnormal item", "Measured value / description", "Reference range", "Clinical significance"],
        "rows": rows,
    }


def inject_findings_table(spec, findings):
    """Inject the abnormal-findings table into chapter 1 (after the dashboard)."""
    if not findings:
        return spec
    elements = spec.setdefault("elements", {})
    root = elements.get(spec.get("root")) or {}
    children = root.get("children") or []

    target_id = None
    for cid in children:
        el = elements.get(cid) or {}
        title = str((el.get("props") or {}).get("title") or "")
        if "individual risk profile" in title.lower() or title.startswith("1"):
            target_id = cid
            break
    if target_id is None and children:
        target_id = children[0]

    table_id = "findings-table"
    elements[table_id] = {
        "type": "Table",
        "props": build_findings_table(findings),
    }
    if target_id and target_id in elements:
        el = elements[target_id]
        kids = list(el.get("children") or [])
        # Place after the dashboard
        pos = 1 if kids and elements.get(kids[0], {}).get("type") == "RiskDashboard" else 0
        kids.insert(pos, table_id)
        el["children"] = kids
    else:
        root.setdefault("children", []).insert(0, table_id)
    return spec


def inject_page_meta(spec, patient=None):
    """Programmatically set the page header: keep only age/sex; no name, exam number or generation time."""
    elements = spec.get("elements") or {}
    root = elements.get(spec.get("root"))
    if not root:
        return spec
    props = root.setdefault("props", {})
    props["title"] = "Personal Multimorbidity Risk Interpretation and Health Management Report"
    props["subtitle"] = "Primary Prevention and Health Management"
    patient = patient or {}
    age = str(patient.get("age") or "").strip()
    sex = str(patient.get("sex") or "").strip()
    parts = []
    if age:
        parts.append(f"{age} years")
    if sex in ("Male", "Female"):
        parts.append(sex)
    elif sex:
        parts.append(sex)
    props["patient"] = "".join(parts) if parts else ""
    props["generatedAt"] = ""   # generation time is intentionally not shown
    return spec


def inject_dashboard(spec, items):
    """Inject RiskDashboard at the start of chapter 1 (Individual Risk Profile)."""
    if not items:
        return spec
    elements = spec.setdefault("elements", {})
    root = spec.get("root")
    root_el = elements.get(root) or {}
    children = root_el.get("children") or []

    # Locate chapter 1 (heading contains "Individual Risk Profile" or "1.")
    target_id = None
    for cid in children:
        el = elements.get(cid) or {}
        title = str((el.get("props") or {}).get("title") or "")
        if "individual risk profile" in title.lower() or title.startswith("1"):
            target_id = cid
            break
    if target_id is None and children:
        target_id = children[0]

    dash_id = "risk-dashboard"
    elements[dash_id] = {
        "type": "RiskDashboard",
        "props": {"title": "Individual Risk Dashboard (sorted by risk tier)", "items": items},
    }
    if target_id and target_id in elements:
        el = elements[target_id]
        el["children"] = [dash_id] + list(el.get("children") or [])
    else:
        root_el.setdefault("children", []).insert(0, dash_id)
    return spec


# ===================== Pipeline entry point =====================
# ===================== Global task-queue scheduler =====================
class _PatientJob:
    """Task status for a single patient.

    A patient's task chain is serialized by dependencies (prepare -> knowledge* -> report ->
    jsonrender); tasks from different patients are fully independent and can be picked up by any worker.
    finalize is lock-protected so timeout marking and task completion cannot overwrite each other.
    """

    __slots__ = (
        "rec", "app_no", "edges_path", "batch", "info", "t0",
        "archive", "key_info", "corpus", "chunks", "knowledge_parts",
        "pending_chunks", "failed_chunks", "md", "md_finish", "spec_finish",
        "findings", "diseases", "assessments", "errors", "warnings",
        "lock", "finalized",
    )

    def __init__(self, rec, batch, info):
        self.rec = rec
        self.app_no = str(rec.get("eid") or "")
        self.edges_path = (rec.get("subgraph_files") or {}).get("edges_json")
        self.batch = batch or {}
        self.info = info or {}
        self.t0 = time.time()
        self.archive = {}
        self.key_info = ""
        self.corpus = ""
        self.chunks = []
        self.knowledge_parts = []
        self.pending_chunks = 0
        self.failed_chunks = []
        self.md = ""
        self.md_finish = None
        self.spec_finish = None
        self.findings = []
        self.diseases = []
        self.assessments = []
        self.errors = []
        self.warnings = []
        self.lock = threading.Lock()
        self.finalized = False


class _ReportScheduler:
    """Global task-queue scheduler.

    Design notes:
      1. The queue holds tasks, not patients: prepare / knowledge / report /
         jsonrender tasks are enqueued independently; whichever finishes first requeues the next stage;
      2. Worker count = REPORT_LLM_MAX_INFLIGHT, limiting in-flight tasks,
         independent of the patient count;
      3. Retry failed tasks per REPORT_TASK_RETRIES; on permanent knowledge-chunk failure use the raw
         evidence fallback so the patient still gets a report;
      4. Overall wait cap REPORT_LLM_WAIT_TIMEOUT; timed-out patients are marked TIMEOUT and returned,
         the pipeline continues and never stalls;
      5. When all tasks finish, outstanding reaches zero and the main thread returns without waiting for leftover threads.
    """

    def __init__(self, jobs, workers, wait_timeout, retries, progress_cb=None):
        self.jobs = jobs
        self.workers = max(1, int(workers))
        self.wait_timeout = float(wait_timeout)
        self.retries = max(1, int(retries))
        self.progress_cb = progress_cb
        self.q = queue.Queue()
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.outstanding = 0
        self.stop = False
        self.total_jobs = len(jobs)
        self.progress_lock = threading.Lock()
        self.progress_idx = 0
        self.threads = []
        self.stats = {"tasks": 0, "retries": 0, "failed_tasks": 0, "fallbacks": 0}

    # ---------- Queue operations ----------
    def submit(self, kind, job, payload=None, attempt=0):
        with self.lock:
            if self.stop:
                return False
            self.outstanding += 1
        self.q.put((kind, job, payload, attempt))
        return True

    def _finish_task(self):
        with self.lock:
            self.outstanding -= 1
            if self.outstanding <= 0:
                self.cv.notify_all()

    # ---------- Lifecycle ----------
    def run(self):
        for i in range(self.workers):
            t = threading.Thread(target=self._worker, name=f"report-w{i}", daemon=True)
            t.start()
            self.threads.append(t)
        deadline = time.time() + self.wait_timeout
        with self.lock:
            while self.outstanding > 0 and not self.stop:
                remain = deadline - time.time()
                if remain <= 0:
                    break
                self.cv.wait(timeout=min(remain, 5.0))
            timed_out = self.outstanding > 0
            self.stop = True
            self.cv.notify_all()
        if timed_out:
            print(
                f"[report] concurrency wait timed out (>{self.wait_timeout:.0f}s), "
                f"{self.outstanding} tasks still unfinished; marking TIMEOUT and continuing"
            )
            for job in self.jobs:
                self._finalize(
                    job, f"TIMEOUT: concurrency wait exceeded {self.wait_timeout:.0f}s"
                )
        # Drain the queue so leftover tasks do not keep running
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        return self.stats

    # ---------- worker ----------
    def _worker(self):
        while True:
            try:
                kind, job, payload, attempt = self.q.get(timeout=0.5)
            except queue.Empty:
                with self.lock:
                    if self.stop or self.outstanding <= 0:
                        return
                continue
            if self.stop:
                self._finish_task()
                continue
            with self.lock:
                self.stats["tasks"] += 1
            try:
                self._run_task(kind, job, payload)
            except Exception as e:
                self._on_task_error(kind, job, payload, attempt, e)
            else:
                self._finish_task()

    def _run_task(self, kind, job, payload):
        if kind == "prepare":
            self._do_prepare(job)
        elif kind == "knowledge":
            self._do_knowledge(job, payload)
        elif kind == "report":
            self._do_report(job)
        elif kind == "jsonrender":
            self._do_jsonrender(job)
        else:
            raise ValueError(f"Unknown task type: {kind}")

    # ---------- Failure handling ----------
    def _on_task_error(self, kind, job, payload, attempt, e):
        msg = f"{type(e).__name__}: {e}"
        print(
            f"[report] task failed kind={kind} eid={job.app_no} "
            f"attempt={attempt + 1}/{self.retries}: {msg}"
        )
        if attempt + 1 < self.retries and not self.stop:
            with self.lock:
                self.stats["retries"] += 1
            time.sleep(min(2 * (attempt + 1), 10))
            # Requeue: the task is still in flight, so outstanding is unchanged
            self.q.put((kind, job, payload, attempt + 1))
            return
        with self.lock:
            self.stats["failed_tasks"] += 1
        if kind == "knowledge":
            # Fallback: use the raw evidence block so the patient still gets a report
            idx = payload
            with job.lock:
                if job.finalized:
                    ready = False
                else:
                    raw = job.chunks[idx] if idx < len(job.chunks) else ""
                    job.knowledge_parts[idx] = raw[:REPORT_KNOWLEDGE_FALLBACK_CHARS]
                    job.failed_chunks.append(idx)
                    job.warnings.append(
                        f"Knowledge-selection chunk {idx + 1} failed; fell back to raw evidence"
                    )
                    job.pending_chunks -= 1
                    ready = job.pending_chunks == 0
            with self.lock:
                self.stats["fallbacks"] += 1
            if ready:
                self.submit("report", job)
        elif kind == "prepare":
            self._finalize(job, f"ERROR: prepare stage failed: {msg}")
        elif kind == "report":
            self._finalize(job, f"ERROR: report generation failed: {msg}")
        elif kind == "jsonrender":
            # md already generated; keep it when structured conversion fails and mark success with a warning
            self._finalize(
                job, "SUCCESS_WITH_WARNINGS",
                extra_warning=f"structured conversion failed: {msg}",
            )
        self._finish_task()

    # ---------- Task implementation ----------
    def _do_prepare(self, job):
        import jev27b_decider as J

        if not job.edges_path or not os.path.exists(job.edges_path):
            self._finalize(job, "NO_SUBGRAPH")
            return

        # 1) Key information (full record, no denoising: keep normal items visible)
        archive = J.fetch_patient_archive(job.app_no)
        archive_text, _ = J.build_archive_text(archive, job.app_no, denoise=False)
        assessments = job.batch.get("computable") or []
        diseases = job.info.get("diseases") or []
        findings = extract_abnormal_findings(archive)
        interpretation = build_clinical_interpretation(archive)
        key_info = build_key_info_with_uncomputable(
            archive_text, diseases, assessments, job.batch.get("not_computable"),
            findings, interpretation,
        )

        # 2) Corpus
        corpus = build_corpus(job.edges_path)

        job.archive = archive
        job.key_info = key_info
        job.corpus = corpus
        job.findings = findings
        job.diseases = diseases
        job.assessments = assessments

        # 3) Knowledge selection: chunk by evidence boundary, enqueue one task per chunk
        if KNOWLEDGE_STEP:
            chunks = _split_corpus(corpus, KNOWLEDGE_CHUNK_CHARS)
            job.chunks = chunks
            job.knowledge_parts = [None] * len(chunks)
            job.pending_chunks = len(chunks)
            for i in range(len(chunks)):
                self.submit("knowledge", job, i)
        else:
            job.chunks = [corpus]
            job.knowledge_parts = [corpus]
            self.submit("report", job)

    def _do_knowledge(self, job, idx):
        text = _filter_one(job.key_info, job.chunks[idx], idx + 1, len(job.chunks))
        with job.lock:
            if job.finalized:
                return
            job.knowledge_parts[idx] = text
            job.pending_chunks -= 1
            ready = job.pending_chunks == 0
        if ready:
            self.submit("report", job)

    def _do_report(self, job):
        if self.stop:
            return
        knowledge = "\n\n".join(p for p in job.knowledge_parts if p) or job.corpus
        if KNOWLEDGE_STEP:
            kn_path = Path(job.edges_path).with_name(
                f"report_{job.app_no}_knowledge.md"
            )
            kn_path.write_text(knowledge, encoding="utf-8")

        md = _sanitize_output(generate_markdown(job.key_info, knowledge))
        md_finish = last_finish_reason()
        if self.stop:
            return
        consistency_warnings = check_negative_consistency(md, job.archive)
        md_path = Path(job.edges_path).with_name(f"report_{job.app_no}.md")
        md_path.write_text(md, encoding="utf-8")
        with job.lock:
            if job.finalized:
                return
            job.md = md
            job.md_finish = md_finish
            job.warnings.extend(consistency_warnings)
        self.submit("jsonrender", job)

    def _do_jsonrender(self, job):
        import jev27b_decider as J

        if self.stop:
            return
        spec, errors = convert_to_jsonrender(job.md)
        spec_finish = last_finish_reason()
        if self.stop:
            return
        spec = inject_page_meta(spec, {
            "age": job.archive.get("app_pat_age"),
            "sex": J._sex_label(job.archive.get("pat_sex")),
        })
        spec = inject_dashboard(
            spec, build_dashboard_items(job.diseases, job.assessments)
        )
        spec = inject_findings_table(spec, job.findings)
        spec_path = Path(job.edges_path).with_name(f"report_{job.app_no}.json")
        spec_path.write_text(
            json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with job.lock:
            if job.finalized:
                return
            job.spec_finish = spec_finish
            job.errors = errors
        status = "SUCCESS" if not errors else "SUCCESS_WITH_WARNINGS"
        self._finalize(job, status, spec_path=spec_path, spec=spec)

    # ---------- Wrap-up ----------
    def _finalize(self, job, status, extra_warning=None, spec_path=None, spec=None):
        import jev27b_decider as J

        with job.lock:
            if job.finalized:
                return
            job.finalized = True
            warnings = list(job.warnings)
            errors = list(job.errors)
            md = job.md
            md_finish = job.md_finish
            spec_finish = job.spec_finish
            key_info = job.key_info
            corpus = job.corpus
            knowledge = "\n\n".join(p for p in job.knowledge_parts if p) or corpus
            archive = job.archive
            failed_chunks = list(job.failed_chunks)
        if extra_warning:
            warnings.append(extra_warning)

        rec = job.rec
        rec["report_status"] = status
        if md:
            md_path = Path(job.edges_path).with_name(f"report_{job.app_no}.md")
            files = {"markdown": str(md_path)}
            if spec_path:
                files["jsonrender"] = str(spec_path)
            rec["report_files"] = files

        meta = {
            "md_chars": len(md),
            "spec_elements": len((spec or {}).get("elements") or {}),
            "validation_errors": errors,
            "key_info_chars": len(key_info),
            "corpus_chars": len(corpus),
            "knowledge_chars": len(knowledge),
            "knowledge_step": KNOWLEDGE_STEP,
            "knowledge_failed_chunks": failed_chunks,
            "consistency_warnings": warnings,
            "md_finish_reason": md_finish,
            "spec_finish_reason": spec_finish,
            "truncated": md_finish == "length" or spec_finish == "length",
            "elapsed_seconds": round(time.time() - job.t0, 1),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "patient": {
                "name": archive.get("pat_name") or archive.get("app_pat_name") or "",
                "sex": J._sex_label(archive.get("pat_sex")),
                "age": archive.get("app_pat_age") or "",
            },
        }
        rec["report_meta"] = meta
        if job.edges_path:
            try:
                meta_path = Path(job.edges_path).with_name(
                    f"report_{job.app_no}_meta.json"
                )
                meta_path.write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as e:
                print(f"[report] failed to write meta eid={job.app_no}: {e}")
        self._emit(job)

    def _emit(self, job):
        if not self.progress_cb:
            return
        with self.progress_lock:
            self.progress_idx += 1
            idx = self.progress_idx
        try:
            self.progress_cb(
                idx, self.total_jobs, job.app_no, job.rec.get("report_status")
            )
        except Exception:
            pass


# ===================== Pipeline entry point =====================
def generate_reports(records, batch_results, per_patient=None, progress_cb=None):
    """Generate md reports + json-render specs for successfully retrieved patients (global task queue, concurrent).

    Concurrency model (unlike the old patient-level thread pool):
      - The queue holds tasks: prepare / knowledge (one per chunk) / report / jsonrender;
      - worker count = REPORT_LLM_MAX_INFLIGHT, limiting in-flight tasks rather than patients;
      - On completion the next-stage task is requeued immediately and picked up by an idle worker with no wait;
      - A single slow request occupies one worker while other patients' tasks proceed;
      - Overall wait cap REPORT_LLM_WAIT_TIMEOUT; timed-out patients are marked TIMEOUT and the pipeline continues;
      - progress_cb is called serially under a lock, so the UI sees the same monotonic progress as serial execution.
    """
    if not REPORT_ENABLED:
        for r in records:
            r.setdefault("report_status", "DISABLED")
        return records

    batch_map = {str(b.get("app_no")): b for b in (batch_results or [])}
    targets = [r for r in records if r.get("status") == "SUCCESS"]
    total = len(targets)
    if not total:
        return records

    jobs = [
        _PatientJob(
            rec,
            batch_map.get(str(rec.get("eid") or "")),
            (per_patient or {}).get(str(rec.get("eid") or "")),
        )
        for rec in targets
    ]

    workers = max(1, REPORT_LLM_MAX_INFLIGHT)
    print(
        f"[report] LLM endpoint {LLM_URL} model {LLM_MODEL}; "
        f"global task-queue concurrency: {total} patients, {workers} workers"
        f"（REPORT_LLM_MAX_INFLIGHT={REPORT_LLM_MAX_INFLIGHT}），"
        f"per-request timeout {LLM_TIMEOUT:.0f}s, overall wait cap "
        f"{REPORT_LLM_WAIT_TIMEOUT:.0f}s"
    )

    sched = _ReportScheduler(
        jobs, workers, REPORT_LLM_WAIT_TIMEOUT, REPORT_TASK_RETRIES,
        progress_cb=progress_cb,
    )
    for job in jobs:
        sched.submit("prepare", job)
    stats = sched.run()
    print(
        f"[report] task queue finished: {stats['tasks']} tasks, "
        f"{stats['retries']} retries, {stats['failed_tasks']} failed tasks, "
        f"{stats['fallbacks']} knowledge fallbacks"
    )
    return records
