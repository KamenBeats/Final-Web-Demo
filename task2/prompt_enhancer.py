"""
task2/prompt_enhancer.py — Task-aware SDXL prompt enhancement for Web Demo.

Adapted from SDXL_Lora/src/inference/prompt_enhancer.py.
Uses M.pipe / M.qwen_* globals directly; handles GPU management internally.
Returns (positive_prompt, negative_prompt) tuple.
"""

import re
import json as _json
import torch

from . import model as M


# ─────────────────────────────────────────────────────────────────
# 1. CLIP token budget helpers
# ─────────────────────────────────────────────────────────────────
_CLIP_MAX_TOKENS    = 75
_CLIP_TARGET_TOKENS = 73


def _clip_tokens(text: str) -> int:
    """Count CLIP tokens (excluding BOS/EOS)."""
    try:
        return len(M.pipe.tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        words = re.findall(r"[a-zA-Z0-9']+|[^a-zA-Z0-9'\s]", text)
        return int(len(words) * 1.35)


def _truncate_to_clip(text: str, max_tokens: int = _CLIP_TARGET_TOKENS) -> str:
    """Safety truncation — drop tags from the end until within budget."""
    if _clip_tokens(text) <= max_tokens:
        return text
    try:
        ids  = M.pipe.tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
        text = M.pipe.tokenizer.decode(ids, skip_special_tokens=True).strip().rstrip(",").strip()
        if "," in text:
            text = ",".join(text.split(",")[:-1]).strip()
        return text
    except Exception:
        tags = [t.strip() for t in text.split(",") if t.strip()]
        while tags and _clip_tokens(", ".join(tags)) > max_tokens:
            tags.pop()
        return ", ".join(tags)


# ─────────────────────────────────────────────────────────────────
# 2. Task type detection  (EN + VI bilingual)
# ─────────────────────────────────────────────────────────────────
_ADD_KEYWORDS = (
    r"\b(add|place|put|insert|include|bring|introduce|hang|install|lay|set|"
    r"thêm|đặt|đưa vào|gắn|lắp|treo)\b"
)
_REMOVE_KEYWORDS = (
    r"\b(remove|delete|erase|eliminate|take out|get rid of|clear|clean up|"
    r"xóa|bỏ|dọn|loại bỏ|xoá)\b"
)
_EDIT_KEYWORDS = (
    r"\b(change|replace|modify|update|repaint|recolor|recolour|swap|turn|make|"
    r"style|redesign|renovate|transform|convert|thay|đổi|sửa|sơn|chỉnh)\b"
)


def _detect_task(prompt: str) -> str:
    """Returns 'add' | 'remove' | 'edit' | 'unknown'."""
    p = prompt.lower()
    if re.search(_ADD_KEYWORDS,    p): return "add"
    if re.search(_REMOVE_KEYWORDS, p): return "remove"
    if re.search(_EDIT_KEYWORDS,   p): return "edit"
    return "unknown"


def _extract_remove_object(raw: str) -> str:
    """Extract the object name from a remove instruction."""
    cleaned = re.sub(
        r",?\s*\b(keep|preserve|maintain|retain|giữ lại|giữ nguyên)\b[^,]*",
        "", raw, flags=re.IGNORECASE
    )
    cleaned = re.sub(_REMOVE_KEYWORDS, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(the|a|an|from|off|out|away|please|just)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\b(on|against|from|off|in|at|near)\s+(the\s+)?"
        r"(wall|floor|ground|ceiling|desk|table|shelf|counter|countertop|corner|side|tường|sàn|bàn|kệ|góc)\b",
        "", cleaned, flags=re.IGNORECASE
    )
    cleaned = re.sub(
        r"\b(tr[eê]n|dưới|trong|tại|cạnh)\s+"
        r"(bàn|kệ|mặt bàn|quầy|tường|sàn|nền|thảm|góc|trần|cửa)\b",
        "", cleaned, flags=re.IGNORECASE
    )
    cleaned = re.sub(
        r"\b(wall|floor|ground|ceiling|corner|tường|sàn|góc)\b",
        "", cleaned, flags=re.IGNORECASE
    )
    cleaned = re.sub(
        r"\b(white|light|dark|gray|grey|beige|cream|black|brown|painted|plaster)\b",
        "", cleaned, flags=re.IGNORECASE
    )
    cleaned = re.sub(r"\s+(tr[eê]n|dưới|trong|from|in|on|at|near|off)\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip(",").strip()
    return cleaned if cleaned else raw


# ── Location / surface context for REMOVE task ───────────────────
_LOC_WALL_KEYWORDS = (
    r"\b(wall|ceiling|panel|backdrop|partition|accent wall|paint|"
    r"tường|trần|vách)\b"
)
_LOC_FLOOR_KEYWORDS = (
    r"\b(floor|ground|carpet|rug|mat|tile|parquet|hardwood|vinyl|flooring|"
    r"sàn|nền|thảm)\b"
)
_LOC_SURFACE_KEYWORDS = (
    r"\b(desk|table|shelf|shelves|counter|countertop|cabinet|dresser|nightstand|"
    r"mantel|ledge|bench|console|sideboard|"
    r"bàn|kệ|mặt bàn|quầy)\b"
)
_LOC_CORNER_KEYWORDS = (
    r"\b(corner|side|edge|alcove|góc)\b"
)
_OBJ_IMPLIES_WALL = (
    r"\b(wardrobe|armoire|bookcase|bookshelf|cabinet|cupboard|closet|dresser|"
    r"refrigerator|fridge|oven|washing machine|dryer|tv unit|entertainment center|"
    r"shelving unit|display cabinet|chest of drawers|tallboy|"
    r"tủ quần áo|tủ sách|tủ lạnh|tủ bếp|kệ sách|máy giặt)\b"
)
_OBJ_IMPLIES_FLOOR = (
    r"\b(sofa|couch|chair|armchair|stool|ottoman|coffee table|side table|"
    r"end table|plant pot|floor lamp|stand|"
    r"ghế|sofa|đi văng|thảm|đèn sàn)\b"
)
_OBJ_IMPLIES_SURFACE = (
    r"\b(vase|lamp|tissue|remote|phone|cup|mug|bottle|box|frame|picture|clock|"
    r"candle|book|bowl|plate|tray|speaker|monitor|keyboard|mouse|ornament|figurine|"
    r"bình hoa|đèn bàn|hộp khăn|điều khiển|khung ảnh|nến|đồng hồ)\b"
)

_BG_COLOR_RE = re.compile(
    r"\b(?:(?:white|light|dark|gray|grey|beige|cream|black|brown|blue|green|yellow|red|"
    r"off-white|charcoal|ivory)\s+)?"
    r"(painted\s+wall|plaster\s+wall|brick\s+wall|concrete\s+wall|"
    r"wooden\s+wall|wallpaper|drywall|marble\s+floor|hardwood\s+floor|"
    r"tile(?:d)?\s+floor|carpet|granite\s+counter|marble\s+counter|"
    r"(?:white|light|dark|gray|grey|beige|cream|black|brown)\s+wall|"
    r"(?:white|light|dark|gray|grey|beige|cream|brown)\s+floor)",
    re.IGNORECASE,
)


def _extract_remove_context(raw: str) -> dict:
    obj  = _extract_remove_object(raw)
    low  = raw.lower()

    bg_match = _BG_COLOR_RE.search(raw)
    bg_hint  = bg_match.group(0).strip() if bg_match else ""

    if re.search(r"\b(on|against|from|off)\s+(the\s+)?(wall|ceiling|panel)\b", low) \
       or re.search(r"\b(tr[eê]n|t[aă]i)\s+(t\u01b0\u1eddng)\b", low):
        location = "wall"
    elif re.search(r"\b(on|from|off)\s+(the\s+)?(floor|ground|carpet|rug|tile)\b", low) \
         or re.search(r"\b(tr[eê]n|d\u01b0\u1edbi)\s+(s\u00e0n|n\u1ec1n|th\u1ea3m)\b", low):
        location = "floor"
    elif re.search(r"\b(on|from|off)\s+(the\s+)?(desk|table|shelf|counter|countertop)\b", low) \
         or re.search(r"\b(tr[eê]n)\s+(b\u00e0n|k\u1ec7|m\u1eb7t b\u00e0n|qu\u1ea7y)\b", low):
        location = "surface"
    elif re.search(_LOC_WALL_KEYWORDS,    low): location = "wall"
    elif re.search(_LOC_FLOOR_KEYWORDS,   low): location = "floor"
    elif re.search(_LOC_SURFACE_KEYWORDS, low): location = "surface"
    elif re.search(_LOC_CORNER_KEYWORDS,  low): location = "corner"
    elif re.search(_OBJ_IMPLIES_WALL,    low): location = "wall"
    elif re.search(_OBJ_IMPLIES_FLOOR,   low): location = "floor"
    elif re.search(_OBJ_IMPLIES_SURFACE, low): location = "surface"
    else:
        location = "unknown"

    return {"object": obj, "location": location, "bg_hint": bg_hint}


# ─────────────────────────────────────────────────────────────────
# 3. System prompts per task
# ─────────────────────────────────────────────────────────────────
_SYS_COMMON = (
    "You are an expert prompt engineer specialising in Stable Diffusion XL (SDXL) "
    "inpainting for interior photography.\n\n"
    "CRITICAL CONSTRAINT — CLIP TOKEN BUDGET:\n"
    "  • SDXL feeds both prompts through a CLIP text encoder with a hard limit of "
    "77 tokens (75 usable after BOS/EOS). Any tokens beyond position 75 are silently "
    "truncated and have ZERO effect on the image.\n"
    "  • positive_prompt TARGET: 60–73 tokens. SPARSE PROMPTS BELOW 50 TOKENS "
    "SEVERELY UNDERPERFORM — always fill the budget with relevant descriptive tags.\n"
    "  • negative_prompt TARGET: 60–73 tokens. Same rule — fill the budget.\n"
    "  • Use comma-separated descriptive tags, NOT full sentences. "
    "Tags should be short (1–4 words each). Never use verbs or articles.\n\n"
    "OUTPUT FORMAT:\n"
    'Return ONLY a JSON object: {"positive_prompt": "...", "negative_prompt": "..."}. '
    "No prose, no markdown fences, no explanation."
)

_SYS_TASK = {
    "add": (
        "TASK: ADD a new object into an existing interior scene.\n\n"
        "positive_prompt RULES (TARGET 60–73 tokens, comma-separated tags only):\n"
        "  1. Name the object first (e.g. 'blue velvet sofa', 'marble coffee table').\n"
        "  2. Add style tags: modern, Scandinavian, industrial, mid-century, etc.\n"
        "  3. Add material/finish tags: solid oak, brushed brass, matte black, etc.\n"
        "  4. Add photorealism anchors: photorealistic, 8k uhd, sharp focus, "
        "professional interior photography, natural lighting.\n"
        "  5. Add seamless-integration tags: physically grounded, matching ambient "
        "lighting, correct perspective, seamless shadow, realistic contact shadow.\n"
        "  TOTAL: TARGET 60–73 tokens (~15–20 tags). NO sentences, NO verbs.\n\n"
        "negative_prompt RULES (TARGET 60–73 tokens):\n"
        "  floating, levitating, wrong scale, clipping, shadow mismatch, "
        "object not touching floor, misaligned perspective, flat texture, "
        "plastic look, wrong lighting, colour inconsistency, "
        "cartoon, anime, blurry, low quality, watermark, distorted, "
        "duplicate objects, extra limbs, deformed — fill to 60–73 tokens."
    ),
    "remove": (
        "TASK: REMOVE an object. The masked area must be filled to look like "
        "the natural background surface that was behind/beneath the removed object.\n\n"
        "⚠️  CRITICAL — READ THE USER MESSAGE CAREFULLY:\n"
        "  The user message will tell you the INFERRED LOCATION (WALL / FLOOR / SURFACE / CORNER / UNKNOWN).\n"
        "  Use ONLY the matching surface type below. Do NOT mix floor tags for a wall task, etc.\n\n"
        "  WALL  → plain painted wall, smooth plaster surface, uniform wall colour, drywall, wallpaper, "
        "painted drywall, bare wall, matching wall tone, consistent paint finish\n"
        "  FLOOR → hardwood floor, oak parquet, marble tile, polished concrete, carpet, vinyl flooring, "
        "stone floor, matching floor pattern, consistent tile grout\n"
        "  SURFACE (desk/table/shelf/counter) → smooth wooden surface, granite countertop, marble surface, "
        "lacquered shelf, matching countertop, consistent surface texture\n"
        "  CORNER → matching wall junction, bare wall corner, consistent wall and floor meeting\n"
        "  UNKNOWN → infer from object type: large tall object (wardrobe/cabinet) → WALL; "
        "rug/lamp/sofa → FLOOR; small decorative item → check context\n\n"
        "  If the user message also provides an EXPLICIT BACKGROUND HINT (e.g. 'white painted wall', "
        "'marble floor'), prioritise that exact description in the positive_prompt.\n\n"
        "positive_prompt RULES (TARGET 60–73 tokens, comma-separated tags only):\n"
        "  1. DO NOT mention the removed object at all.\n"
        "  2. DO NOT add any new object or decoration.\n"
        "  3. Use the correct surface tags from the location mapping above.\n"
        "  4. Add texture/continuity: fine grain, uniform finish, consistent pattern, "
        "seamless fill, same colour tone, realistic continuation, matching ambient lighting, no visible seam.\n"
        "  5. Add quality: photorealistic, 8k uhd, sharp focus, professional interior photography.\n"
        "  TOTAL: TARGET 60–73 tokens. NO object names, NO verbs, NO new furniture.\n\n"
        "negative_prompt RULES (TARGET 60–73 tokens) — TWO MANDATORY STEPS:\n"
        "  STEP 1 — Object suppression (5–7 tags, PUT FIRST):\n"
        "    Exact object name + common synonyms + similar objects that might appear.\n"
        "  STEP 2 — Artifact & hallucination avoidance (fill remaining budget):\n"
        "    ghost outline, object residue, shadow remnant, visible seam,\n"
        "    colour inconsistency, mismatched texture, floating artefact,\n"
        "    any new object, added furniture, added decoration, hallucinated object,\n"
        "    cartoon, blurry, low quality, watermark.\n"
        "  TOTAL: TARGET 60–73 tokens."
    ),
    "edit": (
        "TASK: EDIT an existing object — change its colour, material, or style "
        "while keeping its shape, size, and position identical.\n\n"
        "positive_prompt RULES (TARGET 60–73 tokens, comma-separated tags only):\n"
        "  1. Name the object + its NEW appearance.\n"
        "  2. Add geometry-lock tags: same shape, same position, same proportions, "
        "identical silhouette.\n"
        "  3. Add material-quality tags: photorealistic texture, realistic reflections, "
        "accurate sheen, fine grain detail.\n"
        "  4. Add lighting-match tags: matching room lighting, consistent shadows, "
        "correct ambient occlusion.\n"
        "  5. Add quality anchors: photorealistic, 8k uhd, sharp focus, "
        "professional interior photography.\n"
        "  TOTAL: TARGET 60–73 tokens. NO verbs, NO sentences.\n\n"
        "negative_prompt RULES (TARGET 60–73 tokens):\n"
        "  shape deformation, size change, moved position, wrong geometry, "
        "changed silhouette, different proportions, flat colour, plastic look, "
        "wrong material, inconsistent lighting, colour bleed, "
        "cartoon, anime, blurry, low quality, watermark, "
        "distorted, duplicate — fill to 60–73 tokens."
    ),
    "unknown": (
        "TASK TYPE is unclear — treat it as a general interior inpainting edit.\n\n"
        "positive_prompt: describe the desired change using comma-separated tags "
        "(TARGET 60–73 tokens). Include photorealism anchors.\n\n"
        "negative_prompt: standard quality and artefact avoidance tags "
        "(TARGET 60–73 tokens)."
    ),
}


def _build_system(task: str) -> str:
    return _SYS_COMMON + "\n\n" + _SYS_TASK.get(task, _SYS_TASK["unknown"])


# ─────────────────────────────────────────────────────────────────
# 4. Rule-based fallback
# ─────────────────────────────────────────────────────────────────
_QUALITY_TAGS = "photorealistic, 8k uhd, sharp focus, professional lighting, professional interior photography"

_BASE_NEGATIVE = (
    "cartoon, anime, blurry, low quality, watermark, text, "
    "ugly, deformed, duplicate, floating objects, shadow mismatch, "
    "wrong perspective, distortion, jpeg artefacts, colour inconsistency"
)
_TASK_NEGATIVE_EXTRA = {
    "add":     "floating, levitating, wrong scale, clipping, misplaced, object not touching floor, flat texture",
    "remove":  "ghost outline, object residue, shadow remnant, visible seam, mismatched texture, any new object, added object",
    "edit":    "shape deformation, size change, moved position, flat colour, plastic look, wrong geometry, changed silhouette",
    "unknown": "",
}
_TASK_POSITIVE_PREFIX = {
    "add":     "photorealistic, naturally placed, physically grounded, matching ambient lighting, seamless shadow, correct perspective, realistic contact shadow, ",
    "remove":  "seamless fill, consistent floor texture, consistent wall texture, realistic continuation, same colour tone, no editing artefacts, matching ambient lighting, photorealistic surface, ",
    "edit":    "same shape, same position, same proportions, updated material, photorealistic texture, matching room lighting, identical silhouette, ",
    "unknown": "photorealistic interior, high quality, professional photography, ",
}


def _rule_fallback(raw: str, task: str):
    prefix = _TASK_POSITIVE_PREFIX[task]
    extra  = _TASK_NEGATIVE_EXTRA[task]

    if task == "remove":
        ctx = _extract_remove_context(raw)
        obj = ctx["object"]
        loc = ctx["location"]
        bg  = ctx["bg_hint"]

        _loc_prefix = {
            "wall":    bg if bg else "plain painted wall, smooth plaster wall, uniform wall colour, consistent paint finish, bare wall, same wall tone",
            "floor":   bg if bg else "hardwood floor, consistent floor texture, matching floor pattern, uniform floor surface, same tile grout",
            "surface": bg if bg else "smooth wooden surface, matching countertop, consistent surface texture, same surface material, uniform surface finish",
            "corner":  bg if bg else "plain wall, bare corner, matching wall and floor junction, consistent wall paint",
            "unknown": bg if bg else "seamless fill, consistent background texture, matching surface colour",
        }
        loc_tags = _loc_prefix.get(loc, _loc_prefix["unknown"])
        prefix   = (
            f"{loc_tags}, seamless fill, realistic continuation, same colour tone, "
            "no visible seam, matching ambient lighting, photorealistic surface, "
        )
        if obj:
            extra = f"{obj}, {extra}"

    pos_raw = f"{prefix}{raw.strip().rstrip(',')}, {_QUALITY_TAGS}" if task != "remove" else f"{prefix}{_QUALITY_TAGS}"
    neg_raw = _BASE_NEGATIVE + (f", {extra}" if extra else "")
    return _truncate_to_clip(pos_raw), _truncate_to_clip(neg_raw)


# ─────────────────────────────────────────────────────────────────
# 5. Lenient JSON parser
# ─────────────────────────────────────────────────────────────────
def _parse_llm_json(text: str):
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        d = _json.loads(text)
    except _json.JSONDecodeError:
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if not m:
            raise ValueError(f"No JSON object in LLM output:\n{text[:400]}")
        d = _json.loads(m.group())
    pos = str(d.get("positive_prompt") or d.get("positive") or "").strip()
    neg = str(d.get("negative_prompt") or d.get("negative") or "").strip()
    if not pos:
        raise ValueError("Missing 'positive_prompt' in LLM response")
    return pos, neg


# ─────────────────────────────────────────────────────────────────
# 6. Main enhance_prompt
# ─────────────────────────────────────────────────────────────────
def enhance_prompt(raw_prompt: str) -> tuple:
    """
    Convert raw user instruction → (positive_prompt, negative_prompt).
    Manages Qwen GPU/CPU transfer automatically.
    Falls back to rule-based if Qwen not loaded or LLM output is unparseable.
    """
    task = _detect_task(raw_prompt)
    print(f"[Task2 Enhance] task={task.upper()} | raw: {raw_prompt[:80]}")

    # Ensure models are loaded
    M.ensure_qwen_loaded()
    if M.qwen_tokenizer is None or M.qwen_model is None or M.pipe is None:
        print("[Task2 Enhance] Qwen or pipe not loaded → rule-based fallback")
        return _rule_fallback(raw_prompt, task)

    system  = _build_system(task)

    remove_hint = ""
    if task == "remove":
        ctx = _extract_remove_context(raw_prompt)
        obj = ctx["object"]
        loc = ctx["location"]
        bg  = ctx["bg_hint"]

        _loc_guidance = {
            "wall":    "The object was against/on a WALL. Fill the positive_prompt with WALL surface tags only (painted wall, plaster, drywall, wallpaper, etc.). Do NOT use floor tags.",
            "floor":   "The object was on the FLOOR. Fill the positive_prompt with FLOOR surface tags only (hardwood, parquet, tile, carpet, marble floor, etc.). Do NOT use wall tags.",
            "surface": "The object was on a DESK / TABLE / SHELF / COUNTER. Fill the positive_prompt with that surface's material (wood surface, granite countertop, marble top, lacquered shelf, etc.). Do NOT use wall or floor tags.",
            "corner":  "The object was in a CORNER. Fill with matching wall surface where the object leaned + floor junction.",
            "unknown": "Location is UNKNOWN. Infer from object name: large/tall object (wardrobe, cabinet, bookcase) → treat as WALL. Small decorative item → check if it sits on a surface. Choose the most logical surface type.",
        }
        guidance = _loc_guidance.get(loc, _loc_guidance["unknown"])
        bg_line  = f'\n  Explicit background hint from user: "{bg}" — use this exact description.' if bg else ""

        remove_hint = (
            f'\nREMOVE TASK CONTEXT:\n'
            f'  Extracted object to suppress: "{obj}"\n'
            f'  Inferred location: {loc.upper()}\n'
            f'  {guidance}{bg_line}\n'
            f'  Your negative_prompt MUST start with "{obj}" and 4–6 synonym/variant tags '
            f'BEFORE any artifact-avoidance tags.\n'
        )

    user_msg = (
        f'Interior inpainting instruction: "{raw_prompt}"\n\n'
        f"Detected task type: {task.upper()}\n"
        f"{remove_hint}\n"
        "TOKEN BUDGET REMINDER: Both positive_prompt AND negative_prompt MUST TARGET "
        "60–73 CLIP tokens each (roughly 45–55 comma-separated words/phrases). "
        "PROMPTS WITH FEWER THAN 50 TOKENS SEVERELY UNDERPERFORM — "
        "add more relevant descriptive tags to fill the full budget. "
        'Return ONLY a JSON object with keys "positive_prompt" and "negative_prompt".'
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_msg},
    ]
    text = M.qwen_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )

    # Move Qwen to GPU, run inference, move back to CPU
    M.qwen_model.to(M.T2_DEVICE)
    try:
        inputs = M.qwen_tokenizer(text, return_tensors="pt").to(M.T2_DEVICE)
        with torch.no_grad():
            out = M.qwen_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
            )
        raw_out = M.qwen_tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
    finally:
        M.qwen_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raw_out = re.sub(r"<\s*think\s*>.*?<\s*/\s*think\s*>", "", raw_out, flags=re.DOTALL).strip()
    raw_out = re.sub(r"<\s*think\s*>.*",                   "", raw_out, flags=re.DOTALL).strip()
    raw_out = re.sub(r"<[^>]+>",                           "", raw_out).strip()

    try:
        pos, neg = _parse_llm_json(raw_out)
    except Exception as exc:
        print(f"[Task2 Enhance] ⚠ JSON parse failed ({exc}) → rule-based fallback")
        return _rule_fallback(raw_prompt, task)

    pos_before, neg_before = _clip_tokens(pos), _clip_tokens(neg)
    pos = _truncate_to_clip(pos)
    neg = _truncate_to_clip(neg) if neg else _TASK_NEGATIVE_EXTRA.get(task) or _BASE_NEGATIVE
    pos_tok, neg_tok = _clip_tokens(pos), _clip_tokens(neg)

    if pos_before != pos_tok or neg_before != neg_tok:
        print(f"[Task2 Enhance] ⚠ CLIP safety truncated: "
              f"pos {pos_before}→{pos_tok} tok, neg {neg_before}→{neg_tok} tok")

    print(f"[Task2 Enhance] ✓ pos ({pos_tok} tok): {pos}")
    print(f"[Task2 Enhance] ✓ neg ({neg_tok} tok): {neg}")
    return pos, neg
