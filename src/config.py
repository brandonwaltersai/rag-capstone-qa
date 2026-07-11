"""Configuration and lightweight lane router for the RAG service-center assistant."""
from pathlib import Path
import tiktoken

# --- Models ---
EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL = "gpt-4.1"

# --- Scope ---
ALLOWED_LANES = {"Account Access", "Orders/Refunds"}
ORDERS_CATS_KB = {"ORDER", "REFUND", "DELIVERY", "SHIPPING", "CANCEL"}

TOPK_DEFAULT = 5
ESCALATION_THRESHOLD = 0.25

ESCALATE_PREFIX = "ESCALATE:"
CITATION_PREFIX = "Citations:"

# --- Safety triggers: never let the model handle these directly, always escalate ---
SAFETY_TRIGGERS = [
    "credit card", "card number", "cvv", "cvc", "security code", "pin", "otp", "one-time code", "verification code",
    "social security", "ssn", "bank account", "routing number", "wire transfer", "chargeback",
    "driver's license", "passport", "government id", "full ssn", "date of birth", "dob",
]

LANE_KEYWORDS = {
    "Account Access": [
        "login", "log in", "sign in", "password reset", "reset password", "locked out", "unlock", "2fa", "mfa",
        "verification", "account access", "username", "email change",
    ],
    "Orders/Refunds": [
        "order", "refund", "return", "exchange", "cancel", "cancellation", "shipping", "delivery", "tracking",
        "late", "damaged", "missing item",
    ],
}


def route_lane(query: str) -> str | None:
    """Keyword-based lane router. Returns a lane, or None if ambiguous/out of scope."""
    q = query.lower()
    scores = {lane: sum(1 for k in kws if k in q) for lane, kws in LANE_KEYWORDS.items()}
    if max(scores.values()) == 0:
        return None
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


_enc = tiktoken.get_encoding("cl100k_base")


def n_tokens(s: str) -> int:
    return len(_enc.encode(s))


# --- Local project paths (replaces the notebook's /content layout) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
INDEX_DIR = DATA_DIR / "index"
LOG_DIR = PROJECT_ROOT / "eval" / "logs"

for d in (RAW_DIR, CACHE_DIR, INDEX_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

KB_CLEAN_PATH = DATA_DIR / "kb_clean.csv"
EVAL_PROMPTS_PATH = DATA_DIR / "eval_prompts.csv"
EMB_CACHE = CACHE_DIR / "kb_vectors.npy"
META_CACHE = CACHE_DIR / "kb_vectors_meta.json"
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index.bin"
FAISS_META_PATH = INDEX_DIR / "faiss_index_meta.json"
