import re
from typing import Tuple

DISALLOWED_PATTERNS = [
    r"(?i)password\s*=",
    r"(?i)api[_-]?key",
    r"(?i)token",
    r"(?i)ssh[- ]?key",
]

PROMPT_INJECTION_SIGNS = [
    r"(?i)ignore previous instructions",
    r"(?i)disable safety",
    r"(?i)reveal system prompt",
]

SAFE_MAX_TOKENS = 2000

def basic_content_filter(text: str) -> Tuple[bool, str]:
    for pattern in DISALLOWED_PATTERNS:
        if re.search(pattern, text):
            return False, "Blocked: sensitive secret pattern detected"
    return True, text

def detect_prompt_injection(user_text: str) -> bool:
    for pattern in PROMPT_INJECTION_SIGNS:
        if re.search(pattern, user_text):
            return True
    return False

def enforce_token_limit(text: str) -> str:
    tokens = text.split()
    if len(tokens) > SAFE_MAX_TOKENS:
        kept = []
        count = 0
        for token in tokens:
            if count < SAFE_MAX_TOKENS:
                kept.append(token)
                count += 1
        return " ".join(kept)
    return text
