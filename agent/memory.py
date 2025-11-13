import json

from django.core.cache import cache


MEMORY_TTL = 60 * 30  # 30 minutes memory will be cached


def load_memory(user_id):
    raw = cache.get(f"user:{user_id}:memory")
    if raw:
        return json.loads(raw)
    return []   # list of {"role": "user"/"assistant", "content": "..."}


def save_memory(user_id, messages):
    cache.set(
        f"user:{user_id}:memory",
        json.dumps(messages[-10:]),   # keep last 10
        MEMORY_TTL
    )
