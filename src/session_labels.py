"""Session-scoped pseudonymous labels for display and logs (not employee IDs)."""

import hashlib


def worker_label(person_id: int) -> str:
    """
    Return a stable opaque label for this session track.
    Internal counting still uses numeric person_id; only user-facing text uses this.
    """
    token = hashlib.sha256(f"mtu-session:{person_id}".encode()).hexdigest()[:6].upper()
    return f"Worker-{token}"
