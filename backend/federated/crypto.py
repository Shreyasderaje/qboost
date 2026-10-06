"""Envelope encryption for everything that crosses the wire.

Every payload a clinic sends is serialised to JSON and sealed with a
per-clinic Fernet key (AES-128-CBC + HMAC-SHA256, issued at registration by
the coordinator's key service).  This stands in for the mutual-TLS channel
of a production deployment: the coordinator terminates the envelope on
arrival, exactly as a load balancer terminates TLS, and nothing is ever
accepted from an unregistered clinic.
"""

from __future__ import annotations

import json

from cryptography.fernet import Fernet, InvalidToken


def issue_clinic_key() -> str:
    """Create a new Fernet key for a registering clinic."""
    return Fernet.generate_key().decode("ascii")


def seal(payload: dict, key: str) -> str:
    """Encrypt a JSON-serialisable payload for transport."""
    f = Fernet(key.encode("ascii"))
    blob = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return f.encrypt(blob).decode("ascii")


def unseal(token: str, key: str) -> dict:
    """Decrypt and parse a sealed payload; raises ValueError on tampering."""
    f = Fernet(key.encode("ascii"))
    try:
        blob = f.decrypt(token.encode("ascii"))
    except InvalidToken as exc:
        raise ValueError("sealed payload failed authentication (wrong key or tampered)") from exc
    return json.loads(blob.decode("utf-8"))
