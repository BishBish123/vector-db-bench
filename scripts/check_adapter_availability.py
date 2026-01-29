"""Post-install probe: print which adapters' optional deps imported cleanly.

The `embed`, `chroma`, and `lance` extras carry platform markers
(`sys_platform != 'darwin' or platform_machine != 'x86_64'` etc.) so a
plain `uv sync --extra embed --extra chroma --extra lance` on Intel
macOS resolves to a no-op for those extras and exits 0. The first-time
operator on Intel sees `make install` succeed, runs `make bench-demo`,
and gets a four-row Pareto plot that's actually a two-row plot — with
no signal that pgvector + qdrant are the only adapters that came up.

This script prints a one-line summary per optional dep so the install
output makes the platform gating obvious. Exit code is always 0; this
is informational, not gating — `make install-min` is the documented
escape hatch.
"""

from __future__ import annotations

import importlib
import platform
import sys

# Each row: (display label, importable module, extras the module belongs to,
# explanation when it's missing).
_DEPS: tuple[tuple[str, str, str, str], ...] = (
    (
        "sentence-transformers (embed)",
        "sentence_transformers",
        "embed",
        "torch / sentence-transformers ship no macOS x86_64 wheels",
    ),
    (
        "chromadb (chroma)",
        "chromadb",
        "chroma",
        "chromadb depends on onnxruntime, which has no macOS x86_64 wheel",
    ),
    (
        "lancedb (lance)",
        "lancedb",
        "lance",
        "lancedb publishes no macOS x86_64 wheel",
    ),
)


def _check(module: str) -> bool:
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True


def main() -> int:
    plat = f"{platform.system().lower()} {platform.machine()}"
    print(f"[install] adapter availability check ({plat}, python {sys.version_info.major}.{sys.version_info.minor}):")

    available: list[str] = []
    missing: list[tuple[str, str]] = []
    for label, module, _extra, why in _DEPS:
        if _check(module):
            available.append(label)
            print(f"  OK   {label}")
        else:
            missing.append((label, why))
            print(f"  SKIP {label} -- {why}")

    if missing:
        print()
        print("[install] some adapter extras did not install on this platform.")
        print("[install] this is expected on Intel macOS; the harness still runs")
        print("[install] pgvector + qdrant. For a guaranteed-empty install (no")
        print("[install] platform-gated extras), use `make install-min` instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
