#!/usr/bin/env python3
"""Revalidate and session-split an explicit reviewed manifest into a new file.

No default writes or migration of heuristic legacy labels. Fixed legacy splits
remain untouched. Reviewed spans still require a compatible feature extractor.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
try:
    from . import active_learning as review
except ImportError:
    import active_learning as review


def hsplit(key: str) -> str:
    return review.session_split(key)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audio-root", default=str(review.AUDIO_ROOT))
    args = parser.parse_args(argv)
    try:
        parent_bytes = Path(args.manifest).read_bytes()
        source = json.loads(parent_bytes)
        if not isinstance(source, dict) or source.get("schema") != review.MANIFEST_SCHEMA:
            raise ValueError("only reviewed manifests are accepted; legacy labels need human review")
        rows = [item for section in review.SECTIONS for item in source.get(section, [])]
        manifest = review.build_reviewed_manifest({"schema": review.STORE_SCHEMA, "reviews": rows}, args.audio_root)
        # Derived rows are not the original store: normalization and exclusion
        # of unsure reviews change their identity. Preserve the provenance chain.
        manifest["derived_reviews_sha256"] = manifest.pop("review_store_sha256")
        manifest["review_store_sha256"] = source.get("review_store_sha256")
        manifest["parent_manifest_sha256"] = hashlib.sha256(parent_bytes).hexdigest()
        review.write_new_manifest(args.output, manifest)
        print(json.dumps(manifest["readiness"], allow_nan=False))
        return 0
    except (ValueError, OSError) as exc:
        parser.exit(2, f"review preparation refused: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
