#!/usr/bin/env python3
"""List uncertain candidates by default; export explicit human reviews only.

Legacy manifest.json and fixed splits are never overwritten. This prepares
review data; it does not run the legacy feature extractor or trainers.
"""
from __future__ import annotations
import argparse
import json
try:
    from . import active_learning as review
except ImportError:
    import active_learning as review


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", default=str(review.TELEMETRY))
    parser.add_argument("--audio-root", default=str(review.AUDIO_ROOT))
    parser.add_argument("--project-root", default=str(review.PROJECT_ROOT))
    parser.add_argument("--labels", help="Explicit human review store; required for export")
    parser.add_argument("--output", help="New reviewed manifest, never a legacy/default output")
    args = parser.parse_args(argv)
    try:
        if not args.labels and not args.output:
            review.print_candidates(args.telemetry, args.audio_root, args.project_root)
            return 0
        if not args.labels or not args.output:
            raise ValueError("export requires both --labels and --output")
        manifest = review.build_reviewed_manifest(review.load_store(args.labels), args.audio_root)
        review.write_new_manifest(args.output, manifest)
        print(json.dumps(manifest["readiness"], allow_nan=False))
        return 0
    except (ValueError, OSError) as exc:
        parser.exit(2, f"review preparation refused: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
