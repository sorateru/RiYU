#!/usr/bin/env python3
"""Create a rights/provenance manifest for an anonymized human dataset."""

from __future__ import annotations

import argparse

from mahjong_ai.data import create_human_dataset_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--rights-basis",
        required=True,
        choices=["user_owned", "explicit_permission", "open_license"],
    )
    parser.add_argument("--authorization-reference", required=True)
    parser.add_argument("--redistribution-allowed", action="store_true")
    parser.add_argument("--confirm-ml-rights", action="store_true")
    parser.add_argument("--confirm-anonymized", action="store_true")
    args = parser.parse_args()
    if not args.confirm_ml_rights:
        parser.error("--confirm-ml-rights is required")
    if not args.confirm_anonymized:
        parser.error("--confirm-anonymized is required")
    create_human_dataset_manifest(
        args.output,
        args.datasets,
        dataset_id=args.dataset_id,
        source=args.source,
        rights_basis=args.rights_basis,
        authorization_reference=args.authorization_reference,
        redistribution_allowed=args.redistribution_allowed,
    )
    print(f"wrote and verified {args.output}")


if __name__ == "__main__":
    main()
