#!/usr/bin/env python3
"""Generate a seed-tagged reg_cfg JSON from the register CSV."""

import argparse

from csiir_common import generate_reg_config, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ISP-CSIIR reg_cfg_seedX.json")
    parser.add_argument(
        "--csv",
        default="config/register_table.csv",
        help="register CSV path",
    )
    parser.add_argument("--seed", type=int, required=True, help="seed number used in output filename")
    parser.add_argument(
        "--output",
        default=None,
        help="optional explicit output path",
    )
    args = parser.parse_args()

    payload = generate_reg_config(args.csv)
    output_path = args.output or f"config/reg_cfg_seed{args.seed}.json"
    path = save_json(output_path, payload)
    print(path)


if __name__ == "__main__":
    main()
