#!/usr/bin/env python3
"""Generate input patterns for ISP-CSIIR verification."""

import argparse

from csiir_common import (
    PATTERN_GENERATORS,
    generate_input_pattern,
    load_img_info,
    save_hex_image,
)


def main():
    parser = argparse.ArgumentParser(description="Generate ISP-CSIIR test patterns")
    parser.add_argument("--img-info", type=str, help="img_info JSON")
    parser.add_argument("--config", type=str, help="legacy JSON config")
    parser.add_argument("--pattern", default="random", choices=list(PATTERN_GENERATORS.keys()))
    parser.add_argument('--width', type=int, default=16)
    parser.add_argument('--height', type=int, default=16)
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--output', type=str, default=None, help='输出 hex 文件')
    args = parser.parse_args()

    if args.img_info:
        img_info = load_img_info(args.img_info)
        width = int(img_info["image"]["width"])
        height = int(img_info["image"]["height"])
        pattern = img_info["image"]["pattern"]
        seed = int(img_info.get("seed", 0))
        output = args.output or img_info["image"]["input_path"]
        img = generate_input_pattern(img_info)
    elif args.config:
        img_info = load_img_info(args.config)
        width = int(img_info["image"]["width"])
        height = int(img_info["image"]["height"])
        pattern = img_info["image"]["pattern"]
        seed = int(img_info.get("seed", 0))
        output = args.output or img_info["image"]["input_path"]
        img = generate_input_pattern(img_info)
    else:
        width = args.width
        height = args.height
        pattern = args.pattern
        seed = args.seed
        output = args.output
        if output is None:
            raise SystemExit("--output is required when --img-info is not used")
        img = PATTERN_GENERATORS[pattern](height, width, seed=seed)

    save_hex_image(output, img)

    print(f"Pattern: {pattern} ({width}x{height})")
    if seed is not None:
        print(f"Seed: {seed}")
    print(f"Input range: [{img.min()}, {img.max()}]")
    print(f"Output: {output}")

if __name__ == '__main__':
    main()
