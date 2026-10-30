"""保留训练步数为 5000 倍数的 checkpoint，默认只预览。"""

import argparse
import shutil
from pathlib import Path


DEFAULT_CHECKPOINTS = Path(
    "/home/yzh/具身智能/（Lerobot）微调模型/pi0.5_shapeMatching_b32_s20000/checkpoints"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS, help="checkpoints 文件夹路径"
    )
    parser.add_argument("--delete", action="store_true", help="实际删除；不加此参数时只预览")
    args = parser.parse_args()

    if not args.checkpoints.is_dir():
        parser.error(f"文件夹不存在或不是目录: {args.checkpoints}")

    count = 0
    for folder in sorted(args.checkpoints.iterdir()):
        # 只处理直接子目录；跳过符号链接、普通文件和非数字名称。
        if folder.is_symlink() or not folder.is_dir():
            continue
        if not (folder.name.isascii() and folder.name.isdecimal()):
            continue

        # int("001000") == 1000，前导零不影响判断。
        if int(folder.name) % 5000 == 0:
            print(f"保留: {folder.name}")
            continue

        if args.delete:
            shutil.rmtree(folder)
            print(f"已删除: {folder}")
        else:
            print(f"[预览] 将删除: {folder}")
        count += 1

    if args.delete:
        print(f"完成，共删除 {count} 个模型文件夹。")
    else:
        print(f"预览完成，共 {count} 个模型文件夹待删除。添加 --delete 才会实际删除。")


if __name__ == "__main__":
    main()
