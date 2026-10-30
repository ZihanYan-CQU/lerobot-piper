
import sys
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def load_parquets(directory, columns=None):
    files = sorted(directory.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(directory)
    return pq.read_table(files, columns=columns).to_pydict()


def audit(root):
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())

    meta = load_parquets(
        root / "meta/episodes",
        ["episode_index", "length",
         "dataset_from_index", "dataset_to_index"]
    )

    data = load_parquets(
        root / "data",
        ["index", "episode_index", "frame_index",
         "timestamp", "action", "observation.state"]
    )

    idx = np.asarray(data["index"], dtype=np.int64)
    ep = np.asarray(data["episode_index"], dtype=np.int64)
    frame = np.asarray(data["frame_index"], dtype=np.int64)
    ts = np.asarray(data["timestamp"], dtype=float)

    action = np.asarray(data["action"], dtype=float)
    state = np.asarray(data["observation.state"], dtype=float)

    n = len(idx)
    declared = int(info["total_frames"])

    registered_eps = set(map(int, meta["episode_index"]))
    unregistered_mask = ~np.isin(ep, list(registered_eps))

    finite = (
        np.isfinite(action).all(axis=1)
        & np.isfinite(state).all(axis=1)
    )

    print(f"\n===== {root.name} =====")
    print("元数据总帧数:", declared)
    print("实际 Parquet 行数:", n)
    print("未登记的实际帧数:", int(unregistered_mask.sum()))
    print("非有限数值帧数:", int((~finite).sum()))
    print("重复全局 index 数:", n - len(np.unique(idx)))
    print("index != 物理行号数:",
          int(np.count_nonzero(idx != np.arange(n))))

    # 检查每个已登记 episode 在物理文件中是否完整
    good_frames = 0
    bad_episodes = []

    for i, e in enumerate(meta["episode_index"]):
        e = int(e)
        length = int(meta["length"][i])
        lo = int(meta["dataset_from_index"][i])
        hi = int(meta["dataset_to_index"][i])

        problems = []

        if hi - lo != length:
            problems.append("元数据长度不一致")

        if lo < 0 or hi > n or lo >= hi:
            problems.append("物理行范围错误")
        else:
            if not np.all(ep[lo:hi] == e):
                problems.append("episode 身份不匹配")

            if not np.array_equal(
                frame[lo:hi], np.arange(length)
            ):
                problems.append("frame_index 不连续")

            if not np.array_equal(
                idx[lo:hi], np.arange(lo, hi)
            ):
                problems.append("全局 index 与行位置错位")

            if np.any(np.diff(ts[lo:hi]) <= 0):
                problems.append("时间戳倒退或重复")

            if not np.all(finite[lo:hi]):
                problems.append("动作或状态含 NaN/Inf")

        if problems:
            bad_episodes.append((e, problems))
        else:
            good_frames += length

    print("\n--- 已登记 episode 结构核查 ---")
    print("结构检查通过的帧数:", good_frames)
    print("通过比例:",
          f"{good_frames / max(declared, 1):.2%}")
    print("异常 episode 数:", len(bad_episodes))

    for e, problems in bad_episodes[:20]:
        print("异常 episode:", e, problems)

    # 单独检查所有物理 episode，包括未登记的
    print("\n--- 实际采集轨迹检查 ---")
    for e in np.unique(ep):
        rows = np.flatnonzero(ep == e)
        f = frame[rows]
        t = ts[rows]

        frame_ok = np.array_equal(
            f, np.arange(len(rows))
        )
        time_ok = bool(np.all(np.diff(t) > 0))
        numeric_ok = bool(np.all(finite[rows]))

        if (
            e not in registered_eps
            or not frame_ok
            or not time_ok
            or not numeric_ok
        ):
            print({
                "episode": int(e),
                "帧数": len(rows),
                "已登记": bool(e in registered_eps),
                "帧连续": frame_ok,
                "时间递增": time_ok,
                "数值有限": numeric_ok,
            })

    # 按当前已知的全 episode 读取方式检查首动作配对
    # 这里只检测首个监督动作，尚未覆盖完整 action chunk
    in_range = (idx >= 0) & (idx < n)
    wrong = ~in_range

    valid_rows = np.flatnonzero(in_range)
    target_rows = idx[valid_rows]

    wrong[valid_rows] = (
        (ep[valid_rows] != ep[target_rows])
        | (frame[valid_rows] != frame[target_rows])
    )

    eligible = min(declared, n)
    print("\n--- 首动作监督配对检查 ---")
    print("已登记帧范围内疑似错配:",
          int(wrong[:eligible].sum()), "/", eligible)
    print("疑似错配比例:",
          f"{wrong[:eligible].mean():.2%}")

    # 比较物理上连续的动作和关节状态
    # 仅使用前六个机械臂关节，不包含夹爪
    if action.ndim == 2 and state.ndim == 2:
        dims = min(6, action.shape[1], state.shape[1])
        print("\n--- 动作与状态连续性 ---")

        for name, arr in [
            ("action", action[:, :dims]),
            ("state", state[:, :dims])
        ]:
            adjacent = (
                (ep[1:] == ep[:-1])
                & (frame[1:] == frame[:-1] + 1)
                & finite[1:] & finite[:-1]
            )
            delta = np.linalg.norm(
                np.diff(arr, axis=0)[adjacent], axis=1
            )
            if len(delta):
                print(name, "相邻帧变化 p50/p95/p99:",
                      np.quantile(delta, [.5, .95, .99]))

        # action_t 与未来 state_{t+lag} 的差异
        # 仅适用于绝对关节目标和关节状态单位一致
        for lag in [0, 1, 2, 3, 5, 10]:
            if lag == 0:
                ok = finite.copy()
                a = action[:, :dims]
                s = state[:, :dims]
            else:
                ok = (
                    (ep[:-lag] == ep[lag:])
                    & (frame[lag:] == frame[:-lag] + lag)
                    & finite[:-lag] & finite[lag:]
                )
                a = action[:-lag, :dims]
                s = state[lag:, :dims]

            if np.any(ok):
                error = np.linalg.norm(
                    a[ok] - s[ok], axis=1
                )
                print(
                    f"lag={lag}:",
                    f"median={np.median(error):.5f}",
                    f"p95={np.quantile(error, .95):.5f}"
                )


if __name__ == "__main__":
    for dataset_root in sys.argv[1:]:
        audit(dataset_root)
