import subprocess


def select_free_gpu(verbose: bool = True) -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15,
    )
    rows = [line.strip().split(",") for line in result.stdout.strip().splitlines()]
    util_pct = [int(r[1].strip()) for r in rows]
    mem_free  = [int(r[2].strip()) for r in rows]  # MiB
    best = min(range(len(rows)), key=lambda i: (util_pct[i], -mem_free[i]))
    if verbose:
        for i, (u, f) in enumerate(zip(util_pct, mem_free)):
            mark = "  <-- selected" if i == best else ""
            print(f"[gpu_select] GPU {i}: util={u:3d}%  free={f/1024:.1f} GB{mark}")
    return best


def set_free_gpu(verbose: bool = True):
    import torch
    idx = select_free_gpu(verbose=verbose)
    device = torch.device(f"cuda:{idx}")
    torch.cuda.set_device(device)
    return device