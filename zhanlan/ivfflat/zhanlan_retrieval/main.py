# zhanlan/ivfflat/zhanlan_retrieval/main.py
import multiprocessing as mp

from .config import RuntimeConfig
from .pipeline import run_pipeline


def main():
    # keep behavior consistent with your original script
    mp.set_start_method("spawn", force=True)

    cfg = RuntimeConfig()
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
