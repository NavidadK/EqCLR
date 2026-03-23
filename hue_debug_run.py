import json

import torch

from hue_debug_utils import (
    run_pathmnist_hue_validation,
    run_rotation_hue_combined_demo,
    run_synthetic_hue_cycle_validation,
)


def _to_jsonable(x):
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    return x


def main():
    print("Running synthetic validation: N=1, K=4")
    m1 = run_synthetic_hue_cycle_validation(N=1, K=4, show_plot=True, strict_assertions=False)
    print(json.dumps(_to_jsonable(m1), indent=2))

    print("Running synthetic validation: N=1, K=50")
    m2 = run_synthetic_hue_cycle_validation(N=1, K=50, show_plot=True, strict_assertions=False)
    print(json.dumps(_to_jsonable(m2), indent=2))

    try:
        from medmnist import PathMNIST
        import torchvision.transforms as transforms

        ds = PathMNIST(
            split="train",
            download=False,
            size=28,
            root="data/pathmnist/",
            transform=transforms.ToTensor(),
        )

        print("Running PathMNIST validation: N=1, K=50")
        m3 = run_pathmnist_hue_validation(
            ds,
            sample_idx=0,
            N=1,
            K=50,
            hue_steps=(0, 1, 2, 3),
            show_plot=True,
            strict_assertions=False,
        )
        print(json.dumps(_to_jsonable(m3), indent=2))

        print("Running combined rotation+hue demo: N=4, K=4")
        rgb = ds[0][0].float()
        m4 = run_rotation_hue_combined_demo(rgb, N=4, K=4, hue_step=1, rot_step=1, show_plot=True)
        print(json.dumps(_to_jsonable(m4), indent=2))

    except Exception as exc:
        print(f"Skipped PathMNIST/combined demo: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()

