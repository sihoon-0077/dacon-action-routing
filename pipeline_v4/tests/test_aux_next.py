import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline_v4.common.constants import LABEL2ID  # noqa: E402
from pipeline_v4.train_fold import (  # noqa: E402
    EncodedDataset,
    NEXT_IGNORE_INDEX,
    build_next_labels,
    load_config,
    next_label_coverage,
    step_of,
)


def tiny_samples():
    return [
        {"id": "sess_a-step_01", "action": "read_file"},
        {"id": "sess_b-step_01", "action": "run_bash"},
        {"id": "sess_a-step_03", "action": "edit_file"},
        {"id": "sess_a-step_02", "action": "grep_search"},
        {"id": "sess_b-step_02", "action": "run_tests"},
    ]


def main():
    samples = tiny_samples()
    y_next = build_next_labels(samples)
    expected = [
        LABEL2ID["grep_search"],
        LABEL2ID["run_tests"],
        NEXT_IGNORE_INDEX,
        LABEL2ID["edit_file"],
        NEXT_IGNORE_INDEX,
    ]
    assert y_next.tolist() == expected, y_next.tolist()
    assert abs(next_label_coverage(y_next) - 0.6) < 1e-9
    assert step_of("sess_x-step_07") == 7
    assert step_of("sess_x") == -1

    encodings = {
        "input_ids": torch.tensor([[1, 2, 0], [3, 4, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.long),
    }
    y = torch.tensor([LABEL2ID["read_file"], LABEL2ID["run_bash"]], dtype=torch.long)
    dataset = EncodedDataset(encodings, y.numpy(), y_next[:2])
    row = dataset[0]
    assert row["y_next"].item() == LABEL2ID["grep_search"]
    assert row["y_coarse"].item() == 0

    cfg = load_config("pipeline_v4/configs/mdeberta_v2_2_aux_next_384.yaml")
    assert cfg["serializer"] == "v2_2"
    assert float(cfg["aux_next_weight"]) == 0.2
    print("aux_next ok")


if __name__ == "__main__":
    main()
