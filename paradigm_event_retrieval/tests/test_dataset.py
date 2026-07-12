from pathlib import Path
from PIL import Image
from paradigm_event_retrieval.paradigm_clip.caption_templates import load_caption_templates
from paradigm_event_retrieval.paradigm_clip.dataset import scan_dataset, split_events, write_manifests


def test_scan_split_and_reproducible_manifests(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for event in ("event_001", "event_002", "event_003"):
        path = raw / event / "podiums" / "one.jpg"; path.parent.mkdir(parents=True); Image.new("RGB", (8, 8), "red").save(path)
    (raw / "event_001" / "podiums" / "bad.jpg").write_bytes(b"not-an-image")
    records, corrupt = scan_dataset(raw)
    assert len(records) == 3 and len(corrupt) == 1
    splits = split_events(records, seed=4)
    assert len(set(splits.values())) == 3
    templates_path = Path(__file__).parents[1] / "configs" / "caption_templates.yaml"
    first = write_manifests(records, splits, load_caption_templates(templates_path), tmp_path / "m1", seed=4)
    second = write_manifests(records, splits, load_caption_templates(templates_path), tmp_path / "m2", seed=4)
    assert first["train"].read_text() == second["train"].read_text()
