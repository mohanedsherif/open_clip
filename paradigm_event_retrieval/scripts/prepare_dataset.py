from __future__ import annotations
import argparse
from pathlib import Path
from paradigm_event_retrieval.paradigm_clip.caption_templates import load_caption_templates
from paradigm_event_retrieval.paradigm_clip.dataset import dataset_report, find_duplicates, save_json, scan_dataset, split_events, write_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate, deduplicate, and event-split Paradigm images.")
    parser.add_argument("--raw-dir", required=True); parser.add_argument("--manifests-dir", required=True); parser.add_argument("--templates", required=True)
    parser.add_argument("--report", required=True); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--near-duplicate-threshold", type=int, default=6)
    args = parser.parse_args()
    records, corrupt = scan_dataset(args.raw_dir)
    duplicates = find_duplicates(args.raw_dir, records, args.near_duplicate_threshold)
    splits = split_events(records, args.seed)
    write_manifests(records, splits, load_caption_templates(args.templates), args.manifests_dir, args.seed)
    save_json(dataset_report(records, corrupt, duplicates), args.report)

if __name__ == "__main__": main()
