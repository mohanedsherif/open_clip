from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
from paradigm_event_retrieval.paradigm_clip.config import load_yaml
from paradigm_event_retrieval.paradigm_clip.inference import ParadigmEmbeddingModel

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--manifest", required=True); parser.add_argument("--raw-dir", required=True); parser.add_argument("--output", required=True); args = parser.parse_args()
    config = load_yaml(args.config); rows = list(csv.DictReader(Path(args.manifest).open(encoding="utf-8"))); paths = [str(Path(args.raw_dir) / row["filepath"]) for row in rows]
    started = time.perf_counter(); model = ParadigmEmbeddingModel(config["model_name"], config["pretrained"], device="cpu")
    embeddings = model.embed_images(paths, int(config.get("batch_size", 32))); queries = ["a conference podium", "a branded event backdrop", "an exhibition booth"]
    results = [{"query": q, "results": model.retrieve(q, embeddings, top_k=5)} for q in queries]
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / "smoke_results.json").write_text(json.dumps({"warning": "SMOKE TEST ONLY — NOT A REAL-WORLD MODEL QUALITY BENCHMARK", "model": model.metadata, "image_count": len(paths), "queries": results, "seconds": time.perf_counter()-started}, indent=2), encoding="utf-8")
    print(f"SMOKE TEST ONLY — NOT A REAL-WORLD MODEL QUALITY BENCHMARK\nSaved: {output / 'smoke_results.json'}")
if __name__ == "__main__": main()
