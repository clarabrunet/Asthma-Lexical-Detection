"""Run the full E1 lexical pipeline with stratified k-fold cross-validation.

Loads the clinical corpus from ``data/dataset`` (or ``--dataset``), lemmatises
it, then evaluates the rule-based detector with patient-level CV.

Usage
-----
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --dataset PATH --top_n 5 --n_splits 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asthma_e1.config import CONFIG, PATHS  # noqa: E402
from asthma_e1.data_loading import load_dataset_records  # noqa: E402
from asthma_e1.evaluation import run_cv  # noqa: E402
from asthma_e1.preprocessing import (  # noqa: E402
    build_corpus_map,
    build_spacy_docs,
    load_spacy_model,
    normalize_doc,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PATHS.dataset_dir,
                        help="Dataset root with asma/ and no_asma/ subfolders")
    parser.add_argument("--top_n", type=int, default=CONFIG.top_n_terms)
    parser.add_argument("--n_splits", type=int, default=CONFIG.cv_n_splits)
    args = parser.parse_args()

    t0 = time.time()

    print(f"Loading dataset: {args.dataset}")
    records = load_dataset_records(args.dataset, CONFIG.exclude_patients)
    n_pat = len({r.patient_id for r in records})
    n_pos = len({r.patient_id for r in records if r.label == 1})
    print(f"  {len(records)} documents | {n_pat} patients "
          f"(asthma={n_pos}, non-asthma={n_pat - n_pos})")

    print(f"\nLoading spaCy model ({CONFIG.spacy_model})...")
    nlp = load_spacy_model()
    print("Lemmatising documents...")
    docs_spacy = build_spacy_docs(records, nlp)
    norm_docs = [normalize_doc(d) for d in docs_spacy]
    corpus_map = build_corpus_map(docs_spacy)

    print(f"\n{args.n_splits}-fold CV | top_n={args.top_n}")
    metrics, _ = run_cv(
        records, docs_spacy, norm_docs, corpus_map,
        top_n=args.top_n, n_splits=args.n_splits,
    )

    print("\n" + "-" * 45)
    print(f"  TP={metrics['tp']:.0f}  FP={metrics['fp']:.0f}  "
          f"FN={metrics['fn']:.0f}  TN={metrics['tn']:.0f}")
    print(f"  Precision  : {metrics['precision']:.3f}")
    print(f"  Recall     : {metrics['recall']:.3f}")
    print(f"  F1         : {metrics['f1']:.3f}")
    print(f"  Specificity: {metrics['specificity']:.3f}")
    print(f"  Accuracy   : {metrics['accuracy']:.3f}")
    print("-" * 45)
    print(f"  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
