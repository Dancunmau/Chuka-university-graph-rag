"""
ingest_all.py — Orchestrator for the Chuka University GraphRAG ingestion pipeline.

STEP ORDER (dependency-safe):
  Step 1: ingest_programs     — Faculty + Department + ALL Program nodes
  Step 2: ingest_units        — CourseUnit nodes + links active Programs
  Step 3: ingest_papers       — PastPaper nodes + HAS_PAST_PAPER links
  Step 4: ingest_timetable    — TimetableSlot + Room + HAS_TIMESLOT
  Step 5: ingest_communities  — RepositoryItem + HAS_RESOURCE + FAISS

Usage:
    python src/ingest_all.py               # full pipeline, LIVE
    python src/ingest_all.py --dry-run     # preview counts, no writes
    python src/ingest_all.py --steps 1 2   # run only steps 1 and 2
    python src/ingest_all.py --wipe        # wipe DB then run full pipeline
"""

import sys
import time
import argparse
from neo4j_utils import get_driver, get_project_root, close_driver

sys.path.insert(0, str(get_project_root() / "src"))

import setup_schema
import ingest_programs
import ingest_units
import ingest_papers
import ingest_timetable
import ingest_communities

STEPS = {
    0: (setup_schema.main,       "Setup Graph Schema (Constraints)"),
    1: (ingest_programs.main,    "Faculty + Department + Program nodes"),
    2: (ingest_units.main,       "CourseUnit + Program links (units_ingested=True)"),
    3: (ingest_papers.main,      "PastPaper + HAS_PAST_PAPER"),
    4: (ingest_timetable.main,   "TimetableSlot + Room"),
    5: (ingest_communities.main, "RepositoryItem + FAISS"),
}


def wipe_database(driver):
    print("Wiping all nodes and relationships from the database...")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("Database wiped.\n")


def node_counts(driver) -> dict:
    with driver.session() as session:
        rows = session.run(
            "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC"
        ).data()
    return {r["label"]: r["cnt"] for r in rows if r["label"]}


def print_counts(title: str, counts: dict):
    print(f"\n{'-'*50}")
    print(f"  {title}")
    print(f"{'-'*50}")
    if not counts:
        print("  (empty)")
    for label, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {label:<25} : {cnt:,}")


def main():
    parser = argparse.ArgumentParser(description="Chuka GraphRAG — Ingestion Pipeline")
    parser.add_argument("--dry-run",  action="store_true", help="Preview without writing")
    parser.add_argument("--wipe",     action="store_true", help="Wipe DB before ingesting")
    parser.add_argument("--steps", nargs="+", type=int, metavar="N",
                        help="Run only specific steps e.g. --steps 1 3")
    args = parser.parse_args()

    steps_to_run = sorted(args.steps) if args.steps else list(STEPS.keys())

    print("\n" + "="*60)
    print("  Chuka University GraphRAG — Ingestion Pipeline")
    print("="*60)
    print(f"  Mode  : {'DRY RUN (no writes)' if args.dry_run else 'LIVE'}")
    print(f"  Steps : {steps_to_run}")
    print("="*60)

    driver = get_driver()

    if args.wipe and not args.dry_run:
        wipe_database(driver)

    before = node_counts(driver)
    print_counts("BEFORE ingestion", before)
    print()
    close_driver(driver)

    results = {}
    for step_num in steps_to_run:
        if step_num not in STEPS:
            print(f"  Unknown step {step_num}, skipping.")
            continue
        fn, desc = STEPS[step_num]
        print(f"\n{'='*60}")
        print(f"  STEP {step_num}: {desc}")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            fn(dry_run=args.dry_run)
            elapsed = time.time() - t0
            results[step_num] = ("OK", elapsed)
            print(f"\n  Completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            results[step_num] = ("FAIL", elapsed)
            print(f"\n  ERROR in step {step_num}: {e}")
            import traceback; traceback.print_exc()

    if not args.dry_run:
        driver = get_driver()
        after = node_counts(driver)
        close_driver(driver)
        print_counts("AFTER ingestion", after)

        print(f"\n{'-'*50}")
        print("  Changes")
        print(f"{'-'*50}")
        all_labels = set(before) | set(after)
        for lbl in sorted(all_labels):
            b = before.get(lbl, 0)
            a = after.get(lbl, 0)
            diff = a - b
            if diff != 0:
                sign = "+" if diff > 0 else ""
                print(f"  {lbl:<25} : {b:,} -> {a:,}  ({sign}{diff:,})")

    print(f"\n{'='*60}")
    print("  Pipeline Summary")
    print(f"{'='*60}")
    all_ok = True
    for step_num, (status, elapsed) in results.items():
        _, desc = STEPS[step_num]
        icon = "OK" if status == "OK" else "FAIL"
        print(f"  Step {step_num} [{icon}] {desc} ({elapsed:.1f}s)")
        if status != "OK":
            all_ok = False

    print(f"\n  {'All steps completed successfully!' if all_ok else 'Some steps failed. Check output above.'}")


if __name__ == "__main__":
    main()
