"""Small, explicit set of source, models and evidence for the current release."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def release_files():
    files = {
        ROOT / name for name in (
            ".gitignore", ".gitattributes", "README.md", "FINAL_DELIVERY.md",
            "AUDIT_AND_REVISION_REPORT.md", "NOVEL_METHOD_FINDINGS.md",
            "requirements.txt", "requirements-dev.txt", "app.py",
            "check_leakage.py", "check_leukemia_leakage.py",
            "yolov8n-cls.pt", "paper/README.md", "paper/REVIEW_RESPONSE.md",
        )
    }
    for folder in ("src", "tools", "tests"):
        files.update(p for p in (ROOT / folder).glob("*.py")
                     if p.name not in {"package_release.py", "package_all_release.py", "package_evidence_release.py", "complete_study.py", "fetch_reference_metadata.py"})
    files.update((ROOT / "config").glob("*.json"))
    files.update((ROOT / "paper/final_conference").glob("*.tex"))
    files.update(ROOT / "paper/final_conference" / name
                 for name in ("main.pdf", "BUILD.md", "make_assets.py"))
    files.update((ROOT / "paper/final_conference/figures").glob("*.pdf"))
    for folder in ("data/ALL_verified", "data/WBC_verified"):
        files.update((ROOT / folder).glob("*.csv"))
        files.update((ROOT / folder).glob("*.json"))
    for folder in ("outputs/all_verified", "outputs/all_novel", "outputs/verified"):
        files.update((ROOT / folder).glob("*.json"))
    for folder in ("runs/all_verified", "runs/all_novel"):
        files.update((ROOT / folder).glob("*.json"))
        files.update((ROOT / folder).glob("*.joblib"))
    registry = json.loads((ROOT / "config/model_registry.json").read_text())
    files.update(ROOT / entry["path"] for entry in registry.values())
    for run in (ROOT / "runs/verified").iterdir():
        if run.is_dir():
            files.update(run / name for name in (
                "experiment.json", "args.yaml", "results.csv", "weights/last.pt"
            ))
    for name in (
        "EXPERIMENT_PROTOCOL.md", "LEUKEMIA_PROTOCOL.md", "NOVEL_METHOD_PROTOCOL.md",
        "NOVELTY_LITERATURE_REVIEW.md", "data_audit.json",
        "all_group_audit.json", "bccd_scene_audit.json",
        "cross_dataset_source_alignment.json", "near_duplicate_candidates.json",
        "wbc_augmentation_provenance.csv", "wbc_crossed_source_groups.json",
        "evidence_cache_provenance_migration.json", "evidence_ui_checks.json",
        "all_release_verification.json", "evidence_release_verification.json",
        "final_conference_verification.json", "rbc_reference_feasibility.json",
    ):
        files.add(ROOT / "audit" / name)
    missing = sorted(str(p.relative_to(ROOT)) for p in files if not p.is_file())
    if missing:
        raise FileNotFoundError("Release inputs missing: " + ", ".join(missing))
    return sorted(files)


if __name__ == "__main__":
    paths = release_files()
    print(json.dumps({
        "files": [p.relative_to(ROOT).as_posix() for p in paths],
        "count": len(paths), "bytes": sum(p.stat().st_size for p in paths),
    }, indent=2))
