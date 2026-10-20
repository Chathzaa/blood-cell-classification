"""Verify recorded experiments and the single current conference paper."""
from verify_evidence_release import main as verify_experiments
from finalize_conference_release import verify_paper
if __name__ == '__main__':
    verify_experiments()
    print(verify_paper())
