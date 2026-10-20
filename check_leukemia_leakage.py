"""Verify the grouped ALL manifest without confusing image groups with patients."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
ROOT=Path(__file__).resolve().parent
def main():
    with (ROOT/'data/ALL_verified/manifest.csv').open() as f:rows=list(csv.DictReader(f))
    for key in ('source_id','pixel_sha256','group'):
        splits=defaultdict(set)
        for r in rows:splits[r[key]].add(r['split'])
        assert all(len(s)==1 for s in splits.values()),f'{key} crosses partitions'
    for r in rows:
        assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['file_sha256']
    print(f'Passed: {len(rows)} source images; exact identities and observable image groups do not cross partitions.')
    print('Patient independence remains unknown because patient identifiers are unavailable.')
if __name__=='__main__':main()
