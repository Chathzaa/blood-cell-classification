"""Train the source-aligned WBC classifier. Existing experiments are preserved."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.train_verified_wbc import main

if __name__ == '__main__':
    if '--view' not in sys.argv:
        sys.argv.extend(['--view','crop'])
    main()
