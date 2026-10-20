"""Build the grouped ALL screening dataset; preserve completed studies."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.prepare_all_study import main
if __name__=='__main__':main()
