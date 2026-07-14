import time
from pathlib import Path

from models import TestResult
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


print(ROOT / "app" / "backend" / "graph.py")