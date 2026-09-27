import sys
from pathlib import Path

# Make the algorithm directory importable no matter where pytest runs from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
