import sys
from pathlib import Path

# srcディレクトリをsys.pathに追加
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
