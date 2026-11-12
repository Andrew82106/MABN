import sys
from pathlib import Path
R5=Path(__file__).resolve().parents[1]
PRE=R5.parent
R4=PRE/'round4'
sys.path.append(str(R4/'src'))
from common4 import readl,writel,save,sha

DEPENDENCIES=['results/checkpoints/base.pkl','results/checkpoints/unit_models.pkl','data/unit_inputs.pkl','confirmation/protocol.json','confirmation/results/predictions.jsonl']

