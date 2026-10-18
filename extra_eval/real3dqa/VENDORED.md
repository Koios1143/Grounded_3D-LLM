Vendored copy of the two standalone evaluators from https://github.com/xianzhengma/Real-3DQA
(CC BY-SA 4.0), unmodified. `extra_eval/score.py` imports `clean_answer` / `answer_match`
from `evaluate_debiased_exact_match.py` for SQA3D EM / EM-R as well (the same LEO-style
normalisation + refined match used by Chat-Scene and LEO), and runs both scripts on the
Real-3DQA prediction files.
