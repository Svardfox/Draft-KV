"""Exercise orchestration without starting models or touching GPU workers."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PIPELINE = Path(__file__).resolve().parents[1] / "script/draft_kv/run_stage3_train_and_eval.sh"


class PipelineTest(unittest.TestCase):
    def run_pipeline(self, *, fail=False, missing=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_bash = root / "bash"
            fake_bash.write_text(
                "#!/bin/bash\n"
                'if [[ "$1" == *run_stage3_training.sh ]]; then\n'
                '  echo "TRAIN gpu=$DRAFT_KV_GPU_ID"\n'
                '  [[ "${FAIL_TRAIN:-0}" == 1 ]] && exit 7\n'
                '  [[ "${MISSING_BEST:-0}" != 1 ]] && touch "$DRAFT_KV_STAGE3_OUTPUT_DIR/best.pt"\n'
                '  exit 0\n'
                'fi\n'
                'echo "EVAL gpu=$DRAFT_KV_GPU_ID checkpoint=$DRAFT_KV_CHECKPOINT"\n'
            )
            fake_bash.chmod(0o755)
            output = root / "run with spaces" / "train"
            env = {
                **os.environ,
                "PATH": str(root) + os.pathsep + os.environ["PATH"],
                "DRAFT_KV_CHECKPOINT": "/wrong/inherited/checkpoint.pt",
                "FAIL_TRAIN": str(int(fail)),
                "MISSING_BEST": str(int(missing)),
            }
            result = subprocess.run(
                ["/bin/bash", str(PIPELINE), "--gpu-id", "3", "--output-dir", str(output)],
                env=env, text=True, capture_output=True,
            )
            return result, str(output / "best.pt")

    def test_success_evaluates_new_checkpoint_on_same_gpu(self):
        result, checkpoint = self.run_pipeline()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("TRAIN gpu=3", result.stdout)
        self.assertIn(f"EVAL gpu=3 checkpoint={checkpoint}", result.stdout)
        self.assertLess(result.stdout.index("TRAIN"), result.stdout.index("EVAL"))

    def test_training_failure_prevents_evaluation(self):
        result, _ = self.run_pipeline(fail=True)
        self.assertEqual(result.returncode, 7)
        self.assertNotIn("EVAL", result.stdout)

    def test_missing_checkpoint_prevents_evaluation(self):
        result, _ = self.run_pipeline(missing=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("EVAL", result.stdout)


if __name__ == "__main__":
    unittest.main()
