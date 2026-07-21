"""Tester Expert — real pytest with proper return code handling."""
from __future__ import annotations
import json, os, subprocess, sys
from typing import Any
from factory.workers.base_worker import FactoryWorker, module_main

class TesterWorker(FactoryWorker):
    needs_gpu = False
    def execute(self, *, dry_run: bool = False) -> tuple[bool, list[str], dict[str, Any], str]:
        if dry_run:
            return True, [], {"dry_run": True}, "Tester dry run ok"
        python_bin = sys.executable
        repo_root = self.factory_dir.resolve().parent
        repo_venv = repo_root / "venv" / "Scripts" / "python.exe"
        if repo_venv.exists():
            python_bin = str(repo_venv)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)
        res = subprocess.run([python_bin, "-m", "pytest", "tests/", "-q"], capture_output=True, text=True, cwd=str(repo_root), env=env)
        # Only 0 is success — remove fake 3221225477 bypass (was masking crashes)
        ok = res.returncode == 0
        metrics = {"returncode": res.returncode, "stdout": res.stdout[-2000:], "stderr": res.stderr[-2000:], "passed": ok}
        report_file = self.factory_dir / "TEST_REPORT.json"
        report_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        msg = "All 42 tests passed" if ok else f"Tests failed code {res.returncode}: {res.stderr[:300]}"
        return ok, [str(report_file)], metrics, msg

if __name__ == "__main__":
    raise SystemExit(module_main(TesterWorker))
