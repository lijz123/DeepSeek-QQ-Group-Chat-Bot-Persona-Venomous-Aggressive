"""Run the repository's synthetic offline regression suite.

These tests use fake models/OneBot and temporary files. Passing is not a real
model-quality, QQ upload, dependency-of-generated-code, or NapCat-login test.
No production config/auth file is loaded by this runner. The temporary reports
are removed by default; --report explicitly saves literal subprocess evidence.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def jobs(reports: Path, project: Path | None = None):
    ROOT = project if project is not None else Path(__file__).resolve().parents[1]
    python = sys.executable
    suites = [
        ("FILE_DELIVERY_TESTS", "file_delivery_tests.py"),
        ("CONTEXTUAL_OUTPUT_TESTS", "contextual_output_tests.py"),
        ("WIDE_OUTPUT_TESTS", "wide_output_tests.py"),
        ("SIMULTANEOUS_REPLY_TESTS", "simultaneous_reply_tests.py"),
        ("SEARCH_CLIENT_TESTS", "web_search_tests.py"),
        ("SEARCH_INTEGRATION_TESTS", "search_integration_tests.py"),
        ("IDENTITY_TESTS", "identity_tests.py"),
        ("OWNER_TARGETING_TESTS", "owner_targeting_tests.py"),
        ("DIRECT_VOCATIVE_TESTS", "direct_vocative_tests.py"),
        ("ROAST_DELEGATION_TESTS", "roast_delegation_tests.py"),
        ("OWNER_CHAT_TESTS", "owner_chat_tests.py"),
        ("DYNAMIC_DUEL_TESTS", "duel_generation_tests.py"),
        ("DUEL_TESTS", "duel_tests.py"),
        ("RETALIATION_TESTS", "retaliation_tests.py"),
        ("NOTIFICATION_INTENT_TESTS", "notification_intent_tests.py"),
        ("VOICE_TESTS", "voice_tests.py"),
        ("NATURAL_REMINDER_TESTS", "natural_reminder_tests.py"),
        ("LAUNCHER_PROGRESS_TESTS", "launcher_progress_tests.py"),
        ("REMINDER_ACK_TESTS", "reminder_ack_tests.py"),
        ("MENTION_REMINDER_TESTS", "mention_reminder_tests.py"),
        ("ADAPTIVE_OFFLINE_TESTS", "adaptive_review_tests.py"),
    ]
    result = []
    for label, filename in suites:
        report = reports / (label.lower() + ".json")
        result.append((label, [python, str(ROOT / "tests" / filename), str(ROOT), str(report)], report))
    result.extend([
        ("POPUP_OFFLINE_TESTS", [python, str(ROOT / "tests" / "popup_tests.py"), str(ROOT)], None),
        ("LAUNCHER_OFFLINE_TESTS", [python, str(ROOT / "tests" / "launcher_tests.py")], None),
        ("GPT_OFFLINE_TESTS", [python, str(ROOT / "tests" / "chatgpt_plan_tests.py")], None),
        ("SELFTEST", [python, str(ROOT / "tests" / "selftest.py")], None),
        ("DIALOGUE", [python, "-m", "unittest", "dialogue_selftest", "-v"], None),
    ])
    for label, filename in [
        ("UNDERSTANDING", "understanding_first_tests.py"),
        ("FRIENDLY", "session_tone_tests.py"),
        ("AMBIGUITY", "ambiguity_grounding_tests.py"),
    ]:
        report = reports / (label.lower() + ".json")
        result.append((label, [python, str(ROOT / "tests" / filename), str(ROOT), str(report)], report))
    report = reports / "semantic.json"
    result.append(("SEMANTIC", [python, str(ROOT / "tests" / "semantic_pipeline_tests.py"), str(ROOT), str(report)], report))
    result.extend([
        ("BOT_HELP", [python, str(ROOT / "app" / "bot.py"), "--help"], None),
        ("LOGIN_HELP", [python, str(ROOT / "app" / "chatgpt_auth.py"), "--help"], None),
        ("PREVIEW_HELP", [python, str(ROOT / "app" / "preview.py"), "--help"], None),
    ])
    return result


def checks(report: Path | None, stdout: str, stderr: str) -> int:
    if report is not None and report.is_file():
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
            number = data.get("tests", data.get("tests_run"))
            if type(number) is int and number >= 0:
                return number
        except (OSError, ValueError, TypeError):
            pass
    output = stdout + "\n" + stderr
    numbers = re.findall(r"Ran (\d+) tests?", output)
    if numbers:
        return int(numbers[-1])
    return len(re.findall(r"^\[PASS\]", stdout, re.MULTILINE))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run synthetic offline regression; no real model/QQ/NapCat login.")
    parser.add_argument("--report", type=Path, help="Save literal stdout/stderr JSON evidence (private local report; never commit it).")
    parser.add_argument("--list", action="store_true", help="List suite names without running tests.")
    args = parser.parse_args()
    destination = args.report.expanduser().resolve() if args.report is not None else None
    if destination is not None and destination.exists():
        parser.error("--report already exists; choose a new filename instead of overwriting it")
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", PYTHONPATH=str(ROOT))
    for key in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        env.pop(key, None)
    events = []
    success = True
    with tempfile.TemporaryDirectory(prefix="qqbot offline ") as temporary:
        work = Path(temporary)
        project = work / "project"
        project.mkdir()
        names = {p.name for p in ROOT.iterdir() if p.is_file()
                 and (p.suffix in {".bat", ".ps1", ".md", ".txt"}
                      or p.name in {"config.example.json", "launcher.example.json", "RELEASE_MANIFEST.json", ".gitignore"})}
        # Keep the actual app/tools/tests layout in the isolated test copy.
        # Never flatten source files or copy live generated files/databases.
        for folder in ("app", "tools", "tests"):
            directory = ROOT / folder
            if directory.is_symlink() or (hasattr(directory, "is_junction") and directory.is_junction()):
                raise ValueError("Source directory cannot be linked")
            names.update(p.relative_to(ROOT).as_posix() for p in directory.glob("*.py") if p.is_file())
        # Copy source only. Private config, credentials and live runtime databases
        # never enter the disposable test project, even after the user configures it.
        for name in sorted(names):
            relative = Path(name)
            if (relative.is_absolute() or ".." in relative.parts
                    or (len(relative.parts) > 1 and (len(relative.parts) != 2 or relative.parts[0] not in {"app", "tools", "tests"}))
                    or relative.name in {"config.json", "launcher.local.json", "auth.json"}
                    or (relative.suffix == ".json" and relative.name not in {"config.example.json", "launcher.example.json", "RELEASE_MANIFEST.json"})):
                raise ValueError("Manifest contains a non-public source path")
            source = ROOT / relative
            if source.is_symlink() or not source.is_file():
                raise ValueError("Public source file is missing or linked")
            target = project / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        env["PYTHONPATH"] = os.pathsep.join((str(project / "app"), str(project / "tests")))
        selected = jobs(work, project)
        if args.list:
            for label, _, _ in selected:
                print(label)
            return 0
        print("Offline only: synthetic models/OneBot; generated code is not executed.", flush=True)
        for label, command, report in selected:
            started = time.monotonic()
            try:
                result = subprocess.run(command, cwd=work, env=env, capture_output=True,
                    text=True, encoding="utf-8", errors="replace", timeout=300)
                stdout, stderr, status = result.stdout, result.stderr, result.returncode
            except subprocess.TimeoutExpired as error:
                stdout = error.stdout or ""
                stderr = error.stderr or ""
                if isinstance(stdout, bytes): stdout = stdout.decode("utf-8", errors="replace")
                if isinstance(stderr, bytes): stderr = stderr.decode("utf-8", errors="replace")
                stderr += "\nOffline test exceeded the 300-second timeout."
                status = 124
            except OSError as error:
                stdout, stderr, status = "", str(error), 127
            count = checks(report, stdout, stderr)
            events.append(dict(label=label, command=command, stdout=stdout, stderr=stderr,
                exit_status=status, tests_or_checks=count, elapsed_seconds=round(time.monotonic()-started, 3)))
            print(label + ": " + ("PASS" if status == 0 else "FAIL") + " (" + str(count) + " checks)", flush=True)
            if status:
                success = False
                # Keep failure evidence visible even when no persistent report was requested.
                print(stdout, end="")
                print(stderr, end="", file=sys.stderr)
                break
    total = sum(event["tests_or_checks"] for event in events)
    summary = dict(success=success, suites_run=len(events), tests_or_checks=total,
        scope="Synthetic offline protocol/dataflow checks, not real model or QQ verification",
        real_model_requests=0, real_QQ_messages=0, production_config_read=False,
        events=events)
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print("Private local report saved to the explicitly requested path.")
    print(("PASS" if success else "FAIL") + ": " + str(total) + " offline checks; real model/QQ requests: 0.")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
