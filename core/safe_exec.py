"""Run one allow-listed, read-only command. Nothing else.

Collectors never call subprocess themselves; they go through run_allowed():

  - The command must EXACTLY match one in the manifest's
    [collector_policy] commands list, e.g. ["arp", "-a"]. No extra
    arguments, no wildcards.
  - shell=False: the words are passed straight to the program, so shell
    tricks like "; rm -rf ~" or "$(...)" are just harmless text.
  - stdin is closed, so the command can't wait for input.
  - A timeout and an output cap, so a stuck or chatty command can't hang
    or flood the console.
  - LC_ALL=C on Linux, so output is in plain English and parses the same
    on every machine.
"""

from __future__ import annotations

import os
import shutil
import subprocess

DEFAULT_TIMEOUT = 20            # seconds
MAX_OUTPUT_BYTES = 2_000_000    # 2 MB is far more than any ARP table


class CommandNotAllowed(Exception):
    pass


class CommandFailed(Exception):
    pass


def run_allowed(argv: list[str], allowed: list[list[str]],
                timeout: float = DEFAULT_TIMEOUT) -> str:
    """Run argv if it is on the allow-list and return its output as text."""
    if not isinstance(argv, list) or argv not in allowed:
        raise CommandNotAllowed(f"command {argv!r} is not in this protocol's allow-list")

    program = shutil.which(argv[0])
    if program is None:
        raise CommandFailed(f"program {argv[0]!r} was not found on this system")

    env = dict(os.environ)
    if os.name != "nt":
        env["LC_ALL"] = "C"

    try:
        completed = subprocess.run(
            [program, *argv[1:]],
            shell=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        raise CommandFailed(f"{argv[0]} did not finish within {timeout}s") from None
    except OSError as e:
        raise CommandFailed(f"could not run {argv[0]}: {e}") from None

    output = completed.stdout[:MAX_OUTPUT_BYTES]
    # errors="replace": odd bytes become "?" instead of crashing.
    text = output.decode("utf-8", errors="replace")
    if completed.returncode != 0 and not text.strip():
        err = completed.stderr[:500].decode("utf-8", errors="replace").strip()
        raise CommandFailed(f"{argv[0]} failed (exit {completed.returncode}): {err}")
    return text
