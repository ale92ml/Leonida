"""The verdict logic of runtime/test/check_tiny_frame.py on synthetic output: no loader, GPU or game files."""
import contextlib
import importlib.util
import io
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("check_tiny_frame", ROOT / "runtime" / "test" / "check_tiny_frame.py")
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


def good_output(extra_trace=(), drop=None, binding="shim"):
    """What a passing run prints: the binding report, the numbered trace and the guest's verdict."""
    calls = [call for call in list(check.EXPECTED) if call != drop]
    calls[10:10] = extra_trace  # between the views and the list's commands
    lines = ["tiny_frame.exe  base=0000000140000000 size=61440 entry=+0x1000  shim=7 host=40 module=0 unimplemented=0",
             f"    {binding:<13} d3d12_xs.dll!D3D12XboxCreateDevice",
             "running tiny_frame.exe"]
    lines += [f"[d3d12-trace] #{number} {call}" for number, call in enumerate(calls, 1)]
    lines += ["tiny_frame: PASS all 32 pixels are the clear colour (64, 128, 191, 255)", "tiny_frame: OK",
              "game exited with code 0", ""]
    return "\n".join(lines)


class CheckOutputTest(unittest.TestCase):
    def test_a_passing_run_has_no_problems(self):
        self.assertEqual(check.check_output(0, good_output()), [])

    def test_crlf_output_and_calls_between_the_expected_ones_are_accepted(self):
        output = good_output(extra_trace=["XboxID3D12CommandQueue::GetDesc called"]).replace("\n", "\r\n")
        self.assertEqual(check.check_output(0, output), [])

    def test_nonzero_exit_code(self):
        self.assertIn("exit code 3", check.check_output(3, good_output()))

    def test_missing_verdict_and_failed_check(self):
        output = good_output().replace("tiny_frame: OK", "tiny_frame: FAIL Signal\ntiny_frame: FAILED")
        problems = check.check_output(0, output)
        self.assertIn("the guest did not print `tiny_frame: OK`", problems)
        self.assertIn("the guest reported a failed check", problems)

    def test_missing_loader_exit_line(self):
        output = good_output().replace("game exited with code 0", "game exited with code 1")
        self.assertIn("gta6rt did not report `game exited with code 0`", check.check_output(0, output))

    def test_host_binding_is_reported(self):
        problems = check.check_output(0, good_output(binding="host"))
        self.assertEqual(problems, ["d3d12_xs.dll!D3D12XboxCreateDevice was not bound to the shim"])

    def test_missing_call_names_the_first_one_out_of_order(self):
        problems = check.check_output(0, good_output(drop="ID3D12CommandQueue::Signal ok"))
        self.assertEqual(problems, ["the trace has no `ID3D12CommandQueue::Signal ok` at its place in the "
                                    "first-frame order"])

    def test_a_call_that_did_not_succeed_is_reported(self):
        problems = check.check_output(0, good_output(extra_trace=["ID3D12Device::CreateHeap unsupported"]))
        self.assertEqual(problems, ["traced calls that did not succeed: ID3D12Device::CreateHeap unsupported"])

    def test_a_gap_in_the_numbering_is_an_invalid_trace(self):
        output = good_output().replace("[d3d12-trace] #3 ", "[d3d12-trace] #4 ")
        problems = check.check_output(0, output)
        self.assertEqual(len(problems), 1)
        self.assertTrue(problems[0].startswith("invalid trace sequence at entry 3"))

    def test_no_trace_at_all(self):
        output = "\n".join(line for line in good_output().split("\n") if not line.startswith("[d3d12-trace]"))
        self.assertEqual(check.check_output(0, output), ["the trace has no `D3D12XboxCreateDevice ok` at its place "
                                                         "in the first-frame order"])


class TimeoutProblemTest(unittest.TestCase):
    def test_a_run_that_printed_its_exit_line_hung_on_the_way_out(self):
        problem = check.timeout_problem(good_output().replace("\n", "\r\n"), 90)
        self.assertEqual(problem, "gta6rt printed `game exited with code 0` but did not end within 90 s: the process "
                                  "hung on its way out (ExitProcess), after the guest had finished")

    def test_a_run_without_its_exit_line_is_still_in_the_guest(self):
        output = good_output().split("tiny_frame: PASS all 32")[0]
        problem = check.timeout_problem(output, 2.5)
        self.assertEqual(problem, "gta6rt did not finish within 2.5 s, before `game exited with code`; its last line "
                                  "was: [d3d12-trace] #15 ID3D12CommandQueue::Signal ok")

    def test_no_output_at_all(self):
        self.assertTrue(check.timeout_problem("", 90).endswith("its last line was: (no output)"))

    def test_output_from_a_timeout_on_any_platform(self):
        self.assertEqual(check.as_text(None), "")
        self.assertEqual(check.as_text(b"tiny_frame: OK\xff"), "tiny_frame: OK�")
        self.assertEqual(check.as_text("tiny_frame: OK"), "tiny_frame: OK")


# A stand-in for gta6rt: prints what it is given, flushes, then sleeps (a host that does not end) or exits.
FAKE_LOADER = """import sys, time
sys.stdout.write({output!r})
sys.stdout.flush()
time.sleep({sleep})
"""


class MainTimeoutTest(unittest.TestCase):
    """main() end to end with a Python stand-in for gta6rt, so the timeout path runs without a GPU or a build."""

    def run_main(self, output, sleep, timeout):
        with tempfile.TemporaryDirectory() as folder:
            loader = Path(folder) / "fake_gta6rt.py"
            loader.write_text(FAKE_LOADER.format(output=output, sleep=sleep), encoding="utf-8", newline="\n")
            argv = ["check_tiny_frame.py", str(loader), str(Path(folder) / "tiny_frame.exe"),
                    "--runner", sys.executable, "--runner=-I", "--timeout", str(timeout)]
            printed = io.StringIO()
            start = time.monotonic()
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(printed):
                code = check.main()
            return code, printed.getvalue(), time.monotonic() - start

    def test_a_run_that_ends_passes(self):
        code, printed, _ = self.run_main(good_output(), 0, 60)
        self.assertEqual(code, 0, printed)
        self.assertIn("tiny_frame under gta6rt --run on the Vulkan D3D12 device: OK", printed)

    def test_a_host_that_hangs_after_the_guest_is_reported_with_its_output(self):
        # What Codex's run under the MSYS2 validation layer showed: every check passed, then no exit.
        code, printed, elapsed = self.run_main(good_output(), 120, 5)
        self.assertEqual(code, 1, printed)
        self.assertIn("tiny_frame: OK", printed)  # the output is kept, not lost in a traceback
        self.assertIn("FAIL: gta6rt printed `game exited with code 0` but did not end within 5 s", printed)
        self.assertLess(elapsed, 60)

    def test_a_guest_that_does_not_finish_names_its_last_line(self):
        partial = good_output().split("[d3d12-trace] #3 ")[0]
        code, printed, _ = self.run_main(partial, 120, 5)
        self.assertEqual(code, 1, printed)
        self.assertIn("FAIL: gta6rt did not finish within 5 s, before `game exited with code`; its last line was: "
                      "[d3d12-trace] #2 ID3D12Device::CreateCommandQueue ok", printed)


class SubsequenceTest(unittest.TestCase):
    def test_repeated_entries_need_as_many_occurrences(self):
        self.assertIsNone(check.ordered_subsequence(["a", "a", "b"], ["a", "x", "a", "b"]))
        self.assertEqual(check.ordered_subsequence(["a", "a", "b"], ["a", "b"]), "a")

    def test_order_matters(self):
        self.assertEqual(check.ordered_subsequence(["a", "b"], ["b", "a"]), "b")


if __name__ == "__main__":
    unittest.main()
