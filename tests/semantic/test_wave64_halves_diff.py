"""tools/wave64_halves_diff.py: the 1x64 vs 2x32 differential oracle for the wave64 graphics tiers.

No game data, no GPU, no SPIRV-Tools and no llvm-mc: every program is generated and run on
spirv_wave_interp.py. Besides a clean fixed-seed batch, the negative controls are the two harness traps that
produced false findings while the tool was being written - a fragment that writes a register tools/cf_diff.py
allocates, and a fragment that reads a register it never wrote - so neither can come back unnoticed.

    python -I tests/semantic/test_wave64_halves_diff.py
"""
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import support  # noqa: E402,F401  (sets the tools path)
import cf_diff  # noqa: E402
import wave64_halves_diff as diff  # noqa: E402

# Registers tools/cf_diff.py writes: the uniform scalars it computes on, the mask pairs its nesting allocates
# (sreg(20 + 2*depth), sreg(40 + ...), sreg(50 + ...)), the VGPRs of its bodies, and the pair the vertex frame in
# wave64_halves_diff keeps the launch EXEC in. A fragment that writes one of these diverges for a reason that has
# nothing to do with the analysis: s21, for one, is the high half of a saved 64-bit EXEC that a 32-lane wave never
# uses, so the wave restores a corrupted mask and its halves do not.
# v6 and v7 are the fragments' own accumulators, which the vertex frame folds into an exported word: everything
# else in v0-v11 belongs to cf_diff's bodies or to the frame.
HARNESS_SGPRS = set(cf_diff.SREGS) | set(range(20, 30)) | set(range(40, 60)) | {60, 61}
HARNESS_VGPRS = set(range(0, 12)) - {6, 7}
SGPR = re.compile(r"\bs(\d+)\b|\bs\[(\d+):(\d+)\]")
VGPR = re.compile(r"\bv(\d+)\b|\bv\[(\d+):(\d+)\]")


def written_and_read(text):
    """(written, read) register names of one instruction, from its text: the first operand of an instruction with
    a destination is written, the rest are read. Good enough for the fragment checks below."""
    op, _, rest = text.partition(" ")
    operands = [x.strip() for x in rest.split(",") if x.strip()]
    if not operands or op.startswith(("exp", "s_cmp", "s_branch", "s_cbranch", "v_cmpx")) or "store" in op:
        return set(), set(operands)
    if op.startswith("v_writelane"):
        # vD[lane] = sS leaves every other lane of vD as it was, but the paired v_readlane names the same
        # constant lane, so the fragment never reads a lane it did not write: count vD as written only.
        return {operands[0]}, set(operands[1:])
    return {operands[0]}, set(operands[1:])


def regs(names, pattern):
    out = set()
    for name in names:
        for m in pattern.finditer(name):
            if m.group(1) is not None:
                out.add(int(m.group(1)))
            else:
                out.update(range(int(m.group(2)), int(m.group(3)) + 1))
    return out


class Harness(unittest.TestCase):
    def test_self_test_passes(self):
        """The oracle accepts a per-lane program as tier A and invariant, and reports the lane id as tier C and
        divergent: a harness that compared nothing, or compared the same run twice, fails here."""
        self.assertIs(diff.self_test(), True)

    def test_no_fragment_writes_a_register_the_generator_uses(self):
        for name, lines in sorted(diff.FRAGS.items()):
            with self.subTest(fragment=name):
                written = set()
                for text in lines:
                    written |= written_and_read(text)[0]
                self.assertEqual(regs(written, SGPR) & HARNESS_SGPRS, set(),
                                 f"{name} writes a scalar the harness keeps live")
                self.assertEqual(regs(written, VGPR) & HARNESS_VGPRS, set(),
                                 f"{name} writes a VGPR the harness keeps live")

    def test_every_fragment_writes_its_scratch_before_reading_it(self):
        """A read of an untouched register takes whatever the interpreter starts with and says nothing about any
        shader, so each fragment has to write its own scratch (outside the harness's registers) first."""
        for name, lines in sorted(diff.FRAGS.items()):
            with self.subTest(fragment=name):
                # the frame zeroes v6 and v7 before the body, so a fragment may accumulate into them
                ready = set(HARNESS_SGPRS), set(HARNESS_VGPRS) | {6, 7}
                for text in lines:
                    written, read = written_and_read(text)
                    for pattern, done in ((SGPR, ready[0]), (VGPR, ready[1])):
                        missing = regs(read, pattern) - done
                        self.assertEqual(missing, set(), f"{name}: {text!r} reads {sorted(missing)} unwritten")
                        done |= regs(written, pattern)

    def test_a_register_collision_is_what_it_looks_like(self):
        """The trap itself, kept as a live demonstration: a fragment that writes s21 makes the two cuts differ
        without any help from the analysis. If this ever stops diverging the controls above are testing nothing."""
        bad = dict(diff.FRAGS)
        bad["collide"] = ["v_mov_b32 v30, s10", "v_readfirstlane_b32 s21, v30", "v_add_nc_u32 v6, s21, v6"]
        keys = diff.FRAG_KEYS
        try:
            diff.FRAGS, diff.FRAG_KEYS = bad, ["collide"]
            seen = [diff.check("idioms", 11, i, 3) for i in range(6)]
        finally:
            diff.FRAGS, diff.FRAG_KEYS = dict(bad), keys
            del diff.FRAGS["collide"]
        compared = [r for r in seen if "agree" in r]
        self.assertTrue(compared, "no program ran in both cuts")
        self.assertTrue(any(not r["agree"] for r in compared),
                        "writing the high half of the saved EXEC has to change the 64-lane run")

    def test_generation_is_deterministic(self):
        for mode in diff.MODES:
            with self.subTest(mode=mode):
                self.assertEqual(diff.one_program(mode, 3, 7, 3), diff.one_program(mode, 3, 7, 3))
                self.assertNotEqual(diff.one_program(mode, 3, 7, 3), diff.one_program(mode, 3, 8, 3))

    def test_each_mode_reaches_the_tier_it_is_there_for(self):
        """A batch that is all tier C proves nothing: the oracle only bites on tier A or B. `flow` and `kill`
        reach tier A, and `idioms` is the mode that reaches tier B at all - cf_diff emits no cross-lane
        instruction, so without the fragments the one tier that needs a rewrite is never generated."""
        for mode, want in (("flow", "A"), ("kill", "A"), ("idioms", "B")):
            with self.subTest(mode=mode):
                tiers = [diff.check(mode, 2, i, 3).get("tier") for i in range(12)]
                self.assertIn(want, tiers, f"{mode}: no tier {want} in {tiers}")

    def test_a_small_fixed_seed_batch_has_no_tier_a_or_b_divergence(self):
        for mode in diff.MODES:
            with self.subTest(mode=mode):
                counts = diff.summarise(diff.run_batch(mode, 4, 5, 3))
                self.assertGreater(counts["compared"], 0, f"{mode}: nothing was compared")
                self.assertEqual(counts["unsound"], 0, f"{mode}: a tier A/B program diverged")

    def test_empty_batch_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as exc:
            diff.main(["--programs", "0"])
        self.assertEqual(exc.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
