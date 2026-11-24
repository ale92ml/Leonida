"""Differential check of the wave64 graphics partition analysis: one 64-lane wave against two 32-lane waves.

    python -I tools/wave64_halves_diff.py --programs 200
    python -I tools/wave64_halves_diff.py --mode kill --programs 400 --seed 7
    python -I tools/wave64_halves_diff.py --programs 300 --report w64diff.json

`tools/wave64_halves.py` does not emulate a wave64 VS/PS/GS on a 32-wide subgroup (NVIDIA, Intel, Apple: the host
forms the subgroups of the graphics stages, so there is no partner half to exchange with). It classifies: tier A
exact as translated, tier B exact with the rewrite the emission makes, tier C not exact. The contract
(docs/design.md, *Wave64 graphics: partition invariance*) states exactness as something executable - **the
observable results must not change when the wave is cut into two 32-lane halves** - so a generated program plus
the wave interpreter decides it both ways:

  * tier A or B and the two cuts differ -> a soundness bug in the analysis (the port renders this shader wrong),
  * tier C and the two cuts agree       -> a false negative (coverage given up; not an error, but reported).

Programs are generated, never read from a shader: `tools/cf_diff.py` supplies the RDNA2 control-flow shapes, this
file adds the wave64 frame of a vertex and of a pixel shader, the cross-lane idiom fragments that `cf_diff` never
emits (so tier B is reached at all), and a mutational generator around the pixel-kill early exit. No game data is
involved; nothing is read from disk but the temporary listing each program is written to.

The six modes, three per stage:

    flow / ps              cf_diff's control-flow shapes through that stage's frame
    idioms / ps_idioms     the same with cross-lane fragments spliced in, which is the only way tier B is
                           generated: cf_diff emits no readfirstlane, readlane, writelane, waterfall, swizzle or
                           permlane, so without them the one tier that needs a rewrite never appears
    kill / ps_kill         one or two mutations of the pixel-kill shape the test suite asserts is tier A, so
                           samples land on the boundary of kill_vote / inert_exit / confined / vm_export_before

The pixel modes matter out of proportion to their number: 12,101 of the corpus's 13,405 wave64 graphics shaders
are pixel shaders and 8,585 of its 9,287 tier C, and the quad rules, the helper lanes and the `exp ... vm` that
decides a pixel's fate exist in no other stage.

Exit status: 0 clean, 1 a tier-A/B divergence, 2 nothing was compared.

A program the two cuts cannot both run is reported as a skip, not a pass. The common ones are honest: a tier-C
`v_readlane_b32` past lane 31 makes the interpreter refuse the 32-lane run ("Shuffle from invocation 40, which is
not run"), which is the undefined result a 32-wide GPU would give; a generated loop can outrun the step limit; and
on a branch without the SOPC inequality mnemonic the translator refuses `s_cmp_ne_u32`.

Two traps a fragment must avoid, both of which produced false findings before they were fixed, and both covered by
tests/semantic/test_wave64_halves_diff.py:
  * a register `cf_diff` uses (s10-s13, mask pairs at 20/40/50 + 2*depth, v0-v11): writing s21 clobbers the high
    half of a saved 64-bit EXEC, which a 32-lane wave does not use, so the two cuts differ for a reason that has
    nothing to do with the analysis,
  * a register the fragment reads without writing: the value is whatever the interpreter starts with.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import tempfile
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
SEM_DIR = os.path.join(os.path.dirname(HERE), "tests", "semantic")
for _p in (HERE, SEM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cf_diff  # noqa: E402
from spirv_wave_interp import UnsupportedConstruct, run_wave  # noqa: E402

MODES = ("flow", "idioms", "kill", "ps", "ps_idioms", "ps_kill")
PS_MODES = ("ps", "ps_idioms", "ps_kill")

# A pixel shader's per-lane value is the pixel it covers, not a lane id: interpolate attribute 0, scale it by the
# 64x64 target of the ps_w64_* fixtures and pack it as p = y * 64 + x, the same number
# tests/semantic/test_wave64_halves.py's _ps_xy derives from the 2x2 quad layout of fragment lanes. Both cuts then
# shade the same pixels, so a difference between them is the analysis's business and not the frame's. v0 and v1
# hold the barycentrics until the last interpolation has run, so p lands in v0 only afterwards.
PS_PIXEL = ["v_interp_p1_f32_e32 v12, v0, attr0.x", "v_interp_p2_f32_e32 v12, v1, attr0.x",
            "v_interp_p1_f32_e32 v13, v0, attr0.y", "v_interp_p2_f32_e32 v13, v1, attr0.y",
            "v_mul_f32_e32 v12, 0x42800000, v12", "v_mul_f32_e32 v13, 0x42800000, v13",
            "v_cvt_u32_f32_e32 v12, v12", "v_cvt_u32_f32_e32 v13, v13",
            "v_lshl_add_u32 v0, v13, 6, v12"]

# Scratch registers of the fragments and the kill shapes, kept clear of everything cf_diff allocates: s10-s13 and
# the mask pairs at 20/40/50 + 2*depth are its, and so are v0-v11, so these start at s70 and v30.
FRAGS = {
    # a scalar broadcast into a VGPR, read back: uniform over the lanes the write ran for
    "rfl_uniform": ["v_mov_b32 v30, s10", "v_readfirstlane_b32 s70, v30", "v_add_nc_u32 v6, s70, v6"],
    "rfl_active": ["v_mov_b32 v31, s11", "v_readfirstlane_b32 s71, v31", "v_xor_b32 v6, s71, v6"],
    # readlane of a uniform VGPR, from a lane a 32-wide subgroup does not have
    "readlane_uniform_hi": ["v_mov_b32 v32, s12", "v_readlane_b32 s72, v32, 40", "v_add_nc_u32 v7, s72, v7"],
    "readlane_uniform_var": ["v_mov_b32 v33, s13", "s_mov_b32 s73, 37", "v_readlane_b32 s72, v33, s73",
                             "v_xor_b32 v7, s72, v7"],
    # a scalar spilled to a constant lane and read back (lane 45 is past the end of a 32-wide subgroup)
    "spill_hi": ["s_mov_b32 s74, 0x51", "v_writelane_b32 v34, s74, 45", "v_readlane_b32 s75, v34, 45",
                 "v_add_nc_u32 v6, s75, v6"],
    "spill_lo": ["s_mov_b32 s74, 0x33", "v_writelane_b32 v35, s74, 7", "v_readlane_b32 s75, v35, 7",
                 "v_xor_b32 v6, s75, v6"],
    # a waterfall loop: once per distinct value, which is per subgroup
    "waterfall": ["v_and_b32 v36, 7, v0", "v_readfirstlane_b32 s76, v36", "v_cmp_eq_u32 vcc, s76, v36",
                  "s_and_saveexec_b64 s[78:79], vcc", "s_mul_i32 s77, s76, 17", "v_add_nc_u32 v6, s77, v6",
                  "s_xor_b64 exec, exec, s[78:79]", "s_cbranch_execnz -7", "s_mov_b64 exec, s[60:61]"],
    # cross-lane reads that stay inside a half
    "swizzle": ["ds_swizzle_b32 v37, v0 offset:0x801f", "v_add_nc_u32 v7, v37, v7"],
    "permlane16": ["v_permlane16_b32 v38, v0, s10, s11", "v_xor_b32 v7, v38, v7"],
    # the lane id, 0..63 in the wave and 0..31 in a half
    "lane_id": ["v_mbcnt_lo_u32_b32 v39, -1, 0", "v_mbcnt_hi_u32_b32 v39, -1, v39", "v_add_nc_u32 v6, v39, v6"],
}
FRAG_KEYS = sorted(FRAGS)


class VsEmitter(cf_diff.Emitter):
    """cf_diff's emitter with a wave64 vertex-shader frame: per-lane data from VertexIndex (v0), the scalars the
    bodies compute on initialised, and an epilogue that exports both the per-lane and the scalar state so a
    difference in either shows up as a different word."""

    pixel = False

    def program(self, body):
        self.ins("s_mov_b64 s[60:61], exec")        # a 64-bit op on EXEC: the translator reads the program wave64
        if self.pixel:
            for text in PS_PIXEL:
                self.ins(text)
        self.ins("v_and_b32 v1, 7, v0")
        self.ins("v_lshrrev_b32 v2, 2, v0")
        self.ins("v_mul_lo_u32 v3, v0, 5")
        self.ins("v_and_b32 v3, 15, v3")
        self.ins("v_xor_b32 v8, 9, v0")
        self.ins("v_mov_b32 v9, 0")
        self.ins("v_mov_b32 v11, 1")
        self.ins("v_mov_b32 v5, 0")
        for i, s in enumerate(cf_diff.SREGS):
            self.ins(f"s_mov_b32 s{s}, {(3, 5, 0, 12)[i]}")
        self.ins("v_mov_b32 v6, 0")
        self.ins("v_mov_b32 v7, 0")
        self.nodes(body)
        # Every launch lane folds the state into the exported words, so a difference anywhere is observable: the
        # scalars into v10, and v6 / v7, which the idiom fragments accumulate into, through v9.
        self.ins("s_mov_b64 exec, s[60:61]")
        self.ins(f"v_mov_b32 v10, s{cf_diff.SREGS[0]}")
        for s in cf_diff.SREGS[1:]:
            self.ins("v_lshlrev_b32 v10, 3, v10")
            self.ins(f"v_xor_b32 v10, s{s}, v10")
        self.ins("v_xor_b32 v9, v6, v9")
        self.ins("v_lshlrev_b32 v9, 3, v9")
        self.ins("v_xor_b32 v9, v7, v9")
        if self.pixel:
            self.ins("exp mrt0 v8, v9, v10, v11 done vm")
        else:
            self.ins("exp pos0 v5, v5, v5, v5 done")
            self.ins("exp param0 v8, v9, v10, v11")
        self.ins("s_endpgm")


class PsEmitter(VsEmitter):
    """The same frame for a pixel shader: the per-lane value is the pixel the lane covers and the state leaves
    through mrt0, so the quad rules and the helper lanes of a fragment wave are in play. 12,101 of the corpus's
    13,405 wave64 graphics shaders are pixel shaders, and 8,585 of its 9,287 tier C."""

    pixel = True


def inject(rng, body, count):
    """Splice `count` idiom fragments into random statement positions of a cf_diff body, nested blocks included."""
    used = []
    for _ in range(count):
        key = rng.choice(FRAG_KEYS)
        used.append(key)
        blocks = [body]

        def walk(nodes):
            for node in nodes:
                for x in node[1:]:
                    if isinstance(x, list):
                        blocks.append(x)
                        walk(x)

        walk(body)
        target = rng.choice(blocks)
        at = rng.randrange(len(target) + 1)
        target[at:at] = [("op", text) for text in FRAGS[key]]
    return used


# ---- the pixel-kill early exit -------------------------------------------------------------------------------
# The rule with the most moving parts (kill_vote / inert_exit / confined / vm_export_before in wave64_halves.py)
# and, in the test suite, only hand-written fixtures. Mutating the shape the suite asserts is tier A puts samples
# on the boundary of the rule instead of deep inside tier C, where they prove nothing.
KILL_WORK = ("v_mul_u32_u24_e32 v6, 3, v0", "v_mov_b32_e32 v7, 1")


def kill_base(pixel=False):
    # A pixel shader's exit is the real thing the rule is named after: EXEC back to the live mask, a null export
    # with `vm` so the pixels of an empty half are discarded, and nothing else.
    exit_ = ["s_mov_b64 exec, s[20:21]", "exp null off, off, off, off done vm", "s_endpgm"] if pixel else \
        ["s_mov_b64 exec, s[20:21]", "exp pos0 v5, v5, v5, v5 done", "s_endpgm"]
    return {"head": ["s_mov_b64 s[20:21], exec", "v_cmp_gt_u32_e32 vcc, 40, v0",
                     "s_andn2_b64 s[20:21], s[20:21], vcc"],
            "vote": "s_cbranch_scc0",
            "run": ["s_wqm_b64 s[22:23], s[20:21]", "s_and_b64 exec, exec, s[22:23]"] + list(KILL_WORK),
            "exit": exit_}


def _set(key, index, text):
    return lambda p, r: p[key].__setitem__(index, text)


def _put(key, text):
    return lambda p, r: p.__setitem__(key, text)


def _insert(key, index, text):
    return lambda p, r: p[key].insert(index if index is not None else r.randrange(0, len(p[key]) + 1), text)


def _append(key, text):
    return lambda p, r: p[key].append(text)


KILL_MUTATIONS = [
    # the mask the run side narrows into
    ("derive_mov", _set("run", 0, "s_mov_b64 s[22:23], s[20:21]")),
    ("derive_and_launch", _set("run", 0, "s_and_b64 s[22:23], s[20:21], s[10:11]")),
    ("derive_andn2_vcc", _set("run", 0, "s_andn2_b64 s[22:23], s[20:21], vcc")),
    ("derive_or_launch", _set("run", 0, "s_or_b64 s[22:23], s[20:21], s[10:11]")),
    ("derive_not", _set("run", 0, "s_not_b64 s[22:23], s[20:21]")),
    ("derive_wqm_launch", _set("run", 0, "s_wqm_b64 s[22:23], s[10:11]")),
    # how EXEC is narrowed
    ("narrow_saveexec", _set("run", 1, "s_and_saveexec_b64 s[30:31], s[22:23]")),
    ("narrow_andn2_exec", _set("run", 1, "s_andn2_b64 exec, exec, vcc")),
    ("narrow_andn2_saveexec", _set("run", 1, "s_andn2_saveexec_b64 s[30:31], vcc")),
    ("narrow_mov_exec", _set("run", 1, "s_mov_b64 exec, s[22:23]")),
    ("narrow_cmpx", _set("run", 1, "v_cmpx_le_u32_e32 40, v0")),
    ("narrow_or_saveexec", _set("run", 1, "s_or_saveexec_b64 s[30:31], s[22:23]")),
    ("narrow_xor_saveexec", _set("run", 1, "s_xor_saveexec_b64 s[30:31], s[22:23]")),
    ("narrow_andn1_saveexec", _set("run", 1, "s_andn1_saveexec_b64 s[30:31], vcc")),
    # work and EXEC writes inserted in the run side
    ("work_before", _insert("run", 0, KILL_WORK[0])),
    ("work_mid", _insert("run", 1, KILL_WORK[1])),
    ("work_anywhere", _insert("run", None, KILL_WORK[0])),
    ("widen_mov_launch", _insert("run", None, "s_mov_b64 exec, s[10:11]")),
    ("widen_or_launch", _insert("run", None, "s_or_b64 exec, exec, s[10:11]")),
    ("widen_not", _insert("run", None, "s_not_b64 exec, exec")),
    ("widen_minus_one", _insert("run", None, "s_mov_b64 exec, -1")),
    ("renarrow", _insert("run", None, "s_and_b64 exec, exec, s[22:23]")),
    ("wqm_exec", _insert("run", None, "s_wqm_b64 exec, exec")),
    ("mask_rewrite", _insert("run", None, "s_mov_b64 s[20:21], exec")),
    ("store_in_run", _insert("run", None, "global_store_dword v0, v6, s[4:5]")),
    # the live mask and the SCC the branch votes on
    ("head_cmp_lg", _append("head", "s_cmp_lg_u64 s[20:21], 0")),
    ("head_cmp_eq", _append("head", "s_cmp_eq_u64 s[20:21], 0")),
    ("head_redef", _append("head", "s_mov_b64 s[20:21], exec")),
    ("head_and", _set("head", 2, "s_and_b64 s[20:21], s[20:21], vcc")),
    ("head_xor", _set("head", 2, "s_xor_b64 s[20:21], s[20:21], vcc")),
    ("head_or", _set("head", 2, "s_or_b64 s[20:21], s[20:21], vcc")),
    ("head_cmpx", _put("head", ["s_mov_b64 s[20:21], exec", "v_cmpx_le_u32_e32 40, v0",
                                "s_and_b64 s[20:21], s[20:21], exec"])),
    ("head_bcnt", lambda p, r: p["head"].extend(["s_bcnt1_i32_b64 s24, s[20:21]", "s_cmp_lg_u32 s24, 0"])),
    ("head_threshold", lambda p, r: p["head"].__setitem__(
        1, f"v_cmp_gt_u32_e32 vcc, {r.choice((8, 24, 32, 56))}, v0")),
    # the vote
    ("vote_scc1", _put("vote", "s_cbranch_scc1")),
    ("vote_vccz", _put("vote", "s_cbranch_vccz")),
    ("vote_vccnz", _put("vote", "s_cbranch_vccnz")),
    ("vote_execz", _put("vote", "s_cbranch_execz")),
    # the exit a half with an empty mask takes
    ("exit_zero", _set("exit", 0, "s_mov_b64 exec, 0")),
    ("exit_and", _set("exit", 0, "s_and_b64 exec, exec, s[20:21]")),
    ("exit_launch", _set("exit", 0, "s_mov_b64 exec, s[10:11]")),
    ("exit_wqm", _set("exit", 0, "s_wqm_b64 exec, s[20:21]")),
    ("exit_no_export", lambda p, r: p.__setitem__("exit", [p["exit"][0], "s_endpgm"])),
    ("exit_work", _insert("exit", 1, "v_mov_b32_e32 v6, 5")),
    ("exit_param", _insert("exit", 1, "exp param0 v6, v7, v5, v5")),
    ("exit_store", _insert("exit", 1, "global_store_dword v0, v6, s[4:5]")),
]


def _size(text):
    """Byte size of an instruction, for the branch offsets of the fake encoding the listing carries."""
    lines = cf_diff.Emitter(64)
    lines.ins(text)
    return lines.assemble()[0][1]


PS_ONLY_SKIP = {"exit_param"}          # `exp param0` is not a pixel-shader export


def kill_lines(rng, pixel=False):
    """A mutated pixel-kill shape as a raw instruction listing, and the names of the mutations applied."""
    p, names = kill_base(pixel), []
    choices = [m for m in KILL_MUTATIONS if not (pixel and m[0] in PS_ONLY_SKIP)]
    for _ in range(rng.randrange(1, 3)):
        name, fn = rng.choice(choices)
        fn(p, rng)
        names.append(name)
    epilogue = ["exp mrt0 v6, v7, v5, v5 done vm", "s_endpgm"] if pixel else \
        ["v_mov_b32_e32 v5, 0", "exp pos0 v5, v5, v5, v5 done", "exp param0 v6, v7, v5, v5", "s_endpgm"]
    tail = p["run"] + epilogue
    hop = sum(_size(t) for t in tail) // 4
    body = ["s_mov_b64 s[10:11], exec"] + (list(PS_PIXEL) if pixel else []) + \
        ["v_mov_b32_e32 v6, 0", "v_mov_b32_e32 v7, 0", "v_mov_b32_e32 v5, 0"] + \
        p["head"] + [f"{p['vote']} {hop}"] + tail + p["exit"]
    return names, [(text, _size(text)) for text in body]


# ---- translation and execution -------------------------------------------------------------------------------
def translate(name, lines):
    """The translated SPIR-V and the recorded wave64 graphics verdict, or (None, reason) when it does not
    translate. The listing goes to a temporary folder; no other path is read."""
    import rdna2spirv

    text = cf_diff.listing_text(lines, name)
    with tempfile.TemporaryDirectory(prefix="w64-diff-") as tmp:
        dis = os.path.join(tmp, "disasm")
        os.mkdir(dis)
        path = os.path.join(dis, name + ".s")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(cf_diff.STUB_DXBC)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(path)
                verdict = rdna2spirv.translate.last_gen.w64g
                gc.collect()
        except rdna2spirv.Unsupported as e:
            return None, f"translator unsupported: {e}"
    return spv, verdict


def run_cut(spv, wave):
    """Per lane, the words of the program's param0 export, running 64 lanes as 64/wave waves of `wave`."""
    out = []
    for half in range(64 // wave):
        res = run_wave(spv, wave=wave,
                       builtins={"VertexIndex": lambda lane, base=half * wave: base + lane,
                                 "InstanceIndex": lambda lane: 0,
                                 "BaseVertex": lambda lane: 0, "BaseInstance": lambda lane: 0})
        for o in res["outputs"]:
            words = dict(o) if o else {}
            out.append(next((v for key, v in words.items() if key.endswith("param0")), None))
    return out


def ps_xy(lane):
    """The pixel of a fragment lane on an 8x8 tile: lane 4q+c is the quad's top-left, top-right, bottom-left,
    bottom-right (the same layout as `_ps_xy` in tests/semantic/test_wave64_halves.py)."""
    q, c = divmod(lane, 4)
    return 2 * (q % 4) + (c & 1), 2 * (q // 4) + (c >> 1)


def run_cut_ps(spv, wave):
    """The same for a pixel shader: attribute 0 carries the pixel's centre on a 64x64 target, so each cut shades
    the same pixels, and the words come from mrt0. A lane the shader killed exports nothing and stays None."""
    out = []
    for half in range(64 // wave):
        base = half * wave
        res = run_wave(spv, wave=wave,
                       inputs={0: lambda lane, b=base: ((ps_xy(b + lane)[0] + 0.5) / 64.0,
                                                        (ps_xy(b + lane)[1] + 0.5) / 64.0, 0.0, 0.0)})
        for o in res["outputs"]:
            words = dict(o) if o else {}
            out.append(next((v for key, v in words.items() if key.endswith("mrt0")), None))
    return out


def one_program(mode, seed, index, depth):
    """(label, listing) of program `index` of a batch: deterministic in (mode, seed, index, depth)."""
    rng = random.Random(f"{mode}:{seed}:{index}")
    pixel = mode in PS_MODES
    if mode in ("kill", "ps_kill"):
        names, lines = kill_lines(rng, pixel)
        return ",".join(names), lines
    body = cf_diff.generate(rng, 64, max_depth=depth, budget=rng.randrange(3, 11))
    used = inject(rng, body, rng.randrange(1, 4)) if mode in ("idioms", "ps_idioms") else []
    emitter = (PsEmitter if pixel else VsEmitter)(64)
    emitter.program(body)
    return ",".join(used), emitter.assemble()


def check(mode, seed, index, depth):
    """One program's verdict: a dict with its tier, reasons and whether the two cuts agree."""
    label, lines = one_program(mode, seed, index, depth)
    pixel = mode in PS_MODES
    cut = run_cut_ps if pixel else run_cut
    name = f"{'ps' if pixel else 'vs'}_w64d_{mode}_{seed}_{index:05d}"
    spv, verdict = translate(name, lines)
    if spv is None:
        return {"name": name, "label": label, "skip": verdict}
    if not verdict:
        return {"name": name, "label": label, "skip": "no wave64 graphics verdict (not read as wave64)"}
    out = {"name": name, "label": label, "tier": verdict["tier"],
           "reasons": sorted(verdict["reasons"]), "idioms": sorted(verdict["idioms"])}
    try:
        whole, halves = cut(spv, 64), cut(spv, 32)
    except UnsupportedConstruct as e:
        out["skip"] = f"wave interp: {e}"
        return out
    out["agree"] = whole == halves
    if not out["agree"]:
        out["lanes"] = [k for k in range(64) if whole[k] != halves[k]]
        out["first"] = {"lane": out["lanes"][0], "wave64": str(whole[out["lanes"][0]]),
                        "halves": str(halves[out["lanes"][0]])}
    return out


def run_batch(mode, seed, programs, depth, verbose=False):
    """Every program of a batch, as a list of check() results."""
    results = []
    for i in range(programs):
        r = check(mode, seed, i, depth)
        results.append(r)
        if verbose:
            print(f"  {r['name']} {r.get('tier', '-')} {','.join(r.get('reasons', [])) or '-'}"
                  + ("" if "skip" not in r else f" SKIP {r['skip']}"))
        if r.get("tier") in ("A", "B") and r.get("agree") is False:
            print(f"UNSOUND {r['name']}: tier {r['tier']} but the two cuts differ in {len(r['lanes'])} lanes")
            print(f"  reasons={r['reasons']} idioms={r['idioms']} shape={r['label']}")
            print(f"  lane {r['first']['lane']}: 64 lanes -> {r['first']['wave64']}, "
                  f"2x32 -> {r['first']['halves']}")
            for text, _ in one_program(mode, seed, i, depth)[1]:
                print("    " + text)
            sys.stdout.flush()
    return results


def summarise(results):
    """Counts: tiers, how many tier A/B diverged (each a soundness bug), how many tier C agreed, and the skips."""
    out = {"programs": len(results), "tiers": {}, "unsound": 0, "compared": 0,
           "tier_c_agree": 0, "tier_c_diverge": 0, "skips": {}, "reasons": {}}
    for r in results:
        if "skip" in r and "tier" not in r:
            out["skips"][r["skip"]] = out["skips"].get(r["skip"], 0) + 1
            continue
        tier = r.get("tier", "-")
        out["tiers"][tier] = out["tiers"].get(tier, 0) + 1
        for reason in r.get("reasons", []):
            out["reasons"][reason] = out["reasons"].get(reason, 0) + 1
        if "skip" in r:
            out["skips"][r["skip"]] = out["skips"].get(r["skip"], 0) + 1
            continue
        if "agree" not in r:
            continue
        out["compared"] += 1
        if tier in ("A", "B") and not r["agree"]:
            out["unsound"] += 1
        if tier == "C":
            out["tier_c_agree" if r["agree"] else "tier_c_diverge"] += 1
    return out


def self_test():
    """A fixed shape the analysis calls tier A and one it calls tier C, both run in both cuts: the oracle has to
    accept the first and report the second as a false negative, so a broken harness cannot pass silently."""
    clean = [("s_mov_b64 s[60:61], exec", 4), ("v_mov_b32 v6, 0", 4),
             ("v_mul_lo_u32 v6, 3, v0", 8), ("exp param0 v6, v6, v6, v6", 4), ("s_endpgm", 4)]
    spv, verdict = translate("vs_w64d_self_clean", clean)
    assert spv is not None and verdict and verdict["tier"] == "A", (verdict,)
    assert run_cut(spv, 64) == run_cut(spv, 32), "a per-lane program must not depend on the partition"
    lane = [("s_mov_b64 s[60:61], exec", 4), ("v_mov_b32 v6, 0", 4),
            ("v_mbcnt_lo_u32_b32 v6, -1, 0", 8), ("v_mbcnt_hi_u32_b32 v6, -1, v6", 8),
            ("exp param0 v6, v6, v6, v6", 4), ("s_endpgm", 4)]
    spv, verdict = translate("vs_w64d_self_lane", lane)
    assert spv is not None and verdict and verdict["tier"] == "C", (verdict,)
    assert run_cut(spv, 64) != run_cut(spv, 32), "the lane id has to differ between the wave and its halves"
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description="Run generated wave64 graphics programs as 1x64 and as 2x32 lanes "
                                            "and check the partition analysis against the result.")
    p.add_argument("--programs", type=int, default=100, help="programs per mode (default 100)")
    p.add_argument("--seed", type=int, default=1, help="batch seed; the batch is deterministic in it")
    p.add_argument("--depth", type=int, default=3, help="maximum control-flow nesting of a generated body")
    p.add_argument("--mode", choices=MODES + ("all",), default="all",
                   help="flow/idioms/kill run a vertex frame, ps/ps_idioms/ps_kill a pixel one; idioms splices in the cross-lane fragments, kill mutates the pixel-kill shape")
    p.add_argument("--report", help="write the counts to this JSON file")
    p.add_argument("--verbose", action="store_true", help="print every program's tier")
    p.add_argument("--self-test", action="store_true", help="check the harness itself and exit")
    a = p.parse_args(argv)
    if a.self_test:
        self_test()
        print("self test: ok")
        return 0
    if a.programs < 1:
        p.error("--programs must be at least 1")
    modes = MODES if a.mode == "all" else (a.mode,)
    report, status = {}, 0
    for mode in modes:
        results = run_batch(mode, a.seed, a.programs, a.depth, a.verbose)
        counts = summarise(results)
        report[mode] = counts
        tiers = ", ".join(f"tier {t}: {n}" for t, n in sorted(counts["tiers"].items()))
        print(f"{mode}: {counts['programs']} programs, {tiers or 'none classified'}; "
              f"{counts['compared']} run in both cuts, {counts['unsound']} tier A/B divergent, "
              f"{counts['tier_c_agree']} tier C agreed")
        for reason, n in sorted(counts["reasons"].items(), key=lambda kv: -kv[1])[:8]:
            print(f"    {n:6}  {reason}")
        for skip, n in sorted(counts["skips"].items(), key=lambda kv: -kv[1])[:5]:
            print(f"    skip {n:5}  {skip[:100]}")
        if counts["unsound"]:
            status = 1
    if a.report:
        with open(a.report, "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"report: {a.report}")
    if status == 0 and not any(report[m]["compared"] for m in report):
        print("no program ran in both cuts: nothing was compared", file=sys.stderr)
        return 2
    return status


if __name__ == "__main__":
    sys.exit(main())
