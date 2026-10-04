"""Reproducible, solver-verified teaching animation for the English report.

Run ``python -m analysis.notebook_animation`` from the repository root.
Only presentation assets are written; no experimental observations are changed.
The six-item counterexample is deliberately crafted, not sampled test evidence.
"""
from __future__ import annotations

import argparse
import base64
from functools import cmp_to_key
import hashlib
import html
import io
import json
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "assets" / "notebook"
INSTANCE = {
    "instance_id": "crafted-six-item-teaching-example",
    "weights": [6, 5, 5, 3, 2, 1],
    "values": [13, 10, 10, 4, 2, 1],
    "capacity": 10,
    "item_order": [0, 1, 2, 3, 4, 5],
    "n": 6,
}
LABELS = "ABCDEF"
WIDTH, HEIGHT = 1400, 980
PAPER, INK, MUTED, BORDER = "#F7F8FA", "#23384D", "#627385", "#DCE4EA"
ACCENTS = ["#667789", "#23384D", "#C57429", "#12877F"]
ITEM_COLORS = ["#526E91", "#109F91", "#3CA9B5", "#C78C4E", "#967CA7", "#7B8892"]
PALES = ["#E8EDF3", "#DCF2EC", "#DFF3F5", "#F7ECD9", "#F0E8F3", "#EDF0F2"]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _solution_dict(solution):
    return dict(selected_indices=list(solution.selected_indices), value=solution.value,
                weight_used=solution.weight_used)


def collect_trace(bundle: Path = BUNDLE) -> dict:
    """Replay real decisions and assert equality with production solver outputs."""
    # Solver-generation dependencies are unnecessary when only showing saved media.
    import torch
    from experiment.baselines.methods import greedy_solution, random_solution
    from experiment.eval.reference import ratio_order
    from experiment.rl.environment import KnapsackEnv
    from experiment.rl.model import select_action, solve_instances
    from experiment.rl.training import verify_solution
    from portfolio.bandit import Bandit, context_key
    from portfolio.run import decide, load_network
    from portfolio.strategies import _rank_compare, solve_strategy

    instance = dict(INSTANCE)
    torch.set_num_threads(1)
    network = load_network(bundle / "dqn_seed0.pt")
    saved = json.loads((bundle / "selector_seed1.json").read_text(encoding="utf-8"))
    bandit = Bandit.from_dict(saved["model"])
    key = context_key(instance, saved["config"]["context_bins"])
    selector_solution, arm = decide(instance, bandit, saved["config"], {"dqn_seed0": network})
    solutions = {
        "random": random_solution(instance, 0),
        "greedy": greedy_solution(instance),
        "dqn": solve_instances(network, [instance])[0],
        "selector": selector_solution,
    }
    for solution in solutions.values():
        verify_solution(instance, solution)

    traces = {}
    rng = random.Random(0)
    for name, order in (("random", instance["item_order"]),
                        ("greedy", ratio_order(instance["weights"], instance["values"])),
                        ("dqn", instance["item_order"])):
        remaining, chosen, events = instance["capacity"], [], []
        env = KnapsackEnv(instance) if name == "dqn" else None
        for index in order:
            feasible = instance["weights"][index] <= remaining
            coin = None
            if name == "random":
                coin = rng.random() if feasible else None
                take = feasible and coin < .5
                reason = "coin: take" if take else "coin: skip" if feasible else "does not fit"
            elif name == "greedy":
                take, reason = feasible, "fits: take" if feasible else "does not fit"
            else:
                take = bool(select_action(network, env.state, 0, None))
                reason = "policy: take" if take else "policy: skip" if feasible else "masked: does not fit"
                env.step(int(take))
            before = list(chosen)
            if take:
                chosen.append(index)
                remaining -= instance["weights"][index]
            events.append(dict(item=index, take=take, reason=reason, feasible=feasible,
                               coin=coin, selected_before=before, selected=list(chosen),
                               remaining_capacity=remaining,
                               value=sum(instance["values"][i] for i in chosen)))
        if tuple(sorted(chosen)) != solutions[name].selected_indices:
            raise AssertionError(f"Animated trace differs from {name} solver")
        traces[name] = events

    # Reconstruct ONE genuine winning repair trial, not an invented sequence of
    # bandit actions. Other candidates are deliberately omitted from playback.
    if arm != "swap_repair8":
        raise AssertionError("The saved selector no longer chooses the illustrated strategy")
    base = solutions["greedy"]
    selected = set(base.selected_indices)
    value_order = sorted(range(6), key=lambda i: (-instance["values"][i], instance["weights"][i], i))
    candidates = [i for i in value_order if i not in selected][:8]
    candidate = candidates[0]
    removals = sorted(selected, key=cmp_to_key(_rank_compare(instance, ascending=True)))
    retained = set(selected)
    remaining = instance["capacity"] - base.weight_used
    removed = []
    for index in removals:
        if instance["weights"][candidate] <= remaining:
            break
        retained.remove(index)
        removed.append(index)
        remaining += instance["weights"][index]
    after_removals = sorted(retained)
    retained.add(candidate)
    remaining -= instance["weights"][candidate]
    filled = []
    for index in ratio_order(instance["weights"], instance["values"]):
        if index not in retained and instance["weights"][index] <= remaining:
            retained.add(index)
            filled.append(index)
            remaining -= instance["weights"][index]
    if tuple(sorted(retained)) != selector_solution.selected_indices:
        raise AssertionError("Shown repair trial is no longer a winning solver alternative")
    if solve_strategy(arm, instance) != selector_solution:
        raise AssertionError("Selector output differs from its named strategy")

    exact_value = max(sum(instance["values"][i] for i in range(6) if bits >> i & 1)
                      for bits in range(1 << 6)
                      if sum(instance["weights"][i] for i in range(6) if bits >> i & 1) <= 10)
    return dict(
        schema_version=1, instance=instance, random_seed=0, dqn_seed=0, selector_seed=1,
        status="illustration_only_not_test_evidence",
        caveats=["Deliberately crafted teaching counterexample, not representative performance evidence.",
                 "Six items is below the models' training sizes of 20, 50, 100, and 200.",
                 "The animation's duration is unrelated to solver runtime.",
                 "Selector chooses one strategy; the illustrated repair stages are algorithmic, not RL item actions.",
                 "Only one winning repair candidate is visualized; the solver evaluates other candidates too."],
        checkpoints={name: _sha(bundle / name) for name in ("dqn_seed0.pt", "selector_seed1.json")},
        source_sha256={name: _sha(ROOT / name) for name in (
            "experiment/baselines/methods.py", "experiment/rl/model.py",
            "experiment/rl/environment.py", "portfolio/bandit.py", "portfolio/strategies.py")},
        selector=dict(context_key=list(key), chosen_arm=arm,
                      estimated_rewards=dict(zip(bandit.arms, bandit.estimates(key).tolist()))),
        solutions={name: _solution_dict(solution) for name, solution in solutions.items()},
        traces=traces, exact_optimum=exact_value,
        repair=dict(base=list(base.selected_indices), candidate=candidate, removed=removed,
                    after_removals=after_removals, refill=filled, final=sorted(retained),
                    all_candidate_items=candidates, caption="One winning repair trial; other alternatives omitted."),
    )


def _font(size, bold=False):
    from matplotlib import font_manager
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, size)


FONTS = {}


def _text(draw, xy, text, size=20, color=INK, bold=False, anchor=None):
    key = size, bold
    if key not in FONTS:
        FONTS[key] = _font(*key)
    draw.text(xy, str(text), font=FONTS[key], fill=color, anchor=anchor)


def _round(draw, box, fill, outline=None, radius=14, width=1):
    draw.rounded_rectangle(box, radius, fill=fill, outline=outline, width=width)


def _pack(draw, selected, y, accent, *, unknown=False):
    used = sum(INSTANCE["weights"][i] for i in selected)
    value = sum(INSTANCE["values"][i] for i in selected)
    x, top, unit = 803, y + 47, 37
    _text(draw, (x, y + 14), "BACKPACK", 16, MUTED, True)
    _text(draw, (x + 370, y + 14), f"{used} / 10" if not unknown else "capacity 10", 18, MUTED, anchor="ra")
    _round(draw, (x, top, x + 370, top + 40), "#F0F3F6", BORDER, 9)
    for number in range(1, 10):
        draw.line((x + unit * number, top + 1, x + unit * number, top + 39), fill=BORDER)
    cursor = x
    for index in selected:
        width = unit * INSTANCE["weights"][index]
        _round(draw, (cursor, top, cursor + width - 2, top + 40), ITEM_COLORS[index], radius=6)
        _text(draw, (cursor + width / 2 - 1, top + 20), LABELS[index], 20, "white", True, "mm")
        cursor += width
    _text(draw, (x, y + 98), " + ".join(LABELS[i] for i in selected) if selected else "waiting for solver" if unknown else "empty", 19, accent)
    draw.line((1202, y + 17, 1202, y + 117), fill=BORDER, width=1)
    _text(draw, (1284, y + 15), "VALUE", 16, MUTED, True, "ma")
    _text(draw, (1284, y + 67), "—" if unknown else value, 44, accent, True, "mm")
    if not unknown:
        _text(draw, (1284, y + 103), f"{10 - used} space left", 15, MUTED, anchor="mm")


def _tile(draw, index, x, y, *, taken=None, active=False, accent=INK, membership=False):
    color = PALES[index] if taken else "#F1F4F7"
    _round(draw, (x, y, x + 70, y + 62), color,
           accent if active else ITEM_COLORS[index] if taken else BORDER,
           radius=10, width=3 if active else 1)
    _text(draw, (x + 35, y + 20), LABELS[index], 23, ITEM_COLORS[index] if taken else MUTED, True, "mm")
    label = ("IN" if taken else "OUT") if membership else "TAKE" if taken else "SKIP" if taken is False else "·"
    _text(draw, (x + 35, y + 48), label, 12,
          accent if taken else MUTED, True, "mm")


def render_frame(trace, phase, scan=0, progress=1.0):
    """Render a real state; progress animates only an accepted item's transit."""
    image = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    draw = ImageDraw.Draw(image)
    _text(draw, (32, 23), "Same backpack. Four different decisions.", 33, bold=True)
    _text(draw, (33, 73), "Capacity 10   /   six indivisible items   /   maximize total value", 21, MUTED)
    _round(draw, (1139, 28, 1370, 61), "#E4ECEF", radius=16)
    _text(draw, (1254, 44), "TEACHING EXAMPLE", 15, INK, True, "mm")
    _text(draw, (36, 124), "ITEM MENU", 18, MUTED, True)
    _text(draw, (36, 156), "Weight uses space.", 18, MUTED)
    _text(draw, (36, 183), "Value is the payoff.", 18, MUTED)
    for i, (weight, value) in enumerate(zip(INSTANCE["weights"], INSTANCE["values"])):
        x = 253 + 187 * i
        _round(draw, (x, 113, x + 170, 214), "white", BORDER)
        _round(draw, (x + 12, 125, x + 44, 157), ITEM_COLORS[i], radius=8)
        _text(draw, (x + 28, 141), LABELS[i], 21, "white", True, "mm")
        _text(draw, (x + 59, 127), f"weight {weight}", 19)
        _text(draw, (x + 14, 168), f"value {value}", 23, ITEM_COLORS[i], True)

    captions = {
        "intro": ("01", "One shared constraint", "Every selected item must fit; an item cannot be split."),
        "context": ("02", "The selector makes ONE strategy choice", "Input statistics → learned reward estimates → swap_repair8."),
        "scan": ("03", f"Scan pass {scan + 1} of 6", "Each lane follows its own order; Greedy breaks equal-density ties by lighter weight."),
        "base": ("04", "Chosen strategy: start from Greedy", "The repair solver keeps the Greedy pack as one candidate."),
        "remove": ("05", "One repair trial: make room for B", "Remove low-density items F, D, then A; the other trials are omitted."),
        "fill": ("06", "Insert B, then refill with C", "This real repair trial reaches value 20; the solver compares its candidate packs."),
        "final": ("07", "Compare the returned packs", "On this crafted case, B + C is optimal: exhaustive search checks all 64 subsets."),
    }
    number, title, subtitle = captions[phase]
    _round(draw, (31, 230, 90, 282), INK, radius=13)
    _text(draw, (61, 256), number, 25, "white", True, "mm")
    _text(draw, (110, 227), title, 24, bold=True)
    _text(draw, (110, 261), subtitle, 18, MUTED)

    titles = ["Random", "Greedy", "Original DQN", "Learned selector"]
    mottos = [("Coin flip if feasible", "one trial · seed 0"),
              ("High value / weight first", "take whenever it fits"),
              ("Learned include / skip", "frozen checkpoint · seed 0"),
              ("Choose a whole strategy", "saved bandit · seed 1")]
    after_scan = phase in ("base", "remove", "fill", "final")
    for lane, name in enumerate(("random", "greedy", "dqn", "selector")):
        y, accent = 301 + 148 * lane, ACCENTS[lane]
        _round(draw, (30, y, 1370, y + 135), "white", BORDER)
        _round(draw, (30, y + 14, 35, y + 120), accent, radius=2)
        _text(draw, (49, y + 17), titles[lane], 20 if lane == 3 else 25, accent, True)
        for line, message in enumerate(mottos[lane]):
            _text(draw, (49, y + 61 + 25 * line), message, 16, MUTED)
        if lane < 3:
            events = trace["traces"][name]
            completed = 6 if after_scan else scan + (progress >= 1) if phase == "scan" else 0
            selected = events[completed - 1]["selected"] if completed else []
            current = events[scan] if phase == "scan" else None
            for position, event in enumerate(events):
                status = event["take"] if position < completed else None
                _tile(draw, event["item"], 258 + 82 * position, y + 17, taken=status,
                      active=phase == "scan" and position == scan, accent=accent)
            if current:
                label = LABELS[current["item"]]
                message = f"{label}  ·  {current['reason']}"
                if name == "dqn" and scan == 5:
                    message += " (even though F fits)"
            else:
                message = "ready to scan" if not after_scan else "returned solution"
            _text(draw, (258, y + 98), message, 18, accent)
            _pack(draw, selected, y, accent)
            if current and current["take"] and 0 < progress < 1:
                start = (258 + 82 * scan + 35, y + 47)
                prior_used = sum(INSTANCE["weights"][i] for i in current["selected_before"])
                end = (803 + prior_used * 37 + INSTANCE["weights"][current["item"]] * 18.5, y + 67)
                t = progress * progress * (3 - 2 * progress)
                x = start[0] + (end[0] - start[0]) * t
                top = start[1] + (end[1] - start[1]) * t - 32 * 4 * t * (1 - t)
                _round(draw, (x - 25, top - 21, x + 25, top + 21), ITEM_COLORS[current["item"]], radius=9)
                _text(draw, (x, top), LABELS[current["item"]], 24, "white", True, "mm")
        else:
            known = phase in ("base", "remove", "fill", "final")
            selected = trace["repair"]["base"] if phase == "base" else trace["repair"]["after_removals"] if phase == "remove" else trace["repair"]["final"] if phase in ("fill", "final") else []
            if known:
                for index in range(6):
                    _tile(draw, index, 258 + 82 * index, y + 17, taken=index in selected, accent=accent, membership=True)
                message = "Greedy candidate retained" if phase == "base" else "trial B: remove F → D → A" if phase == "remove" else "swap_repair8 returns B + C"
                _text(draw, (258, y + 98), message, 18, accent)
            else:
                _round(draw, (258, y + 16, 443, y + 81), "#EBF5F3", radius=10)
                _text(draw, (350, y + 34), "CONTEXT", 14, accent, True, "mm")
                _text(draw, (350, y + 61), "6 items · C/Σw = .45", 15, INK, anchor="mm")
                _text(draw, (465, y + 48), "→", 30, accent, anchor="mm")
                _round(draw, (489, y + 16, 740, y + 81), "#EBF5F3", accent if phase != "intro" else BORDER, radius=10)
                _text(draw, (614, y + 34), "CHOSEN STRATEGY", 14, accent, True, "mm")
                _text(draw, (614, y + 61), "swap_repair8" if phase != "intro" else "…", 23, accent, True, "mm")
                _text(draw, (258, y + 98), "Strategy execution shown next" if phase != "intro" else "Input statistics; no item decisions", 18, accent)
            _pack(draw, selected, y, accent, unknown=not known)

    _text(draw, (34, 911), "WHY THIS CASE?", 16, INK, True)
    _text(draw, (205, 908), "A has the best density, but taking A blocks the better pair B + C.", 20, INK)
    _text(draw, (34, 949), "Crafted illustration, not test evidence. n = 6 is below training sizes. Repair stages are condensed. Playback is not runtime.", 16, MUTED)
    return image


def _png_uri(image):
    stream = io.BytesIO()
    image.quantize(colors=128, method=Image.Quantize.FASTOCTREE).save(stream, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def _player_html(images, captions):
    frames = json.dumps([_png_uri(image) for image in images])
    descriptions = json.dumps(captions)
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Knapsack method walkthrough</title>
<style>
*{box-sizing:border-box}body{margin:0;padding:12px;background:#f7f8fa;color:#23384d;font-family:system-ui,sans-serif}
main{max-width:1400px;margin:auto}img{display:block;width:100%;height:auto;border-radius:12px}
#scene{max-height:calc(100vh - 170px);object-fit:contain}
.controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:13px 4px 4px}
button,select{font:inherit;font-size:15px;border:1px solid #cad6df;border-radius:7px;background:white;padding:8px 14px;color:#23384d;cursor:pointer}
button:hover{background:#e7f2ef}button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #12877f}
#play{background:#12877f;color:white;border-color:#12877f;min-width:78px}#position{flex:1;min-width:130px;accent-color:#12877f}
#step{font-size:15px;min-width:80px}#caption{margin:10px 4px 4px;font-size:16px;line-height:1.4}
.note{font-size:13px;color:#627385;margin:5px 4px}label{font-size:14px}noscript{font-size:16px}
</style></head><body><main>
<img id="scene" alt="Four solver lanes showing exact item decisions and a condensed strategy repair." src="__FIRST__">
<div class="controls"><button id="play" type="button">Play</button><button id="previous" type="button" aria-label="Previous step">← Previous</button><button id="next" type="button" aria-label="Next step">Next →</button><button id="replay" type="button">Replay</button>
<input id="position" type="range" min="0" max="11" value="0" aria-label="Animation step"><span id="step">1 / 12</span>
<label for="speed">Speed</label><select id="speed"><option value="3000">Slow</option><option value="1800" selected>Normal</option><option value="900">Fast</option></select></div>
<p id="caption" aria-live="polite"></p><p class="note">Step through the actual solver trace. The selector chooses a strategy once; the final three stages condense one real repair trial.</p>
<noscript>JavaScript is disabled. Open method_animation.gif for playback or method_storyboard.png for the final comparison.</noscript>
</main><script>
const frames=__FRAMES__, descriptions=__CAPTIONS__;
let current=0,timer=null; const byId=id=>document.getElementById(id);
function draw(){byId('scene').src=frames[current];byId('position').value=current;byId('step').textContent=(current+1)+' / '+frames.length;byId('caption').textContent=descriptions[current];}
function pause(){if(timer!==null)clearInterval(timer);timer=null;byId('play').textContent='Play';}
function play(){pause();if(current===frames.length-1)current=0;draw();byId('play').textContent='Pause';timer=setInterval(()=>{if(current===frames.length-1){pause();return;}current++;draw();},Number(byId('speed').value));}
function step(delta){pause();current=Math.max(0,Math.min(frames.length-1,current+delta));draw();}
byId('play').onclick=()=>timer===null?play():pause();byId('previous').onclick=()=>step(-1);byId('next').onclick=()=>step(1);
byId('replay').onclick=()=>{current=0;play();};byId('position').oninput=e=>{pause();current=Number(e.target.value);draw();};
byId('speed').onchange=()=>{if(timer!==null)play();};draw();
</script></body></html>""".replace("__FIRST__", _png_uri(images[0])).replace("__FRAMES__", frames).replace("__CAPTIONS__", descriptions)


def animation_html(bundle: Path = BUNDLE, height: int = 890) -> str:
    """Return an offline iframe player plus embedded GIF/static fallback links.

    Notebook usage: ``display(HTML(animation_html(BUNDLE)))``. All media are data
    URIs, so exported HTML remains playable without the repository or a server.
    """
    bundle = Path(bundle)
    document = (bundle / "method_animation.html").read_text(encoding="utf-8")
    gif = base64.b64encode((bundle / "method_animation.gif").read_bytes()).decode("ascii")
    png = base64.b64encode((bundle / "method_storyboard.png").read_bytes()).decode("ascii")
    return (f'<iframe title="Interactive knapsack method walkthrough" sandbox="allow-scripts" '
            f'srcdoc="{html.escape(document, quote=True)}" style="width:100%;height:min({int(height)}px,94vh);min-height:540px;border:0;" '
            'loading="lazy"></iframe>'
            '<details><summary>GIF playback fallback</summary>'
            f'<img alt="Animated knapsack solver decisions" style="max-width:100%;height:auto" src="data:image/gif;base64,{gif}"></details>'
            '<details><summary>Static final comparison</summary>'
            f'<img alt="Final solver packs and total values" style="max-width:100%;height:auto" src="data:image/png;base64,{png}"></details>')


def generate(bundle: Path = BUNDLE):
    bundle = Path(bundle)
    trace = collect_trace(bundle)
    frames, durations = [], []
    keyframes, captions = [], []

    def add(phase, duration, scan=0, progress=1, key=False, caption=""):
        frame = render_frame(trace, phase, scan, progress)
        frames.append(frame)
        durations.append(duration)
        if key:
            keyframes.append(frame.resize((1120, 784), Image.Resampling.LANCZOS))
            captions.append(caption)

    add("intro", 1000, key=True, caption="Start: six items share one capacity of 10. Each item can be selected at most once.")
    add("context", 1500, key=True, caption="The saved contextual bandit selects swap_repair8 using all four input statistics. This is its one RL action.")
    for scan in range(6):
        add("scan", 220, scan, 0)
        for progress in (.2, .4, .6, .8):
            add("scan", 110, scan, progress)
        add("scan", 640, scan, 1, True,
            f"Scan pass {scan + 1}: " + " / ".join(
                f"{name.upper() if name == 'dqn' else name.capitalize()} considers {LABELS[trace['traces'][name][scan]['item']]}: {trace['traces'][name][scan]['reason']}"
                for name in ("random", "greedy", "dqn")))
    add("base", 1200, key=True, caption="Repair, condensed: begin from A + D + F, the genuine Greedy pack. The solver retains this baseline candidate.")
    add("remove", 1200, key=True, caption="One winning trial: force candidate B. Remove F, D, then A in increasing density order to make room.")
    add("fill", 1200, key=True, caption="Insert B and refill in density order with C. The result is a genuine candidate returned by swap_repair8; other trials are omitted.")
    add("final", 4500, key=True, caption="Final values: Random 15; Greedy 18; original DQN 17; learned selector 20. B + C is the exact optimum on this crafted example, not a benchmark result.")
    gif_path = bundle / "method_animation.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True, disposal=1)
    frames[-1].save(bundle / "method_storyboard.png", optimize=True)
    (bundle / "method_animation.html").write_text(_player_html(keyframes, captions), encoding="utf-8")
    with Image.open(gif_path) as saved_gif:
        actual_duration = 0
        for index in range(saved_gif.n_frames):
            saved_gif.seek(index)
            actual_duration += saved_gif.info["duration"]
        if actual_duration != sum(durations):
            raise AssertionError("GIF did not preserve the intended playback duration")
        trace["animation"] = dict(width=WIDTH, height=HEIGHT, gif_frames=saved_gif.n_frames,
                                  duration_ms=actual_duration, player_keyframes=len(keyframes))
    trace["outputs"] = {name: dict(bytes=(bundle / name).stat().st_size, sha256=_sha(bundle / name))
                        for name in ("method_animation.gif", "method_animation.html", "method_storyboard.png")}
    (bundle / "method_animation.json").write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(solutions=trace["solutions"], selector=trace["selector"]["chosen_arm"],
                         animation=trace["animation"], outputs=trace["outputs"]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    generate(parser.parse_args().bundle)
