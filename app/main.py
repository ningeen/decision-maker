from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import List, Optional

import os
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

from nicegui import ui
import nicegui.run as ng_run

from mcda import get_method, list_methods
from mcda.core import MethodContext
from models import Project, Result
from storage import list_projects, load_project, save_project


@dataclass
class AppState:
    project: Project


state = AppState(project=Project(name="default"))
method_heading: Optional[ui.label] = None
method_hint_primary: Optional[ui.label] = None
method_hint_secondary: Optional[ui.label] = None


def resize_square_matrix(matrix: List[List[float]], size: int, default: float = 1.0) -> List[List[float]]:
    if size <= 0:
        return []
    new_matrix: List[List[float]] = []
    for i in range(size):
        row: List[float] = []
        for j in range(size):
            if i < len(matrix) and j < len(matrix[i]):
                value = matrix[i][j]
            else:
                value = 1.0 if i == j else default
            row.append(value)
        new_matrix.append(row)
    return new_matrix


def resize_scores(scores: List[List[float]], options: int, criteria: int) -> List[List[float]]:
    new_scores: List[List[float]] = []
    for i in range(options):
        row = scores[i] if i < len(scores) else []
        new_row: List[float] = []
        for j in range(criteria):
            if j < len(row):
                new_row.append(row[j])
            else:
                new_row.append(0.0)
        new_scores.append(new_row)
    return new_scores


def resize_list(values: List, size: int, default) -> List:
    resized = list(values[:size])
    while len(resized) < size:
        resized.append(default)
    return resized


def ensure_bwm_state() -> None:
    project = state.project
    n_items = len(project.criteria)
    if n_items == 0:
        project.bwm_best_index = None
        project.bwm_worst_index = None
        project.bwm_best_to_others = []
        project.bwm_others_to_worst = []
        return

    project.bwm_best_to_others = resize_list(project.bwm_best_to_others, n_items, 1.0)
    project.bwm_others_to_worst = resize_list(project.bwm_others_to_worst, n_items, 1.0)

    if project.bwm_best_index is None or not (0 <= project.bwm_best_index < n_items):
        project.bwm_best_index = 0
    if project.bwm_worst_index is None or not (0 <= project.bwm_worst_index < n_items):
        project.bwm_worst_index = n_items - 1 if n_items > 1 else 0
    if n_items > 1 and project.bwm_best_index == project.bwm_worst_index:
        project.bwm_worst_index = n_items - 1 if project.bwm_best_index != n_items - 1 else 0

    project.bwm_best_to_others = [max(1.0, float(value)) for value in project.bwm_best_to_others]
    project.bwm_others_to_worst = [max(1.0, float(value)) for value in project.bwm_others_to_worst]
    project.bwm_best_to_others[project.bwm_best_index] = 1.0
    project.bwm_others_to_worst[project.bwm_worst_index] = 1.0


UI_SCALE_TO_SAATY = {
    1: 2.0,
    2: 3.0,
    3: 5.0,
    4: 7.0,
    5: 9.0,
}

SAATY_VALUES = list(UI_SCALE_TO_SAATY.values())
CONSISTENCY_THRESHOLD = 0.1
PROMETHEE_FUNCTIONS = {
    "t1": "t1 Usual (no thresholds)",
    "t2": "t2 U-shape (Q)",
    "t3": "t3 V-shape (P)",
    "t4": "t4 Level (Q, P)",
    "t5": "t5 Linear (Q, P)",
    "t6": "t6 Gaussian (S)",
    "t7": "t7 Quasi (S)",
}
PROMETHEE_DIRECTIONS = {
    "max": "Maximize (benefit)",
    "min": "Minimize (cost)",
}


def ahp_ratio_from_ui(value: float) -> float:
    if value == 0:
        return 1.0
    sign = 1 if value > 0 else -1
    magnitude = int(round(abs(value)))
    magnitude = min(5, max(1, magnitude))
    ratio = UI_SCALE_TO_SAATY[magnitude]
    return ratio if sign > 0 else 1.0 / ratio


def ui_value_from_ratio(ratio: float) -> int:
    if ratio <= 0:
        return 0
    if abs(ratio - 1.0) < 1e-9:
        return 0
    sign = 1
    value = ratio
    if ratio < 1.0:
        sign = -1
        value = 1.0 / ratio
    closest = min(SAATY_VALUES, key=lambda candidate: abs(candidate - value))
    ui_value = next(key for key, mapped in UI_SCALE_TO_SAATY.items() if mapped == closest)
    return sign * ui_value


def max_pairwise_discrepancy(criteria: List[str], pairwise: List[List[float]], weights: List[float]):
    if not criteria or len(criteria) != len(weights):
        return None
    best = None
    for i in range(len(criteria)):
        for j in range(i + 1, len(criteria)):
            actual = pairwise[i][j]
            if actual <= 0 or weights[j] == 0:
                continue
            expected = weights[i] / weights[j]
            discrepancy = abs(math.log(actual / expected))
            if best is None or discrepancy > best[0]:
                best = (discrepancy, i, j, actual, expected)
    return best


def ensure_state_consistency() -> None:
    project = state.project
    project.pairwise = resize_square_matrix(project.pairwise, len(project.criteria), default=1.0)
    project.weights = project.weights[: len(project.criteria)]
    if len(project.weights) != len(project.criteria):
        project.weights = [1.0 / len(project.criteria)] * len(project.criteria) if project.criteria else []
    project.promethee_weights = resize_list(project.promethee_weights, len(project.criteria), 1.0)
    project.promethee_functions = resize_list(project.promethee_functions, len(project.criteria), "t1")
    project.promethee_q = resize_list(project.promethee_q, len(project.criteria), 0.0)
    project.promethee_p = resize_list(project.promethee_p, len(project.criteria), 0.0)
    project.promethee_s = resize_list(project.promethee_s, len(project.criteria), 0.0)
    project.promethee_directions = resize_list(project.promethee_directions, len(project.criteria), "max")
    ensure_bwm_state()
    project.scores = resize_scores(project.scores, len(project.options), len(project.criteria))


def add_criterion(name: str) -> None:
    name = name.strip()
    if not name:
        ui.notify("Criterion name cannot be empty.")
        return
    if name in state.project.criteria:
        ui.notify("Criterion already exists.")
        return
    state.project.criteria.append(name)
    ensure_state_consistency()
    criteria_view.refresh()
    pairwise_view.refresh()
    bwm_view.refresh()
    promethee_view.refresh()
    weights_view.refresh()
    options_view.refresh()
    results_view.refresh()


def remove_criterion(index: int) -> None:
    state.project.criteria.pop(index)
    ensure_state_consistency()
    criteria_view.refresh()
    pairwise_view.refresh()
    bwm_view.refresh()
    promethee_view.refresh()
    weights_view.refresh()
    options_view.refresh()
    results_view.refresh()


def update_pairwise(i: int, j: int, value: float | None) -> None:
    if value is None:
        return
    if value < -5 or value > 5:
        ui.notify("Pairwise value must be between -5 and 5.")
        return
    ratio = ahp_ratio_from_ui(float(value))
    project = state.project
    project.pairwise[i][j] = ratio
    project.pairwise[j][i] = 1.0 / ratio
    pairwise_view.refresh()


def update_bwm_best(value: str | None) -> None:
    if not value:
        return
    criteria = state.project.criteria
    if value not in criteria:
        return
    state.project.bwm_best_index = criteria.index(value)
    ensure_state_consistency()
    bwm_view.refresh()


def update_bwm_worst(value: str | None) -> None:
    if not value:
        return
    criteria = state.project.criteria
    if value not in criteria:
        return
    state.project.bwm_worst_index = criteria.index(value)
    ensure_state_consistency()
    bwm_view.refresh()


def update_bwm_best_to_others(index: int, value: float | None) -> None:
    if value is None:
        return
    if value < 1 or value > 9:
        ui.notify("BWM values must be between 1 and 9.")
        return
    state.project.bwm_best_to_others[index] = float(value)


def update_bwm_others_to_worst(index: int, value: float | None) -> None:
    if value is None:
        return
    if value < 1 or value > 9:
        ui.notify("BWM values must be between 1 and 9.")
        return
    state.project.bwm_others_to_worst[index] = float(value)


def compute_weights() -> None:
    project = state.project
    if not project.criteria:
        ui.notify("Add at least one criterion to compute weights.")
        return
    if project.method_id in ("ahp", "bwm") and len(project.criteria) < 2:
        ui.notify("Add at least two criteria to compute weights.")
        return
    if project.method_id == "bwm":
        if project.bwm_best_index is None or project.bwm_worst_index is None:
            ui.notify("Select best and worst criteria before computing weights.")
            return
    method = get_method(project.method_id)
    try:
        context = build_context()
        result = method.compute_weights(project.criteria, project.pairwise, context=context)
    except ModuleNotFoundError as exc:
        ui.notify(f"Missing dependency: {exc.name}. Install it and retry.")
        return
    project.weights = result.weights
    project.consistency_ratio = result.consistency_ratio
    weights_view.refresh()
    results_view.refresh()


def update_method_hints() -> None:
    if method_heading is None or method_hint_primary is None or method_hint_secondary is None:
        return
    if state.project.method_id == "promethee_ii":
        method_heading.set_text("PROMETHEE II parameters")
        method_hint_primary.set_text(
            "Provide weights, preference functions, and thresholds for each criterion."
        )
        method_hint_secondary.set_text(
            "Scores are used directly; choose Minimize for cost criteria."
        )
    elif state.project.method_id == "bwm":
        method_heading.set_text("Best-Worst comparisons")
        method_hint_primary.set_text(
            "Pick the best and worst criteria, then rate best vs others and others vs worst (1-9)."
        )
        method_hint_secondary.set_text(
            "1 means equal importance; higher numbers mean stronger preference."
        )
    else:
        method_heading.set_text("Pairwise comparison (upper triangle)")
        method_hint_primary.set_text(
            "Use -5 to 5 scale (0 = equal importance). Positive means row is more important; negative means less important."
        )
        method_hint_secondary.set_text(
            "Internally this maps to the AHP 1/9-9 scale; lower triangle is filled automatically."
        )


def build_context() -> MethodContext:
    project = state.project
    return MethodContext(
        criteria=list(project.criteria),
        directions=list(project.promethee_directions),
        preference_functions=list(project.promethee_functions),
        q=list(project.promethee_q),
        p=list(project.promethee_p),
        s=list(project.promethee_s),
        weights_raw=list(project.promethee_weights),
        best_index=project.bwm_best_index,
        worst_index=project.bwm_worst_index,
        best_to_others=list(project.bwm_best_to_others),
        others_to_worst=list(project.bwm_others_to_worst),
    )


def add_option(name: str) -> None:
    name = name.strip()
    if not name:
        ui.notify("Option name cannot be empty.")
        return
    if name in state.project.options:
        ui.notify("Option already exists.")
        return
    state.project.options.append(name)
    ensure_state_consistency()
    options_view.refresh()
    results_view.refresh()


def remove_option(index: int) -> None:
    project = state.project
    project.options.pop(index)
    project.scores.pop(index)
    options_view.refresh()
    results_view.refresh()


def update_score(option_index: int, criterion_index: int, value: float | None) -> None:
    if value is None:
        return
    if value < 1 or value > 10:
        ui.notify("Score must be between 1 and 10.")
        return
    state.project.scores[option_index][criterion_index] = float(value)
    results_view.refresh()


def update_promethee_weight(index: int, value: float | None) -> None:
    if value is None:
        return
    if value < 0:
        ui.notify("Weight must be 0 or greater.")
        return
    state.project.promethee_weights[index] = float(value)
    weights_view.refresh()


def update_promethee_function(index: int, value: str | None) -> None:
    if not value:
        return
    state.project.promethee_functions[index] = value


def update_promethee_threshold(kind: str, index: int, value: float | None) -> None:
    if value is None:
        return
    if value < 0:
        ui.notify("Threshold must be 0 or greater.")
        return
    if kind == "q":
        state.project.promethee_q[index] = float(value)
    elif kind == "p":
        state.project.promethee_p[index] = float(value)
    elif kind == "s":
        state.project.promethee_s[index] = float(value)


def update_promethee_direction(index: int, value: str | None) -> None:
    if not value:
        return
    state.project.promethee_directions[index] = value


def recompute_results() -> None:
    project = state.project
    method = get_method(project.method_id)
    context = build_context()
    if project.method_id == "promethee_ii":
        project.weights = method.compute_weights(project.criteria, project.pairwise, context=context).weights
        weights_view.refresh()
    else:
        if not project.weights or len(project.weights) != len(project.criteria):
            ui.notify("Compute weights before scoring options.")
            return
    if not project.options:
        ui.notify("Add at least one option to score.")
        return
    option_scores = {
        option: project.scores[idx]
        for idx, option in enumerate(project.options)
    }
    try:
        ranked = method.compute_scores(project.weights, option_scores, context=context)
    except ModuleNotFoundError as exc:
        ui.notify(f"Missing dependency: {exc.name}. Install it and retry.")
        return
    project.results = [Result(option=name, score=score) for name, score in ranked]
    results_view.refresh()


def save_current() -> None:
    path = save_project(state.project)
    ui.notify(f"Saved to {path}")
    refresh_saved_projects()


def load_named(name: str) -> None:
    if not name:
        ui.notify("Choose a project to load.")
        return
    loaded = load_project(name)
    if loaded is None:
        ui.notify("Project not found on disk.")
        return
    state.project = loaded
    ensure_state_consistency()
    project_name_input.value = state.project.name
    method_select.value = state.project.method_id
    update_method_hints()
    criteria_view.refresh()
    pairwise_view.refresh()
    promethee_view.refresh()
    weights_view.refresh()
    weights_action.refresh()
    options_view.refresh()
    results_view.refresh()
    ui.notify(f"Loaded {state.project.name}")


def refresh_saved_projects() -> None:
    saved_select.options = list_projects()
    saved_select.update()


ui.page_title("Decision Helper")

with ui.column().classes("w-full max-w-6xl mx-auto p-6"):
    ui.label("Decision Helper").classes("text-3xl font-semibold")
    ui.label("Multi-criteria decision analysis with AHP, Best-Worst, or PROMETHEE II.").classes(
        "text-gray-500"
    )

    with ui.card().classes("w-full"):
        ui.label("Project").classes("text-lg font-semibold")
        with ui.row().classes("items-center"):
            project_name_input = ui.input("Project name", value=state.project.name)

            def on_project_name_change(event) -> None:
                state.project.name = (event.value or "").strip() or "Untitled"

            project_name_input.on_value_change(on_project_name_change)
            ui.button("Save", on_click=save_current)

            method_options = list_methods()

            def on_method_change(event) -> None:
                state.project.method_id = event.value
                state.project.consistency_ratio = None
                state.project.weights = []
                state.project.results = []
                ensure_state_consistency()
                update_method_hints()
                weights_view.refresh()
                results_view.refresh()
                weights_action.refresh()
                pairwise_view.refresh()
                bwm_view.refresh()
                promethee_view.refresh()

            method_select = ui.select(
                options=method_options,
                value=state.project.method_id,
                label="Weighting method",
                on_change=on_method_change,
            )

        with ui.row().classes("items-center"):
            saved_select = ui.select(options=list_projects(), label="Saved projects")
            ui.button("Load", on_click=lambda: load_named(saved_select.value))
            ui.button("Refresh list", on_click=refresh_saved_projects)

    with ui.stepper().classes("w-full") as stepper:
        with ui.step("1. Criteria"):
            with ui.card().classes("w-full"):
                ui.label("Add criteria").classes("text-lg font-semibold")
                with ui.row().classes("items-center"):
                    criterion_input = ui.input("Criterion name")

                    def submit_criterion() -> None:
                        add_criterion(criterion_input.value)
                        criterion_input.set_value("")

                    criterion_input.on("keydown.enter", lambda: submit_criterion())
                    ui.button("Add", on_click=submit_criterion)

                @ui.refreshable
                def criteria_view() -> None:
                    if not state.project.criteria:
                        ui.label("No criteria yet.").classes("text-gray-500")
                        return
                    with ui.column().classes("gap-2"):
                        for idx, criterion in enumerate(state.project.criteria):
                            with ui.row().classes("items-center justify-between"):
                                ui.label(criterion)
                                ui.button("Remove", on_click=lambda i=idx: remove_criterion(i)).props("outline color=negative")

                criteria_view()

                with ui.stepper_navigation():
                    ui.button("Next", on_click=stepper.next)

        with ui.step("2. Weights"):
            with ui.card().classes("w-full"):
                method_heading = ui.label("").classes("text-lg font-semibold")
                method_hint_primary = ui.label("").classes("text-gray-500 text-sm")
                method_hint_secondary = ui.label("").classes("text-gray-500 text-sm")
                update_method_hints()

                @ui.refreshable
                def pairwise_view() -> None:
                    if state.project.method_id != "ahp":
                        return
                    criteria = state.project.criteria
                    if len(criteria) < 2:
                        ui.label("Add at least two criteria.").classes("text-gray-500")
                        return
                    grid_template = f"grid-template-columns: 160px repeat({len(criteria)}, 120px);"
                    min_width = 160 + (len(criteria) * 120)
                    with ui.element("div").classes("w-full overflow-x-auto").style("max-width: 100%;"):
                        with ui.column().classes("gap-2"):
                            with ui.element("div").style(
                                f"display: grid; {grid_template} align-items: center; gap: 12px; min-width: {min_width}px;"
                            ):
                                ui.label("")
                                for criterion in criteria:
                                    ui.label(criterion).classes("text-center").style("justify-self: center;")

                            for i, row_name in enumerate(criteria):
                                with ui.element("div").style(
                                    f"display: grid; {grid_template} align-items: center; gap: 12px; min-width: {min_width}px;"
                                ):
                                    ui.label(row_name)
                                    for j in range(len(criteria)):
                                        if i == j:
                                            ui.label("0").classes("text-center").style("justify-self: center;")
                                        elif i < j:
                                            value = ui_value_from_ratio(state.project.pairwise[i][j])
                                            ui.number(
                                                value=value,
                                                min=-5,
                                                max=5,
                                                step=1,
                                                format="%d",
                                                on_change=lambda e, ii=i, jj=j: update_pairwise(ii, jj, e.value),
                                            ).classes("w-full").props('input-class="text-center" dense')
                                        else:
                                            ui_value = ui_value_from_ratio(state.project.pairwise[i][j])
                                            ui.label(f"{ui_value}").classes("text-center text-gray-500").style(
                                                "justify-self: center;"
                                            )

                pairwise_view()

                @ui.refreshable
                def bwm_view() -> None:
                    if state.project.method_id != "bwm":
                        return
                    criteria = state.project.criteria
                    if len(criteria) < 2:
                        ui.label("Add at least two criteria.").classes("text-gray-500")
                        return
                    best_idx = (
                        state.project.bwm_best_index
                        if state.project.bwm_best_index is not None
                        else 0
                    )
                    worst_idx = (
                        state.project.bwm_worst_index
                        if state.project.bwm_worst_index is not None
                        else (len(criteria) - 1)
                    )

                    with ui.row().classes("items-center gap-4"):
                        ui.select(
                            options=criteria,
                            value=criteria[best_idx],
                            label="Best criterion",
                            on_change=lambda e: update_bwm_best(e.value),
                        ).classes("w-64")
                        ui.select(
                            options=criteria,
                            value=criteria[worst_idx],
                            label="Worst criterion",
                            on_change=lambda e: update_bwm_worst(e.value),
                        ).classes("w-64")

                    grid_template = "grid-template-columns: 160px 200px 200px;"
                    min_width = 160 + 200 + 200

                    with ui.element("div").classes("w-full overflow-x-auto").style("max-width: 100%;"):
                        with ui.element("div").style(
                            f"display: grid; {grid_template} align-items: center; gap: 12px; min-width: {min_width}px;"
                        ):
                            ui.label("Criterion")
                            ui.label("Best vs criterion")
                            ui.label("Criterion vs worst")

                        for idx, criterion in enumerate(criteria):
                            with ui.element("div").style(
                                f"display: grid; {grid_template} align-items: center; gap: 12px; min-width: {min_width}px;"
                            ):
                                ui.label(criterion)
                                if idx == best_idx:
                                    ui.label("1").classes("text-center text-gray-500").style("justify-self: center;")
                                else:
                                    ui.number(
                                        value=state.project.bwm_best_to_others[idx],
                                        min=1,
                                        max=9,
                                        step=1,
                                        format="%d",
                                        on_change=lambda e, i=idx: update_bwm_best_to_others(i, e.value),
                                    ).classes("w-full").props('input-class="text-center" dense')
                                if idx == worst_idx:
                                    ui.label("1").classes("text-center text-gray-500").style("justify-self: center;")
                                else:
                                    ui.number(
                                        value=state.project.bwm_others_to_worst[idx],
                                        min=1,
                                        max=9,
                                        step=1,
                                        format="%d",
                                        on_change=lambda e, i=idx: update_bwm_others_to_worst(i, e.value),
                                    ).classes("w-full").props('input-class="text-center" dense')

                    ui.label("Use the 1-9 scale (1 = equal importance, 9 = extremely more important).").classes(
                        "text-sm text-gray-500 mt-2"
                    )

                bwm_view()

                @ui.refreshable
                def promethee_view() -> None:
                    if state.project.method_id != "promethee_ii":
                        return
                    criteria = state.project.criteria
                    if not criteria:
                        ui.label("Add criteria to configure PROMETHEE II.").classes("text-gray-500")
                        return

                    grid_template = "grid-template-columns: 160px 190px 110px 200px 90px 90px 90px;"
                    min_width = 160 + 190 + 110 + 200 + 90 + 90 + 90

                    with ui.element("div").classes("w-full overflow-x-auto").style("max-width: 100%;"):
                        with ui.element("div").style(
                            f"display: grid; {grid_template} align-items: center; gap: 12px; min-width: {min_width}px;"
                        ):
                            ui.label("Criterion")
                            ui.label("Direction")
                            ui.label("Weight")
                            ui.label("Preference function")
                            ui.label("Q")
                            ui.label("P")
                            ui.label("S")

                        for idx, criterion in enumerate(criteria):
                            with ui.element("div").style(
                                f"display: grid; {grid_template} align-items: center; gap: 12px; min-width: {min_width}px;"
                            ):
                                ui.label(criterion)
                                ui.select(
                                    options=PROMETHEE_DIRECTIONS,
                                    value=state.project.promethee_directions[idx],
                                    on_change=lambda e, i=idx: update_promethee_direction(i, e.value),
                                ).classes("w-full")
                                ui.number(
                                    value=state.project.promethee_weights[idx],
                                    min=0,
                                    step=0.1,
                                    format="%.2f",
                                    on_change=lambda e, i=idx: update_promethee_weight(i, e.value),
                                ).classes("w-full").props('input-class="text-center" dense')
                                ui.select(
                                    options=PROMETHEE_FUNCTIONS,
                                    value=state.project.promethee_functions[idx],
                                    on_change=lambda e, i=idx: update_promethee_function(i, e.value),
                                ).classes("w-full")
                                ui.number(
                                    value=state.project.promethee_q[idx],
                                    min=0,
                                    step=0.1,
                                    format="%.2f",
                                    on_change=lambda e, i=idx: update_promethee_threshold("q", i, e.value),
                                ).classes("w-full").props('input-class="text-center" dense')
                                ui.number(
                                    value=state.project.promethee_p[idx],
                                    min=0,
                                    step=0.1,
                                    format="%.2f",
                                    on_change=lambda e, i=idx: update_promethee_threshold("p", i, e.value),
                                ).classes("w-full").props('input-class="text-center" dense')
                                ui.number(
                                    value=state.project.promethee_s[idx],
                                    min=0,
                                    step=0.1,
                                    format="%.2f",
                                    on_change=lambda e, i=idx: update_promethee_threshold("s", i, e.value),
                                ).classes("w-full").props('input-class="text-center" dense')

                    ui.label("Preference function guide:").classes("text-sm text-gray-500 mt-2")
                    ui.label("Let d be the score difference (after applying Minimize/Maximize).").classes(
                        "text-sm text-gray-500"
                    )
                    ui.label("Q = indifference threshold, P = preference threshold, S = shape/scale.").classes(
                        "text-sm text-gray-500"
                    )
                    ui.label("t1 usual: d <= 0 -> 0, d > 0 -> 1 (no thresholds).").classes("text-sm text-gray-500")
                    ui.label("t2 U-shape: d <= Q -> 0, d > Q -> 1 (use Q).").classes("text-sm text-gray-500")
                    ui.label("t3 V-shape: 0..P grows linearly, d >= P -> 1 (use P).").classes(
                        "text-sm text-gray-500"
                    )
                    ui.label("t4 level: d <= Q -> 0, Q..P -> 0.5, d >= P -> 1 (use Q,P).").classes(
                        "text-sm text-gray-500"
                    )
                    ui.label("t5 linear: d <= Q -> 0, Q..P linear to 1, d >= P -> 1 (use Q,P).").classes(
                        "text-sm text-gray-500"
                    )
                    ui.label("t6 Gaussian: smooth curve with S controlling spread (use S > 0).").classes(
                        "text-sm text-gray-500"
                    )
                    ui.label("t7 quasi: slow start up to S, then full preference (use S > 0).").classes(
                        "text-sm text-gray-500"
                    )
                    ui.label("Use non-negative thresholds in the same units as your scores; keep Q <= P.").classes(
                        "text-sm text-gray-500"
                    )
                    ui.label("Weights are normalized automatically when you compute weights or results.").classes(
                        "text-sm text-gray-500"
                    )

                promethee_view()

                @ui.refreshable
                def weights_action() -> None:
                    label = (
                        "Normalize weights"
                        if state.project.method_id == "promethee_ii"
                        else "Compute weights"
                    )
                    ui.button(label, on_click=compute_weights).classes("mt-2")

                weights_action()

                @ui.refreshable
                def weights_view() -> None:
                    if not state.project.criteria:
                        ui.label("Add criteria to see weights.").classes("text-gray-500")
                        return
                    if not state.project.weights or len(state.project.weights) != len(state.project.criteria):
                        ui.label("No weights computed yet.").classes("text-gray-500")
                        return
                    title = "Weights"
                    if state.project.method_id == "promethee_ii":
                        title = "Weights (normalized)"
                    ui.label(title).classes("text-md font-semibold mt-4")
                    with ui.column().classes("gap-2"):
                        for criterion, weight in zip(state.project.criteria, state.project.weights):
                            ui.label(f"{criterion}: {weight:.4f}")
                    if state.project.method_id == "ahp" and state.project.consistency_ratio is not None:
                        cr = state.project.consistency_ratio
                        if math.isnan(cr):
                            ui.label("Consistency ratio: n/a").classes("text-sm text-gray-500")
                            return
                        ui.label(f"Consistency ratio: {cr:.4f}").classes("text-sm text-gray-500")
                        if cr > CONSISTENCY_THRESHOLD:
                            ui.label(
                                f"Warning: CR is above {CONSISTENCY_THRESHOLD:.2f}. Consider revising pairwise judgments."
                            ).classes("text-sm text-negative font-semibold")
                        discrepancy = max_pairwise_discrepancy(
                            state.project.criteria,
                            state.project.pairwise,
                            state.project.weights,
                        )
                        if discrepancy:
                            _, i, j, actual, expected = discrepancy
                            actual_ui = ui_value_from_ratio(actual)
                            expected_ui = ui_value_from_ratio(expected)
                            ui.label(
                                "Tip: To improve consistency (lower CR), adjust the largest mismatch."
                            ).classes("text-sm text-gray-500")
                            ui.label(
                                f"Biggest discrepancy: {state.project.criteria[i]} vs {state.project.criteria[j]}."
                            ).classes("text-sm text-gray-500")
                            ui.label(
                                f"You set {actual_ui} (ratio {actual:.3f}); weights imply {expected_ui} (ratio {expected:.3f})."
                            ).classes("text-sm text-gray-500")
                    if state.project.method_id == "bwm" and state.project.consistency_ratio is not None:
                        xi = state.project.consistency_ratio
                        ui.label(f"Max deviation (xi): {xi:.4f}").classes("text-sm text-gray-500")
                        ui.label("Lower is better; 0 means fully consistent.").classes("text-sm text-gray-500")

                weights_view()

                with ui.stepper_navigation():
                    ui.button("Back", on_click=stepper.previous).props("flat")
                    ui.button("Next", on_click=stepper.next)

        with ui.step("3. Options & Scores"):
            with ui.card().classes("w-full"):
                ui.label("Add options").classes("text-lg font-semibold")
                with ui.row().classes("items-center"):
                    option_input = ui.input("Option name")

                    def submit_option() -> None:
                        add_option(option_input.value)
                        option_input.set_value("")

                    option_input.on("keydown.enter", lambda: submit_option())
                    ui.button("Add", on_click=submit_option)

                @ui.refreshable
                def options_view() -> None:
                    if not state.project.criteria:
                        ui.label("Add criteria before scoring options.").classes("text-gray-500")
                        return
                    if not state.project.options:
                        ui.label("No options yet.").classes("text-gray-500")
                        return
                    if state.project.method_id == "promethee_ii":
                        ui.label("Scores (use natural scale; mark cost criteria as Minimize in PROMETHEE II).").classes(
                            "text-sm text-gray-500"
                        )
                    else:
                        ui.label("Scores (higher is better)").classes("text-sm text-gray-500")
                    criteria = state.project.criteria
                    option_col_width = 220
                    score_col_width = 140
                    action_col_width = 110
                    min_width = option_col_width + (len(criteria) * score_col_width) + action_col_width
                    grid_template = (
                        f"grid-template-columns: {option_col_width}px "
                        f"repeat({len(criteria)}, {score_col_width}px) {action_col_width}px;"
                    )

                    with ui.element("div").classes("w-full overflow-x-auto").style("max-width: 100%;"):
                        with ui.column().classes("gap-2"):
                            with ui.element("div").style(
                                f"display: grid; {grid_template} align-items: center; gap: 12px; min-width: {min_width}px;"
                            ):
                                ui.label("Option")
                                for criterion in criteria:
                                    ui.label(criterion).classes("text-center").style("justify-self: center;")
                                ui.label("")
                            for idx, option in enumerate(state.project.options):
                                with ui.element("div").style(
                                    f"display: grid; {grid_template} align-items: center; gap: 12px; min-width: {min_width}px;"
                                ):
                                    ui.label(option).classes("break-words pr-2")
                                    for j in range(len(criteria)):
                                        ui.number(
                                            value=state.project.scores[idx][j],
                                            min=1,
                                            max=10,
                                            step=1,
                                            format="%d",
                                            on_change=lambda e, ii=idx, jj=j: update_score(ii, jj, e.value),
                                        ).classes("w-full").props('input-class="text-center" dense')
                                    ui.button("Remove", on_click=lambda i=idx: remove_option(i)).props(
                                        "outline color=negative"
                                    ).classes("w-full")

                options_view()

                with ui.stepper_navigation():
                    ui.button("Back", on_click=stepper.previous).props("flat")
                    ui.button("Next", on_click=stepper.next)

        with ui.step("4. Results"):
            with ui.card().classes("w-full"):
                ui.label("Ranking").classes("text-lg font-semibold")
                ui.button("Recompute results", on_click=recompute_results)

                @ui.refreshable
                def results_view() -> None:
                    if not state.project.results:
                        ui.label("No results yet.").classes("text-gray-500")
                        return
                    if state.project.method_id == "promethee_ii":
                        ui.label("Net flow (higher is better).").classes("text-sm text-gray-500")
                    with ui.column().classes("gap-2"):
                        for result in state.project.results:
                            ui.label(f"{result.option}: {result.score:.4f}")

                results_view()

                with ui.stepper_navigation():
                    ui.button("Back", on_click=stepper.previous).props("flat")
                    ui.button("Save", on_click=save_current)


ensure_state_consistency()
ng_run.setup = lambda: None
host = os.getenv("HOST", "127.0.0.1")
port = int(os.getenv("PORT", "8080"))
ui.run(reload=False, host=host, port=port)
