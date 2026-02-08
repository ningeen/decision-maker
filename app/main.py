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
from models import Project, Result
from storage import list_projects, load_project, save_project


@dataclass
class AppState:
    project: Project


state = AppState(project=Project(name="default"))
method_heading: Optional[ui.label] = None
method_hint_primary: Optional[ui.label] = None
method_hint_secondary: Optional[ui.label] = None


def resolve_method_id(selection: str, options: dict) -> str:
    if selection in options:
        return selection
    for method_id, label in options.items():
        if label == selection:
            return method_id
    return "ahp"


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


UI_SCALE_TO_SAATY = {
    1: 2.0,
    2: 3.0,
    3: 5.0,
    4: 7.0,
    5: 9.0,
}

SAATY_VALUES = list(UI_SCALE_TO_SAATY.values())
CONSISTENCY_THRESHOLD = 0.1


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


def ensure_state_consistency() -> None:
    project = state.project
    project.pairwise = resize_square_matrix(project.pairwise, len(project.criteria), default=1.0)
    project.weights = project.weights[: len(project.criteria)]
    if len(project.weights) != len(project.criteria):
        project.weights = [1.0 / len(project.criteria)] * len(project.criteria) if project.criteria else []
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
    weights_view.refresh()
    options_view.refresh()
    results_view.refresh()


def remove_criterion(index: int) -> None:
    state.project.criteria.pop(index)
    ensure_state_consistency()
    criteria_view.refresh()
    pairwise_view.refresh()
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


def compute_weights() -> None:
    project = state.project
    if len(project.criteria) < 2:
        ui.notify("Add at least two criteria to compute weights.")
        return
    method = get_method(project.method_id)
    try:
        result = method.compute_weights(project.criteria, project.pairwise)
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
    if state.project.method_id == "choix":
        method_heading.set_text("Pairwise preferences (upper triangle)")
        method_hint_primary.set_text(
            "Use -5 to 5 scale (0 = equal preference). Positive means row is preferred; negative means less preferred."
        )
        method_hint_secondary.set_text(
            "Choix converts this to pairwise wins for a Bradley-Terry model; lower triangle is filled automatically."
        )
    else:
        method_heading.set_text("Pairwise comparison (upper triangle)")
        method_hint_primary.set_text(
            "Use -5 to 5 scale (0 = equal importance). Positive means row is more important; negative means less important."
        )
        method_hint_secondary.set_text(
            "Internally this maps to the AHP 1/9-9 scale; lower triangle is filled automatically."
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


def recompute_results() -> None:
    project = state.project
    if not project.weights or len(project.weights) != len(project.criteria):
        ui.notify("Compute weights before scoring options.")
        return
    if not project.options:
        ui.notify("Add at least one option to score.")
        return
    method = get_method(project.method_id)
    option_scores = {
        option: project.scores[idx]
        for idx, option in enumerate(project.options)
    }
    ranked = method.compute_scores(project.weights, option_scores)
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
    weights_view.refresh()
    options_view.refresh()
    results_view.refresh()
    ui.notify(f"Loaded {state.project.name}")


def refresh_saved_projects() -> None:
    saved_select.options = list_projects()
    saved_select.update()


ui.page_title("Decision Helper")

with ui.column().classes("w-full max-w-6xl mx-auto p-6"):
    ui.label("Decision Helper").classes("text-3xl font-semibold")
    ui.label("Multi-criteria decision analysis with AHP weighting.").classes("text-gray-500")

    with ui.card().classes("w-full"):
        ui.label("Project").classes("text-lg font-semibold")
        with ui.row().classes("items-center"):
            project_name_input = ui.input("Project name", value=state.project.name)

            def on_project_name_change(event) -> None:
                state.project.name = (event.value or "").strip() or "Untitled"

            project_name_input.on_value_change(on_project_name_change)
            ui.button("Save", on_click=save_current)
            def on_method_change(event) -> None:
                state.project.method_id = resolve_method_id(event.value, method_options)
                state.project.consistency_ratio = None
                update_method_hints()
                weights_view.refresh()

            method_options = list_methods()
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
                    criteria = state.project.criteria
                    if len(criteria) < 2:
                        ui.label("Add at least two criteria.").classes("text-gray-500")
                        return
                    with ui.column().classes("gap-2"):
                        grid_template = f"grid-template-columns: 160px repeat({len(criteria)}, 120px);"

                        with ui.element("div").style(
                            f"display: grid; {grid_template} align-items: center; gap: 12px;"
                        ):
                            ui.label("")
                            for criterion in criteria:
                                ui.label(criterion).classes("text-center").style("justify-self: center;")

                        for i, row_name in enumerate(criteria):
                            with ui.element("div").style(
                                f"display: grid; {grid_template} align-items: center; gap: 12px;"
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
                                        ui.label(f"{ui_value}").classes("text-center text-gray-500").style("justify-self: center;")

                pairwise_view()

                ui.button("Compute weights", on_click=compute_weights).classes("mt-2")

                @ui.refreshable
                def weights_view() -> None:
                    if not state.project.weights:
                        ui.label("No weights computed yet.").classes("text-gray-500")
                        return
                    ui.label("Weights").classes("text-md font-semibold mt-4")
                    with ui.column().classes("gap-2"):
                        for criterion, weight in zip(state.project.criteria, state.project.weights):
                            ui.label(f"{criterion}: {weight:.4f}")
                    if state.project.consistency_ratio is not None:
                        cr = state.project.consistency_ratio
                        if math.isnan(cr):
                            ui.label("Consistency ratio: n/a").classes("text-sm text-gray-500")
                            return
                        ui.label(f"Consistency ratio: {cr:.4f}").classes("text-sm text-gray-500")
                        if cr > CONSISTENCY_THRESHOLD:
                            ui.label(
                                f"Warning: CR is above {CONSISTENCY_THRESHOLD:.2f}. Consider revising pairwise judgments."
                            ).classes("text-sm text-negative font-semibold")

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
                    ui.label("Scores (higher is better)").classes("text-sm text-gray-500")
                    with ui.column().classes("gap-2"):
                        with ui.row().classes("items-center gap-4"):
                            ui.label("Option").classes("w-32")
                            for criterion in state.project.criteria:
                                ui.label(criterion).classes("w-32 text-center")
                            ui.label("").classes("w-24")
                        for idx, option in enumerate(state.project.options):
                            with ui.row().classes("items-center gap-4"):
                                ui.label(option).classes("w-32")
                                for j in range(len(state.project.criteria)):
                                    ui.number(
                                        value=state.project.scores[idx][j],
                                        min=1,
                                        max=10,
                                        step=1,
                                        format="%d",
                                        on_change=lambda e, ii=idx, jj=j: update_score(ii, jj, e.value),
                                    ).classes("w-32").props('input-class="text-center" dense')
                                ui.button("Remove", on_click=lambda i=idx: remove_option(i)).props("outline color=negative").classes("w-24")

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
