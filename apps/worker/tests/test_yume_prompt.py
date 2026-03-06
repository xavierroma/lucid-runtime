from __future__ import annotations

from wm_worker.models import ActionPayload
from wm_worker.yume_prompt import compose_yume_prompt


def test_compose_yume_prompt_prefixes_scene_with_motion_instructions() -> None:
    prompt = compose_yume_prompt(
        "A neon-lit city street with reflective puddles.",
        ActionPayload(
            keys=["W", "A"],
            mouse_dx=-2.0,
            mouse_dy=0.0,
            actual_distance=4.0,
            angular_change_rate=4.0,
            view_rotation_speed=4.0,
        ),
    )

    assert prompt.startswith("First-person perspective.")
    assert "The camera pushes forward and left (W+A)." in prompt
    assert "The camera pans to the left (←)." in prompt
    assert "Actual distance moved:4 at 100 meters per second." in prompt
    assert "View rotation speed:4." in prompt
    assert prompt.endswith("A neon-lit city street with reflective puddles.")


def test_compose_yume_prompt_reduces_motion_values_when_idle() -> None:
    prompt = compose_yume_prompt(
        "",
        ActionPayload(
            keys=[],
            mouse_dx=0.0,
            mouse_dy=0.0,
            actual_distance=7.0,
            angular_change_rate=6.0,
            view_rotation_speed=5.0,
        ),
    )

    assert "The movement direction of the camera remains stationary (·)." in prompt
    assert "The rotation direction of the camera remains stationary (·)." in prompt
    assert "Actual distance moved:0.5 at 100 meters per second." in prompt
    assert "Angular change rate (turn speed):0.5." in prompt
    assert "View rotation speed:0.5." in prompt


def test_compose_yume_prompt_leaves_structured_prompt_unchanged() -> None:
    structured = (
        "First-person perspective. The camera pushes forward (W). "
        "The rotation direction of the camera remains stationary (·). "
        "Actual distance moved:4 at 100 meters per second. "
        "Angular change rate (turn speed):0. View rotation speed:0. "
        "A misty forest path at dawn."
    )

    prompt = compose_yume_prompt(structured, ActionPayload())

    assert prompt == structured
