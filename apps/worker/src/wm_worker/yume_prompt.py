from __future__ import annotations

from wm_worker.models import ActionPayload


_DEFAULT_SCENE_PROMPT = (
    "A realistic explorable world with clear depth, coherent geometry, and natural motion."
)


def compose_yume_prompt(
    scene_prompt: str,
    action: ActionPayload,
    *,
    default_scene_prompt: str = _DEFAULT_SCENE_PROMPT,
) -> str:
    normalized_scene = (scene_prompt or "").strip() or default_scene_prompt
    if _looks_like_structured_yume_prompt(normalized_scene):
        return normalized_scene

    key_description, has_motion = _describe_keyboard_motion(action.keys)
    mouse_description, has_rotation = _describe_camera_rotation(
        action.mouse_dx, action.mouse_dy
    )

    actual_distance = _clamp(action.actual_distance, minimum=0.1, maximum=10.0)
    angular_change_rate = _clamp(
        action.angular_change_rate,
        minimum=0.1,
        maximum=10.0,
    )
    view_rotation_speed = _clamp(
        action.view_rotation_speed,
        minimum=0.1,
        maximum=10.0,
    )

    if not has_motion:
        actual_distance = min(actual_distance, 0.5)
    if not has_rotation:
        angular_change_rate = min(angular_change_rate, 0.5)
        view_rotation_speed = min(view_rotation_speed, 0.5)

    return " ".join(
        [
            "First-person perspective.",
            key_description,
            mouse_description,
            (
                f"Actual distance moved:{_format_float(actual_distance)} "
                "at 100 meters per second."
            ),
            (
                "Angular change rate (turn speed):"
                f"{_format_float(angular_change_rate)}."
            ),
            f"View rotation speed:{_format_float(view_rotation_speed)}.",
            normalized_scene,
        ]
    ).strip()


def _looks_like_structured_yume_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    return "actual distance moved:" in lowered and (
        "first-person perspective" in lowered
        or "first-person view" in lowered
        or "fpv" in lowered
    )


def _describe_keyboard_motion(keys: list[str]) -> tuple[str, bool]:
    normalized_keys = tuple(_normalize_keys(keys))
    mapping = {
        (): "The movement direction of the camera remains stationary (·).",
        ("W",): "The camera pushes forward (W).",
        ("S",): "The camera pulls backward (S).",
        ("A",): "The camera strafes left (A).",
        ("D",): "The camera strafes right (D).",
        ("W", "A"): "The camera pushes forward and left (W+A).",
        ("W", "D"): "The camera pushes forward and right (W+D).",
        ("S", "A"): "The camera pulls backward and left (S+A).",
        ("S", "D"): "The camera pulls backward and right (S+D).",
    }
    if normalized_keys in mapping:
        return mapping[normalized_keys], bool(normalized_keys)
    if not normalized_keys:
        return mapping[()], False
    return f"The camera moves using keyboard input ({'+'.join(normalized_keys)}).", True


def _describe_camera_rotation(mouse_dx: float, mouse_dy: float) -> tuple[str, bool]:
    threshold = 0.1
    if abs(mouse_dx) < threshold and abs(mouse_dy) < threshold:
        return "The rotation direction of the camera remains stationary (·).", False
    if abs(mouse_dx) >= abs(mouse_dy):
        if mouse_dx > 0:
            return "The camera pans to the right (→).", True
        return "The camera pans to the left (←).", True
    if mouse_dy > 0:
        return "The camera tilts downward (↓).", True
    return "The camera tilts upward (↑).", True


def _normalize_keys(keys: list[str]) -> list[str]:
    order = {"W": 0, "S": 1, "A": 2, "D": 3}
    normalized: list[str] = []
    for key in keys:
        value = str(key).strip().upper()
        if value and value not in normalized:
            normalized.append(value)
    normalized.sort(key=lambda value: (order.get(value, 99), value))
    return normalized[:4]


def _clamp(value: float, *, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _format_float(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")
