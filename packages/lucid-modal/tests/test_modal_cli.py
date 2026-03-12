from __future__ import annotations

from pathlib import Path

from lucid_modal import cli


def test_run_modal_uses_entrypoint_from_env_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_file = tmp_path / "waypoint.env"
    env_file.write_text(
        "\n".join(
            [
                "MODAL_PROJECT_PATH=examples/waypoint_modal",
                "MODAL_PROJECT_SRC=examples/waypoint_modal/src",
                "MODAL_APP_ENTRYPOINT=examples/waypoint_modal/src/waypoint_modal_example/modal_app.py",
            ]
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def _fake_exec_modal(
        project_path: str,
        modal_args: list[str],
        env: dict[str, str],
        *,
        check: bool = True,
    ) -> int:
        captured["project_path"] = project_path
        captured["modal_args"] = modal_args
        captured["env"] = env
        captured["check"] = check
        return 0

    monkeypatch.setattr(cli, "_exec_modal", _fake_exec_modal)

    args = cli.build_parser().parse_args(
        ["deploy", "--env-file", str(env_file)],
    )

    assert cli._run_modal(args) == 0
    assert captured["project_path"] == "examples/waypoint_modal"
    assert captured["modal_args"] == [
        "deploy",
        "examples/waypoint_modal/src/waypoint_modal_example/modal_app.py",
    ]
    resolved_pythonpath = str(captured["env"]["PYTHONPATH"])
    assert "examples/waypoint_modal/src" in resolved_pythonpath
