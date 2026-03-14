from __future__ import annotations

from pathlib import Path

from lucid.modal import cli


def test_run_modal_defaults_to_current_project_and_src(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "waypoint_modal"
    src_root = project_root / "src" / "waypoint_modal_example"
    src_root.mkdir(parents=True)
    (src_root / "__init__.py").write_text("", encoding="utf-8")
    (src_root / "modal_app.py").write_text("app = None\n", encoding="utf-8")

    env_file = project_root / "waypoint.env"
    env_file.write_text(
        "\n".join(
            [
                "MODAL_APP_ENTRYPOINT=waypoint_modal_example.modal_app",
            ]
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def _fake_exec_modal(
        project_root: Path,
        modal_args: list[str],
        env: dict[str, str],
        *,
        check: bool = True,
    ) -> int:
        captured["project_root"] = project_root
        captured["modal_args"] = modal_args
        captured["env"] = env
        captured["check"] = check
        return 0

    monkeypatch.setattr(cli, "_exec_modal", _fake_exec_modal)
    monkeypatch.chdir(project_root)

    args = cli.build_parser().parse_args(
        ["deploy", "--env-file", str(env_file)],
    )

    assert cli._run_modal(args) == 0
    assert captured["project_root"] == project_root.resolve()
    assert captured["modal_args"] == [
        "deploy",
        "-m",
        "waypoint_modal_example.modal_app",
    ]
    resolved_pythonpath = str(captured["env"]["PYTHONPATH"]).split(":")[0]
    assert resolved_pythonpath == str((project_root / "src").resolve())


def test_run_modal_converts_src_file_entrypoint_to_module_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "waypoint_modal"
    src_root = project_root / "src" / "waypoint_modal_example"
    src_root.mkdir(parents=True)
    (src_root / "__init__.py").write_text("", encoding="utf-8")
    (src_root / "modal_app.py").write_text("app = None\n", encoding="utf-8")

    env_file = project_root / "waypoint.env"
    env_file.write_text(
        "MODAL_APP_ENTRYPOINT=src/waypoint_modal_example/modal_app.py\n",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def _fake_exec_modal(
        project_root: Path,
        modal_args: list[str],
        env: dict[str, str],
        *,
        check: bool = True,
    ) -> int:
        captured["project_root"] = project_root
        captured["modal_args"] = modal_args
        captured["env"] = env
        captured["check"] = check
        return 0

    monkeypatch.setattr(cli, "_exec_modal", _fake_exec_modal)
    monkeypatch.chdir(project_root)

    args = cli.build_parser().parse_args(
        ["download-model", "--env-file", str(env_file)],
    )

    assert cli._run_modal(args) == 0
    assert captured["project_root"] == project_root.resolve()
    assert captured["modal_args"] == [
        "run",
        "-m",
        "waypoint_modal_example.modal_app::download_model",
    ]
