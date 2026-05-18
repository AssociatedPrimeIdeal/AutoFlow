from autoflow.cli import build_parser


def test_cli_background_phase_correction_is_opt_in():
    args = build_parser().parse_args(["demo.h5"])
    assert args.background_phase_correction is False

    args = build_parser().parse_args(["demo.h5", "--bgc"])
    assert args.background_phase_correction is True


def test_cli_short_bgc_parameter_names_are_supported():
    args = build_parser().parse_args(
        ["demo.h5", "--bgc", "--bgc-fit-order", "2", "--bgc-threshold", "0.05"]
    )
    assert args.background_phase_correction is True
    assert args.background_phase_fit_order == 2
    assert args.background_phase_threshold == 0.05


def test_cli_legacy_long_bgc_parameter_names_remain_supported():
    args = build_parser().parse_args(
        [
            "demo.h5",
            "--bgc",
            "--background-phase-fit-order",
            "1",
            "--background-phase-threshold",
            "0.2",
        ]
    )
    assert args.background_phase_fit_order == 1
    assert args.background_phase_threshold == 0.2
