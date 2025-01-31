if __name__ == "__main__":
    import os
    import sys
    import subprocess as sp
    import numpy as np
    import torch

    sp.check_call([sys.executable, "-m", "pip", "install", "./emager-py[finn]"])

    import finn.builder.build_dataflow as build
    import finn.builder.build_dataflow_config as build_cfg

    from emager_py import dataset
    from emager_py import transforms
    import emager_py.torch.models as etm
    from emager_py.finn import (
        custom_build_steps,
        model_transformations,
        model_validation,
        boards,
    )
    from emager_py.utils import set_logging

    import utils
    import globals

    set_logging()

    MODEL_PARAMS = utils.get_model_params_from_disk()
    utils.unlock_finn()

    # print("Hello")
    # sys.exit(0)

    # Create build directory
    build_dir = utils.format_finn_output_dir(
        MODEL_PARAMS["subject"],
        MODEL_PARAMS["quantization"],
        MODEL_PARAMS["shots"],
    )

    onnx_model = build_dir + "model.onnx"

    torch_model = utils.load_model(
        etm.EmagerSCNN(MODEL_PARAMS["quantization"]),
        MODEL_PARAMS["subject"],
        MODEL_PARAMS["session"],
        MODEL_PARAMS["repetition"],
        MODEL_PARAMS["quantization"],
    )

    # Add topk on model ?
    input_bits = 16 if globals.TRANSFORM == "default" else 8
    model_transformations.save_model_as_qonnx(
        torch_model,
        onnx_model,
        globals.EMAGER_DATA_SHAPE,
        "INT16" if input_bits == 16 else "UINT8",
        show=False,
    )

    if isinstance(globals.TRANSFORM, str):
        transform = transforms.transforms_lut[globals.TRANSFORM]

    # Validate brevitas vs ONNX
    model_validation.validate_brevitas_qonnx(
        torch_model,
        onnx_model,
        globals.VALIDATION_EMAGER_ROOT,
        MODEL_PARAMS["subject"],
        MODEL_PARAMS["session"],
        transform,
        10,
        globals.EMAGER_DATA_SHAPE,
    )

    # Generate validation data and labels
    pvd, pvl = dataset.generate_processed_validation_data(
        globals.VALIDATION_EMAGER_ROOT,
        MODEL_PARAMS["subject"],
        MODEL_PARAMS["session"],
        transform,
        build_dir,
    )
    pvd = pvd.reshape((-1, 1, *globals.EMAGER_DATA_SHAPE))
    pred = torch_model(torch.from_numpy(pvd)).detach().numpy()
    valid_samples = 10
    np.save(build_dir + "/input", pvd[:valid_samples])
    np.save(build_dir + "/expected_output", pred[:valid_samples])

    # Create finn board definition if necessary
    board_name = globals.FINN_TARGET_BOARD
    if not boards.is_board_exists(board_name):
        print(f"Generating board definition for {board_name}")
        board_name = boards.add_board(
            f'{os.environ["XILINX_VIVADO"]}/data/boards/board_files/{board_name}/A.0/board.xml',
            template_board="Pynq-Z2",
        )

    # Set required model metadata properties
    custom_build_steps.CUSTOM_MODEL_PROPERTIES["emager_pynq_path"] = (
        globals.TARGET_EMAGER_PYNQ_PATH
    )
    custom_build_steps.insert_custom_ip(
        os.getcwd() + "/src/insert_rhd2164.tcl",
        RHD2164_SPI_ROOT=os.getcwd() + "/rhd2164-spi-fpga/",
    )

    # Set build steps and build config
    dataflow_steps = custom_build_steps.default_finn_flow_custom_ip()
    # dataflow_steps = custom_build_steps.finn_flow_only_deploy()
    # dataflow_steps = custom_build_steps.default_finn_flow_export_bd()
    # dataflow_steps = build_cfg.default_build_dataflow_steps[0:2]

    cfg = build.DataflowBuildConfig(
        steps=dataflow_steps,
        # start_step=dataflow_steps[-8],  # Only build custom BD
        # start_step=dataflow_steps[-3],  # Only copy BD
        output_dir=build_dir,
        mvau_wwidth_max=36,
        target_fps=1000,
        synth_clk_period_ns=10.0,
        # fpga_part="xc7z020clg400-1",
        board=board_name,
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        # stitched_ip_gen_dcp=True,
        enable_build_pdb_debug=False,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
            build_cfg.DataflowOutputType.BITFILE,
            build_cfg.DataflowOutputType.PYNQ_DRIVER,
            build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
        ],
        verify_steps=[
            build_cfg.VerificationStepType.STREAMLINED_PYTHON,  # TopK breaks verify steps
            build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM,
            build_cfg.VerificationStepType.STITCHED_IP_RTLSIM,
        ],
        save_intermediate_models=True,
        verify_input_npy=build_dir + "input.npy",
        verify_expected_output_npy=build_dir + "expected_output.npy",
    )

    # Build dataflow cfg
    # If error during step_create_dataflow_partition, check if your model is too large
    # Also test if Conv2D layers have bias=False

    build.build_dataflow_cfg(onnx_model, cfg)
