"""Orchestrating the dMRI-preprocessing workflow."""
from ... import config
from pathlib import Path
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from ...interfaces import DerivativesDataSink


def init_dwi_preproc_wf(dwi_file):
    """
    Build a preprocessing workflow for one DWI run.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from dmriprep.config.testing import mock_config
            from dmriprep import config
            from dmriprep.workflows.dwi.base import init_dwi_preproc_wf
            with mock_config():
                wf = init_dwi_preproc_wf(
                    f"{config.execution.layout.root}/"
                    "sub-THP0005/dwi/sub-THP0005_dwi.nii.gz"
                )

    Parameters
    ----------
    dwi_file : :obj:`os.PathLike`
        One diffusion MRI dataset to be processed.

    Inputs
    ------
    dwi_file
        dwi NIfTI file
    in_bvec
        File path of the b-vectors
    in_bval
        File path of the b-values

    Outputs
    -------
    dwi_reference
        A 3D :math:`b = 0` reference, before susceptibility distortion correction.
    dwi_mask
        A 3D, binary mask of the ``dwi_reference`` above.
    gradients_rasb
        A *RASb* (RAS+ coordinates, scaled b-values, normalized b-vectors, BIDS-compatible)
        gradient table.

    See Also
    --------
    * :py:func:`~dmriprep.workflows.dwi.util.init_dwi_reference_wf`
    * :py:func:`~dmriprep.workflows.dwi.outputs.init_reportlets_wf`

    """
    from ...interfaces.vectors import CheckGradientTable
    from .util import init_dwi_reference_wf
    from .outputs import init_reportlets_wf

    layout = config.execution.layout

    dwi_file = Path(dwi_file)
    config.loggers.workflow.debug(
        f"Creating DWI preprocessing workflow for <{dwi_file.name}>"
    )

    # Build workflow
    workflow = Workflow(name=_get_wf_name(dwi_file.name))

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                # DWI
                "dwi_file",
                "in_bvec",
                "in_bval",
                # From anatomical
                "t1w_preproc",
                "t1w_mask",
                "t1w_dseg",
                "t1w_aseg",
                "t1w_aparc",
                "t1w_tpms",
                "template",
                "anat2std_xfm",
                "std2anat_xfm",
                "subjects_dir",
                "subject_id",
                "t1w2fsnative_xfm",
                "fsnative2t1w_xfm",
            ]
        ),
        name="inputnode",
    )
    inputnode.inputs.dwi_file = str(dwi_file.absolute())
    inputnode.inputs.in_bvec = str(layout.get_bvec(dwi_file))
    inputnode.inputs.in_bval = str(layout.get_bval(dwi_file))

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["dwi_reference", "dwi_mask", "gradients_rasb"]),
        name="outputnode",
    )

    gradient_table = pe.Node(CheckGradientTable(), name="gradient_table")

    dwi_reference_wf = init_dwi_reference_wf(
        mem_gb=config.DEFAULT_MEMORY_MIN_GB, omp_nthreads=config.nipype.omp_nthreads
    )

    # MAIN WORKFLOW STRUCTURE
    # fmt:off
    workflow.connect([
        (inputnode, gradient_table, [("dwi_file", "dwi_file"),
                                     ("in_bvec", "in_bvec"),
                                     ("in_bval", "in_bval")]),
        (inputnode, dwi_reference_wf, [("dwi_file", "inputnode.dwi_file")]),
        (gradient_table, dwi_reference_wf, [("b0_ixs", "inputnode.b0_ixs")]),
        (dwi_reference_wf, outputnode, [
            ("outputnode.ref_image", "dwi_reference"),
            ("outputnode.dwi_mask", "dwi_mask"),
        ]),
        (gradient_table, outputnode, [("out_rasb", "gradients_rasb")]),
    ])
    # fmt:on

    if config.workflow.run_reconall:
        from niworkflows.interfaces.nibabel import ApplyMask
        from niworkflows.anat.coregistration import init_bbreg_wf
        from ...utils.misc import sub_prefix as _prefix

        # Mask the T1w
        t1w_brain = pe.Node(ApplyMask(), name="t1w_brain")

        bbr_wf = init_bbreg_wf(
            debug=config.execution.debug,
            epi2t1w_init=config.workflow.dwi2t1w_init,
            omp_nthreads=config.nipype.omp_nthreads,
        )

        ds_report_reg = pe.Node(
            DerivativesDataSink(
                base_directory=str(config.execution.output_dir),
                datatype="figures",
            ),
            name="ds_report_reg",
            run_without_submitting=True,
        )

        def _bold_reg_suffix(fallback):
            return "coreg" if fallback else "bbregister"

        # fmt:off
        workflow.connect([
            (inputnode, bbr_wf, [
                ("fsnative2t1w_xfm", "inputnode.fsnative2t1w_xfm"),
                (("subject_id", _prefix), "inputnode.subject_id"),
                ("subjects_dir", "inputnode.subjects_dir"),
            ]),
            # T1w Mask
            (inputnode, t1w_brain, [("t1w_preproc", "in_file"),
                                    ("t1w_mask", "in_mask")]),
            (inputnode, ds_report_reg, [("dwi_file", "source_file")]),
            # BBRegister
            (dwi_reference_wf, bbr_wf, [
                ("outputnode.ref_image", "inputnode.in_file")
            ]),
            (bbr_wf, ds_report_reg, [
                ('outputnode.out_report', 'in_file'),
                (('outputnode.fallback', _bold_reg_suffix), 'desc')]),
        ])
        # fmt:on

    # Eddy distortion correction
    # Need acqp, dwi, mask, in_index
    eddy_wf = init_eddy_wf()
    # fmt:off
    workflow.connect([
        (dwi_reference_wf, eddy_wf, [
            ("outputnode.ref_image", "inputnode.dwi_file"),
            ("outputnode.dwi_mask", "inputnode.dwi_mask"),
        ]),
        (inputnode, gradient_table, [
            ("in_bvec", "in_bvec"),
            ("in_bval", "in_bval")
        ]),
    ])
    # fmt:on


    # REPORTING ############################################################
    reportlets_wf = init_reportlets_wf(str(config.execution.output_dir))
    # fmt:off
    workflow.connect([
        (inputnode, reportlets_wf, [("dwi_file", "inputnode.source_file")]),
        (dwi_reference_wf, reportlets_wf, [
            ("outputnode.ref_image", "inputnode.dwi_ref"),
            ("outputnode.dwi_mask", "inputnode.dwi_mask"),
            ("outputnode.validation_report", "inputnode.validation_report"),
        ]),
    ])
    # fmt:on

    return workflow

def gen_index(in_file):
    # Generate the index file for eddy
    import os
    import numpy as np
    import nibabel as nib
    from nipype.pipeline import engine as pe
    from nipype.interfaces import fsl, utility as niu
    from nipype.utils.filemanip import fname_presuffix

    out_file = fname_presuffix(
        in_file,
        suffix="_index.txt",
        newpath=os.path.abspath("."),
        use_ext=False,
    )
    vols = nib.load(in_file).shape[-1]
    index_lines = np.ones((vols,))
    index_lines_reshape = index_lines.reshape(1, index_lines.shape[0])
    np.savetxt(out_file, index_lines_reshape, fmt="%i")
    return out_file

def gen_acqparams(in_file, total_readout_time=0.05):
    # Generate the acqp file for eddy
    import os
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    # Get the metadata for the dwi
    layout = config.execution.layout
    metadata = str(layout.get_metadata(dwi_file))

    # Generate output file name
    out_file = fname_presuffix(
        in_file,
        suffix="_acqparams.txt",
        newpath=os.path.abspath("."),
        use_ext=False,
    )

    # Dictionary for converting directions to acqp lines
    acq_param_dict = {
        "j": "0 1 0 %.7f",
        "j-": "0 -1 0 %.7f",
        "i": "1 0 0 %.7f",
        "i-": "-1 0 0 %.7f",
        "k": "0 0 1 %.7f",
        "k-": "0 0 -1 %.7f",
    }

    # Get encoding direction from json
    if metadata.get("PhaseEncodingDirection"):
        pe_dir = metadata.get("PhaseEncodingDirection")
    else
        pe_dir = metadata.get("PhaseEncodingAxis")

    # Get readout time from json, use default of 0.05 otherwise
    if metadata.get("TotalReadoutTime"):
        total_readout = metadata.get("TotalReadoutTime")
    else
        total_readout = total_readout_time

    # Construct the acqp file lines
    acq_param_lines = acq_param_dict[pe_dir] % total_readout

    # Write to the acqp file
    with open(out_file, "w") as f:
        f.write(acq_param_lines)

    return out_file


def init_eddy_wf(
        name="eddy_wf",
    ):
    """
    Create an eddy workflow for head motion distortion correction on the dwi.
    Parameters
    ----------
    dwi_file : :obj:`str`, optional
        Workflow name (default: ``"eddy_wf"``)
    dwi_mask : :obj:`str`, optional
        Workflow name (default: ``"eddy_wf"``)
    in_bvec : :obj:`str`, optional
        Workflow name (default: ``"eddy_wf"``)
    in_bval : :obj:`str`, optional
        Workflow name (default: ``"eddy_wf"``)
    Outputs
    -------
    out_<B0FieldIdentifier>.fmap :
        The preprocessed fieldmap.
    out_<B0FieldIdentifier>.fmap_ref :
        The preprocessed fieldmap reference.
    out_<B0FieldIdentifier>.fmap_coeff :
        The preprocessed fieldmap coefficients.
    """
    from nipype.interfaces.fsl import ApplyMask, Eddy, EddyQuad

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "dwi_file",
                "dwi_mask",
                "in_bvec",
                "in_bval",
            ]
        ),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "out_eddy",
                "out_rotated_bvecs"
            ]
        ),
        name="outputnode",
    )

    workflow = Workflow(name=name)

    eddy = pe.Node(
        Eddy(repol=True, cnr_maps=True, residuals=True, method="jac"),
        name="fsl_eddy",
    )

    # Generate the acqp file for eddy
    eddy_acqp = pe.Node(
        niu.Function(
            input_names=["in_file"],
            output_names=["out_file"],
            function=gen_acqparams,
        ),
        name="eddy_acqp",
    )

    # Generate the index file for eddy
    gen_idx = pe.Node(
        niu.Function(
            input_names=["in_file"],
            output_names=["out_file"],
            function=gen_index,
        ),
        name="gen_index",
    )

    eddy_quad = pe.Node(EddyQuad(verbose=True), name="eddy_quad")

    # Connect the workflow
    # fmt:off
    workflow.connect([
        (inputnode, eddy, [
            ("dwi_file", "inputnode.fieldmap"),
            ("dwi_mask", "inputnode.fmap_ref"),
            ("in_bvec", "inputnode.fmap_coeff"),
            ("in_bval", "inputnode.fmap_coeff"),
        ]),
        (acqp, eddy, [("out_file", "in_acqp")]),
        (gen_idx, eddy, [("out_file", "in_index")]),
        # Eddy Quad Outputs
        (inputnode, eddy_quad, [
            ("dwi_mask", "mask_file"),
            ("in_bval", "bval_file"),
        ]),
        (eddy, eddy_quad, [("out_rotated_bvecs", "bvec_file")]),
        (acqp, eddy_quad, [("out_file", "param_file")]),
        (gen_idx, eddy_quad, [("out_file", "idx_file")]),
        (eddy, outputnode, [
            ("out_corrected", "out_eddy"),
            ("out_rotated_bvecs", "out_rotated_bvecs")
        ]),
    ])
    # fmt:on

    return workflow

def _get_wf_name(filename):
    """
    Derive the workflow name for supplied DWI file.

    Examples
    --------
    >>> _get_wf_name('/completely/made/up/path/sub-01_dir-AP_acq-64grad_dwi.nii.gz')
    'dwi_preproc_dir_AP_acq_64grad_wf'

    >>> _get_wf_name('/completely/made/up/path/sub-01_dir-RL_run-01_echo-1_dwi.nii.gz')
    'dwi_preproc_dir_RL_run_01_echo_1_wf'

    """
    from pathlib import Path

    fname = Path(filename).name.rpartition(".nii")[0].replace("_dwi", "_wf")
    fname_nosub = "_".join(fname.split("_")[1:])
    return f"dwi_preproc_{fname_nosub.replace('.', '_').replace(' ', '').replace('-', '_')}"
