from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu, fsl, afni

"""Registration workflows for :abbr:`DWI (diffusion weighted imaging)` data."""

def init_bbreg_wf(use_bbr, bold2t1w_dof, bold2t1w_init, omp_nthreads, name='bbreg_wf'):
    """
    Build a workflow to run FreeSurfer's ``bbregister``.
    This workflow uses FreeSurfer's ``bbregister`` to register a BOLD image to
    a T1-weighted structural image.
    It is a counterpart to :py:func:`~fmriprep.workflows.bold.registration.init_fsl_bbr_wf`,
    which performs the same task using FSL's FLIRT with a BBR cost function.
    The ``use_bbr`` option permits a high degree of control over registration.
    If ``False``, standard, affine coregistration will be performed using
    FreeSurfer's ``mri_coreg`` tool.
    If ``True``, ``bbregister`` will be seeded with the initial transform found
    by ``mri_coreg`` (equivalent to running ``bbregister --init-coreg``).
    If ``None``, after ``bbregister`` is run, the resulting affine transform
    will be compared to the initial transform found by ``mri_coreg``.
    Excessive deviation will result in rejecting the BBR refinement and
    accepting the original, affine registration.
    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes
            from fmriprep.workflows.bold.registration import init_bbreg_wf
            wf = init_bbreg_wf(use_bbr=True, bold2t1w_dof=9,
                               bold2t1w_init='register', omp_nthreads=1)
    Parameters
    ----------
    use_bbr : :obj:`bool` or None
        Enable/disable boundary-based registration refinement.
        If ``None``, test BBR result for distortion before accepting.
    bold2t1w_dof : 6, 9 or 12
        Degrees-of-freedom for BOLD-T1w registration
    bold2t1w_init : str, 'header' or 'register'
        If ``'header'``, use header information for initialization of BOLD and T1 images.
        If ``'register'``, align volumes by their centers.
    name : :obj:`str`, optional
        Workflow name (default: bbreg_wf)
    Inputs
    ------
    in_file
        Reference BOLD image to be registered
    fsnative2t1w_xfm
        FSL-style affine matrix translating from FreeSurfer T1.mgz to T1w
    subjects_dir
        FreeSurfer SUBJECTS_DIR
    subject_id
        FreeSurfer subject ID (must have folder in SUBJECTS_DIR)
    Outputs
    -------
    out_file
        Output registration file
    itk_bold_to_t1
        Affine transform from ``ref_bold_brain`` to T1 space (ITK format)
    itk_t1_to_bold
        Affine transform from T1 space to BOLD space (ITK format)
    out_report
        Reportlet for assessing registration quality
    fallback
        Boolean indicating whether BBR was rejected (mri_coreg registration returned)
    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.freesurfer import (
        PatchedBBRegisterRPT as BBRegisterRPT,
        PatchedMRICoregRPT as MRICoregRPT,
        PatchedLTAConvert as LTAConvert
    )
    from niworkflows.interfaces.nitransforms import ConcatenateXFMs

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
The BOLD reference was then co-registered to the T1w reference using
`bbregister` (FreeSurfer) which implements boundary-based registration [@bbr].
Co-registration was configured with {dof} degrees of freedom{reason}.
""".format(dof={6: 'six', 9: 'nine', 12: 'twelve'}[bold2t1w_dof],
           reason='' if bold2t1w_dof == 6 else
                  'to account for distortions remaining in the BOLD reference')

    inputnode = pe.Node(
        niu.IdentityInterface([
            'in_file',
            'fsnative2t1w_xfm', 'subjects_dir', 'subject_id']),  # BBRegister
        name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(['itk_bold_to_t1', 'itk_t1_to_bold', 'out_report', 'fallback',
            'out_file']),
        name='outputnode')

    if bold2t1w_init not in ("register", "header"):
        raise ValueError(f"Unknown BOLD-T1w initialization option: {bold2t1w_init}")

    # For now make BBR unconditional - in the future, we can fall back to identity,
    # but adding the flexibility without testing seems a bit dangerous
    if bold2t1w_init == "header":
        if use_bbr is False:
            raise ValueError("Cannot disable BBR and use header registration")
        if use_bbr is None:
            LOGGER.warning("Initializing BBR with header; affine fallback disabled")
            use_bbr = True

    merge_ltas = pe.Node(niu.Merge(2), name='merge_ltas', run_without_submitting=True)
    concat_xfm = pe.Node(ConcatenateXFMs(inverse=True), name='concat_xfm')

    workflow.connect([
        # Output ITK transforms
        (inputnode, merge_ltas, [('fsnative2t1w_xfm', 'in2')]),
        (merge_ltas, concat_xfm, [('out', 'in_xfms')]),
        (concat_xfm, outputnode, [('out_xfm', 'itk_bold_to_t1')]),
        (concat_xfm, outputnode, [('out_inv', 'itk_t1_to_bold')]),
    ])

    # Define both nodes, but only connect conditionally
    mri_coreg = pe.Node(
        MRICoregRPT(dof=bold2t1w_dof, sep=[4], ftol=0.0001, linmintol=0.01,
                    generate_report=not use_bbr),
        name='mri_coreg', n_procs=omp_nthreads, mem_gb=5)

    bbregister = pe.Node(
        BBRegisterRPT(dof=bold2t1w_dof, contrast_type='t2', registered_file=True,
                      out_lta_file=True, generate_report=True),
        name='bbregister', mem_gb=12)

    # Use mri_coreg
    if bold2t1w_init == "register":
        workflow.connect([
            (inputnode, mri_coreg, [('subjects_dir', 'subjects_dir'),
                                    ('subject_id', 'subject_id'),
                                    ('in_file', 'source_file')]),
        ])

        # Short-circuit workflow building, use initial registration
        if use_bbr is False:
            workflow.connect([
                (mri_coreg, outputnode, [('out_report', 'out_report')]),
                (mri_coreg, merge_ltas, [('out_lta_file', 'in1')])
                # Select bbregister output
                (mri_coreg, outputnode, [('out_reg_file', 'out_file')]),
            ])
            outputnode.inputs.fallback = True

            return workflow

    # Use bbregister
    workflow.connect([
        (inputnode, bbregister, [('subjects_dir', 'subjects_dir'),
                                 ('subject_id', 'subject_id'),
                                 ('in_file', 'source_file')]),
    ])

    if bold2t1w_init == "header":
        bbregister.inputs.init = "header"
    else:
        workflow.connect([(mri_coreg, bbregister, [('out_lta_file', 'init_reg_file')])])

    # Short-circuit workflow building, use boundary-based registration
    if use_bbr is True:
        workflow.connect([
            (bbregister, outputnode, [('out_report', 'out_report')]),
            (bbregister, merge_ltas, [('out_lta_file', 'in1')]),
            # Select bbregister output
            (bbregister, outputnode, [('out_reg_file', 'out_file')]),
        ])
        outputnode.inputs.fallback = False

        return workflow

    # Only reach this point if bold2t1w_init is "register" and use_bbr is None

    transforms = pe.Node(niu.Merge(2), run_without_submitting=True, name='transforms')
    reports = pe.Node(niu.Merge(2), run_without_submitting=True, name='reports')

    lta_ras2ras = pe.MapNode(LTAConvert(out_lta=True), iterfield=['in_lta'],
                             name='lta_ras2ras', mem_gb=2)
    compare_transforms = pe.Node(niu.Function(function=compare_xforms), name='compare_transforms')

    select_transform = pe.Node(niu.Select(), run_without_submitting=True, name='select_transform')
    select_report = pe.Node(niu.Select(), run_without_submitting=True, name='select_report')

    workflow.connect([
        (bbregister, transforms, [('out_lta_file', 'in1')]),
        (mri_coreg, transforms, [('out_lta_file', 'in2')]),
        # Normalize LTA transforms to RAS2RAS (inputs are VOX2VOX) and compare
        (transforms, lta_ras2ras, [('out', 'in_lta')]),
        (lta_ras2ras, compare_transforms, [('out_lta', 'lta_list')]),
        (compare_transforms, outputnode, [('out', 'fallback')]),
        # Select output transform
        (transforms, select_transform, [('out', 'inlist')]),
        (compare_transforms, select_transform, [('out', 'index')]),
        (select_transform, merge_ltas, [('out', 'in1')]),
        # Select output report
        (bbregister, reports, [('out_report', 'in1')]),
        (mri_coreg, reports, [('out_report', 'in2')]),
        (reports, select_report, [('out', 'inlist')]),
        (compare_transforms, select_report, [('out', 'index')]),
        (select_report, outputnode, [('out', 'out_report')]),
    ])

    return workflow

