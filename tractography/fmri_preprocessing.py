### Essentially we first need to take the data, do the different techniques in the PDF
# I found, then based on that use nilearn to make the connectome using an atlas
# to join voxels in a region into a region, then we can use the connectome to
# do the machine learning
# Could either use fmriprep for this, or do it using FSL stuff

import matplotlib.pyplot as plt
from tract_helpers import *
import subprocess
import nibabel as nib
from nipype.interfaces.fsl import (BET, ExtractROI, FAST, FLIRT, ImageMaths,
                                   MCFLIRT, SliceTimer, Threshold)
from nipype.interfaces.spm import Smooth
from nipype.algorithms.rapidart import ArtifactDetect
from nipype import Workflow, Node, MapNode
from sc_functions import *

from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
import nipype.interfaces.io as nio

from IPython.display import Image


# Define the main function
def main():
    # ----------------------- Defining main folders and paths ----------------------- #
    # Get paths for DWI, T1 and FMRI, depending on whether we're in HPC or not
    hpc = False
    info_source, select_files, NIPYPE_OUTPUT_FOLDER = get_nipype_datasource(hpc)
    check_output_folders_with_subfolders(NIPYPE_OUTPUT_FOLDER, "Nipype")

    # ----------------------- Defining Nipype nodes and commands ----------------------- #
    # ExtractROI - skip dummy scans
    extract = Node(ExtractROI(t_min=4, t_size=-1, output_type='NIFTI'),
                name="extract")
    # MCFLIRT - motion correction
    mcflirt = Node(MCFLIRT(
                        mean_vol=True,
                        save_plots=True,
                        save_mats=True,
                        output_type="NIFTI"),
                    name = "mcflirt")
    # SliceTimer - slice timing correction
    slice_timer = Node(SliceTimer(
                        interleaved=True,
                        index_dir=False,
                        output_type="NIFTI",
                        time_repetition=0),
                    name = "slice_timer")
    # Smooth - smoothing
    smooth = Node(Smooth(), name="smooth")
    fwhm = [4, 8]
    smooth.iterables = ("fwhm", fwhm)
    # Artifact detection
    artifact_detect = Node(ArtifactDetect(
                            norm_threshold=2,
                            zintensity_threshold=3,
                            mask_type="spm_global",
                            parameter_source="FSL",
                            use_differences=[True, False],
                            plot_type="svg"),
                        name="artifact_detect")
    # BET - Skull stripping
    bet_anat = Node(BET(
                        frac=0.5,
                        mask=True,
                        outline=True,
                        robust=True,
                        output_type="NIFTI_GZ"),
                    name="bet_anat")
    # FAST - Segmentation
    segmentation = Node(FAST(output_type="NIFTI_GZ"), 
                        name="segmentation",
                        mem_gb=4)
    # Select WM segmentation file from segmentation output
    def get_wm(files):
        return files[-1]
    # Threshold - Thresholding
    threshold = Node(Threshold(thresh=0.5,
                                args="-bin",
                                output_type="NIFTI_GZ"),
                    name="threshold")
    # FLIRT - Pre-alignment of function to anatomical
    coreg_pre = Node(FLIRT(dof=6,
                           output_type="NIFTI_GZ"),
                    name="coreg_pre")
    # FLIRT - Coregistration of function to anatomical with BBR
    coreg_bbr = Node(FLIRT(dof=6,
                            cost="bbr",
                            output_type="NIFTI_GZ"),
                    name="coreg_bbr")
    # Apply coregistration warp to functional data
    applywarp = Node(FLIRT(interp="spline",
                            apply_isoxfm=4,
                            output_type="NIFTI"),
                    name="applywarp")
    # Apply coregistration warp to mean file
    applywarp_mean = Node(FLIRT(interp="spline",
                                apply_isoxfm=4,
                                output_type="NIFTI_GZ"),
                        name="applywarp_mean")

    # ----------------------- Create coregistration workflow and connect nodes ----------------------- #
    # Create coregistration workflow
    coregwf = Workflow(name="coregwf")

    coregwf_folder = os.path.join(NIPYPE_OUTPUT_FOLDER, "coregwf_dir")
    check_output_folders_with_subfolders(coregwf_folder, "Coregwf")

    coregwf.base_dir = coregwf_folder

    # Connect nodes of coregistration workflow
    coregwf.connect([(bet_anat, segmentation, [('out_file', 'in_files')]),
                    (segmentation, threshold, [(('partial_volume_files', get_wm),
                                                'in_file')]),
                    (bet_anat, coreg_pre, [('out_file', 'reference')]),
                    (threshold, coreg_bbr, [('out_file', 'wm_seg')]),
                    (coreg_pre, coreg_bbr, [('out_matrix_file', 'in_matrix_file')]),
                    (coreg_bbr, applywarp, [('out_matrix_file', 'in_matrix_file')]),
                    (bet_anat, applywarp, [('out_file', 'reference')]),
                    (coreg_bbr, applywarp_mean, [('out_matrix_file', 'in_matrix_file')]),
                    (bet_anat, applywarp_mean, [('out_file', 'reference')]),
                    ])
    
    # ----------------------- Create preprocessing workflow and connect nodes ----------------------- #
    # Create datasink or output folder for important outputs
    datasink_folder = os.path.join(NIPYPE_OUTPUT_FOLDER, "datasink_dir")
    check_output_folders_with_subfolders(datasink_folder, "Datasink")

    datasink = Node(DataSink(base_directory=NIPYPE_OUTPUT_FOLDER,
                            container="datasink_dir"),
                    name="datasink")
    ## Use the following DataSink output substitutions
    substitutions = [('_subject_id_', 'sub-'),
                    ('_fwhm_', 'fwhm-'),
                    ('_roi', ''),
                    ('_mcf', ''),
                    ('_st', ''),
                    ('_flirt', ''),
                    ('.nii_mean_reg', '_mean'),
                    ('.nii.par', '.par'),
                    ]
    subjFolders = [('fwhm-%s/' % f, 'fwhm-%s_' % f) for f in fwhm]
    substitutions.extend(subjFolders)
    datasink.inputs.substitutions = substitutions

    # Create preprocessing workflow
    preprocwf = Workflow(name="preprocwf")
    preprocwf.base_dir = os.path.join(NIPYPE_OUTPUT_FOLDER, "preprocwf_dir")

    # Connect nodes of preprocessing workflow
    preprocwf.connect([(info_source, select_files, [('subject_id', 'subject_id')]),
                        (select_files, extract, [('fmri', 'in_file')]),
                        (extract, mcflirt, [('roi_file', 'in_file')]),
                        (mcflirt, slice_timer, [('out_file', 'in_file')]),

                        (select_files, coregwf, [('t1', 'bet_anat.in_file'),
                                                ('t1', 'coreg_bbr.reference')]),
                        (mcflirt, coregwf, [('mean_img', 'coreg_pre.in_file'),
                                            ('mean_img', 'coreg_bbr.in_file'),
                                            ('mean_img', 'applywarp_mean.in_file')]),
                        (slice_timer, coregwf, [('slice_time_corrected_file', 'applywarp.in_file')]),
                        
                        (coregwf, smooth, [('applywarp.out_file', 'in_files')]),

                        (mcflirt, datasink, [('par_file', 'preproc.@par')]),
                        (smooth, datasink, [('smoothed_files', 'preproc.@smooth')]),
                        (coregwf, datasink, [('applywarp_mean.out_file', 'preproc.@mean')]),

                        (coregwf, artifact_detect, [('applywarp.out_file', 'realigned_files')]),
                        (mcflirt, artifact_detect, [('par_file', 'realignment_parameters')]),

                        (coregwf, datasink, [('coreg_bbr.out_matrix_file', 'preproc.@mat_file'),
                                            ('bet_anat.out_file', 'preproc.@brain')]),
                        (artifact_detect, datasink, [('outlier_files', 'preproc.@outlier_files'),
                                        ('plot_files', 'preproc.@plot_files')]),
                        ])
    
    # ----------------------- Run the workflow ----------------------- #
    # Visualize the workflow as simple and detailed graphs
    preprocwf.write_graph(graph2use='colored', format='png', simple_form=True)
    preprocwf.write_graph(graph2use='flat', format='png', simple_form=True)
    # Write the images to the working directory
    Image(filename=preprocwf.base_dir + "/preprocwf/graph.png")
    Image(filename=preprocwf.base_dir + "/preprocwf/graph_detailed.png")

    # Run the workflow
    preprocwf.run('MultiProc', plugin_args={'n_procs': 2})


# Get the data
if __name__ == '__main__':
    main()
