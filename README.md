# PySamplingQuality
_Released under the GNU General Public License version 3 by MikeN12_

v25.11.16

Python tool to assess the sampling quality of Molecular Dynamics (MD) simulations using multiple trajectories

Checked version is released with the following modules:

1. Overlap calculation

        Generate_RMSD_Matrices()
        determineR_using_RMSD_distributions()
        Generate_EventCurves()
        Calc_Overlap()

2. Clustering

        Generate_Clustering()
        Merge_Clustering_different_Thresholds()
        Generate_Centers_GLOBAL_singles()
        Generate_CDE_to_File()
        Generate_Slope_Error()

3. Visualization

        Plot_ClusterProfile()
        Plot_Slope_Error_Plateau_NrClust()
        Plot_Overlap_VS_Threshold()
        Plot_HeatMap_1vs1()
        Plot_HeatMap_as_Dendro()
        Plot_Overlap_VS_Time()
        Plot_Overlap_VS_Cluster()

It is possible, to split trajectories into smaller pieces to fit into the memory of the computation node.

A tutorial and description is provided as a JuPyteR notebook __PySamplingQuality\_Tutorial.ipynb__. The tutorial contains the general syntax in two representations, and all necessary parameter descriptions. The corresponding files (configurations, results and plots) are located in __Tutorial/__.
