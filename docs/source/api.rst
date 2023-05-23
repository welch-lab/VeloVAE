.. automodule:: velovae

API
===

Import VeloVAE as::

    import velovae as vv


Preprocessing
------------------

.. autosummary::
   :toctree: .

   preprocess

Models
------------------

.. toctree::
    model

Velocity Utility
-----------------------------
.. autosummary::
   :toctree: .
   
   rna_velocity_vanillavae
   velovae.rna_velocity_vae
   velovae.rna_velocity_brode

Analysis
------------------

.. autosummary::
   :toctree: .

   post_analysis
   cross_boundary_correctness
   time_score
   inner_cluster_coh
   velocity_consistency
   gen_cross_boundary_correctness
   gen_cross_boundary_correctness_test

Performance Logger
------------------

.. autosummary:: 
   :toctree: .

   PerfLogger
   PerfLogger.insert
   PerfLogger.plot
   PerfLogger.save

Plotting Utility
------------------

.. autosummary::
   :toctree: .

   set_dpi
   get_colors
   plot_sig
   plot_phase
   plot_cluster
   cellwise_vel
   cellwise_vel_embedding
   plot_phase_vel
   plot_velocity
   plot_legend
   plot_heatmap
   plot_time
   plot_time_var
   plot_state_var
   plot_phase_grid
   plot_sig_grid
   plot_time_grid
   plot_rate_grid
   plot_trajectory_3d
   plot_transition_graph
   plot_rate_hist