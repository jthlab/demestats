demestats Core Functions API Reference
==============

.. automodule:: demestats
   :no-members:
   :no-special-members:

Model Constraint
----------

.. placeholder autofunction:: demestats.event_tree.EventTree
.. autofunction:: demestats.event_tree.EventTree.variables
.. autofunction:: demestats.event_tree.EventTree.variable_for
.. autofunction:: demestats.constr.constraints_for
.. autofunction:: demestats.constr.print_constraints
.. autofunction:: demestats.fit.util.alternative_constraint_rep
.. autofunction:: demestats.fit.util.modify_constraints_for_equality
.. autofunction:: demestats.util.create_inequalities
.. autofunction:: demestats.util.joint_sfs_from_haploids

momi3 (SFS) Statistics
--------------

.. autofunction:: demestats.sfs.ExpectedSFS
.. autofunction:: demestats.sfs.ExpectedSFS.__call__
.. autofunction:: demestats.loglik.sfs_loglik.prepare_projection
.. autofunction:: demestats.sfs.ExpectedSFS.tensor_prod

ICR Statistics
--------------

.. autofunction:: demestats.iicr.IICRCurve
.. autofunction:: demestats.iicr.IICRCurve.__call__
.. autofunction:: demestats.iicr.mf.IICRMeanFieldCurve
.. autofunction:: demestats.iicr.mf.IICRMeanFieldCurve.__call__
   

CCR Statistics
--------------

.. autofunction:: demestats.ccr.exact.CCRCurve
.. autofunction:: demestats.ccr.mf.CCRMeanFieldCurve
.. autofunction:: demestats.ccr.curve.__call__
