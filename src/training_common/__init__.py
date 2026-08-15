"""Robot-agnostic training primitives shared by all task packages.

No module here may hard-code a robot's joint order, mass, contact names or
PPO experiment settings. Those belong to ``themis_training``, ``g1_training``
and ``jingchu01_training`` respectively.
"""
