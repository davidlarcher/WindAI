Wind-AI applies Reinforcement Learning to wind turbine farm management.

The AI framework is based on ray `ray lib <https://github.com/raysan5/raylib>`_ and the nevironment simulator is based on Floris from NREL https://github.com/NREL/floris
----------------------------

**more on our work at https://intelliance.ai/.**


Background and Objectives
=========================
The number of yaw positions in a wind turbine farm can exceed the number of atomes in the universe. Therefore analytic methods are overwelmed : We use Deep Reinforcement Learning to optimize yaw control for wake steering and global wind turbine farm optimization



.. _installation:

Installation
============

.. code-block:: bash

    # Download the source code (including Floris)
    git clone https://github.com/davidlarcher/WindAI.git

    # Install ray
    pip install ray

.. _references:

References
=======

This work is done based on :
- FLORIS. Version 2.2.0 (2020). Available at https://github.com/NREL/floris.
- RAY Available at https://github.com/ray-project/ray


License
=======

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
