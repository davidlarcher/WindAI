# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


import os

import numpy as np
import matplotlib.pyplot as plt


from WindAI.floris import tools as wfct
from WindAI.floris.tools import visualization as vis
#from tools.optimization.scipy.yaw import YawOptimization


def farminit(num_wt):
    print("Running FLORIS with no yaw...")
    # Instantiate the FLORIS object
    file_dir = os.path.dirname(os.path.abspath(__file__))
    print(file_dir)
    fi = wfct.floris_interface.FlorisInterface(
        os.path.join(file_dir, "../examples/example_input.json")
    )
    # Set turbine locations to 2 turbines in a row
    D = fi.floris.farm.turbines[0].rotor_diameter
    print(num_wt)
    layout_x = [7 * D * i for i in range(0, num_wt)]
    layout_y = [0] * num_wt
    print(layout_x, layout_y)
    fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))
    return fi



def plotfarm(fi):
    print("Plotting the FLORIS flowfield...")
    # =============================================================================
    # Initialize the horizontal cut
    hor_plane = fi.get_hor_plane(x_resolution=400, y_resolution=100)
    # Plot and show
    fig, ax = plt.subplots()
    wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
    ax.set_title("Baseline Case for U = 8 m/s, Wind Direction = 270$^\circ$")
    plt.show()


def yawptimize():


    # =============================================================================

    # =============================================================================
    print("Finding optimal yaw angles in FLORIS...")
    # =============================================================================
    # Set bounds for allowable wake steering
    min_yaw = 0.0
    max_yaw = 25.0
    # Instantiate the Optimization object
    yaw_opt = YawOptimization(fi, minimum_yaw_angle=min_yaw, maximum_yaw_angle=max_yaw)
    # Perform optimization
    yaw_angles = yaw_opt.optimize()
    print("==========================================")
    print("yaw angles = ")
    for i in range(len(yaw_angles)):
        print("Turbine ", i, "=", yaw_angles[i], " deg")
    # Assign yaw angles to turbines and calculate wake
    fi.calculate_wake(yaw_angles=yaw_angles)
    power_opt = fi.get_farm_power()
    print("==========================================")
    print(
        "Total Power Gain = %.1f%%" % (100.0 * (power_opt - power_initial) / power_initial)
    )
    print("==========================================")
    # =============================================================================
    print("Plotting the FLORIS flowfield with yaw...")
    # =============================================================================
    # Initialize the horizontal cut
    hor_plane = fi.get_hor_plane(x_resolution=400, y_resolution=100)
    # Plot and show
    fig, ax = plt.subplots()
    wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
    ax.set_title("Optimal Wake Steering for U = 8 m/s, Wind Direction = 270$^\circ$")
    plt.show()


