import math
import random
import sys
from typing import Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle


from maps.map_intermediate_01 import MyMapIntermediate01
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.gui_map.gui_sr import GuiSR
from spg_overlay.utils.utils import normalize_angle

import math
import random
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.utils import normalize_angle


class SimpleRescueDrone(DroneAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_turning = False
        self.target_angle = 0

    def define_message_for_all(self):
        """
        No communication is needed for this basic implementation.
        """
        pass

    def control(self):
        """
        Control logic:
        - Explores by moving toward the most open area based on LIDAR.
        - Grabs a wounded person when detected within range.
        """
        # Default movement commands
        command = {"forward": 1.0, "rotation": 0.0, "grasper": 0}

        # Process semantic sensor for entities
        semantic_values = self.semantic_values()
        if semantic_values:
            for data in semantic_values:
                # Check for wounded person
                if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and data.distance < 50:
                    # Attempt to grab the wounded person
                    command["forward"] = 0.0
                    command["rotation"] = 0.0
                    command["grasper"] = 1
                    return command

        # Use LIDAR to determine open areas
        lidar_values = self.lidar_values()
        if lidar_values.any():
            # Get the angle with the maximum distance
            self.target_angle = lidar_values.argmax(lidar_values)

        # Align drone to the target angle
        current_angle = self.measured_compass_angle()
        diff_angle = normalize_angle(self.target_angle - current_angle)

        if abs(diff_angle) > 0.2:  # Rotate to align with target angle
            command["rotation"] = 1.0 if diff_angle > 0 else -1.0
            command["forward"] = 0.0  # Stop forward motion while turning

        return command
