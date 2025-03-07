import numpy as np
from typing import List
from spg_overlay.entities.wounded_person import WoundedPerson
from solutions.utils.pose import Position
from solutions.utils.dataclasses_config import TrackingParams

class TrackedWounded:
    """Stores encountered wounded person informations. Positions are in WORLD COORDINATES"""
    def __init__(self, position: np.ndarray, rescued=False):
        self.position: np.ndarray = position
        self.rescued: bool = rescued

class ExplorationTracker:
    """
    Stores important information encountered during map exploration.
    Wounded persons, no_com_zones, etc...
    Positions are in WORLD COORDINATES
    """
    def __init__(self):
        self.wounded_persons: List[TrackedWounded] = []

    def identify_wounded(self, wounded_sighting_position: np.ndarray, id_distance_threshold):
        """
        If there are wounded persons tracked in self.wounded_persons that are close enough to the wounded sighting position,
        return the closest one.
        Else return None.
        This function tackles the issue that we can't give global id to wounded persons, therefore we need to identify them through their finding position.
        """
        if not self.wounded_persons:
            return None
    
        closest_wounded = min(self.wounded_persons, 
                            key=lambda w: np.linalg.norm(w.position - wounded_sighting_position))
        
        # Check if within threshold distance
        if np.linalg.norm(closest_wounded.position - wounded_sighting_position) <= id_distance_threshold:
            return closest_wounded

    def add_wounded(self, wounded_sighting_position: np.ndarray):
        id_distance_threshold = TrackingParams.wounded_id_add_distance_threshold
        if self.identify_wounded(wounded_sighting_position, id_distance_threshold) is None:    # Avoid duplicates
            self.wounded_persons.append(TrackedWounded(wounded_sighting_position))

    def remove_wounded(self, wounded_sighting_position: np.ndarray):
        id_distance_threshold = TrackingParams.wounded_id_remove_distance_threshold
        wounded = self.identify_wounded(wounded_sighting_position, id_distance_threshold)
        if wounded is None:
            return None
        else:
            self.wounded_persons.remove(wounded)

    def are_there_unrescued_wounded(self):
        return any([not w.rescued for w in self.wounded_persons])