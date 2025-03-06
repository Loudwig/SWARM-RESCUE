from spg_overlay.entities.wounded_person import WoundedPerson
from utils.pose import Position
from solutions.utils.dataclasses_config import TrackingParams

class TrackedWounded:
    """Stores encountered wounded person informations. Positions are in WORLD COORDINATES"""
    def __init__(self, position, rescued=False):
        self.position = position
        self.rescued = rescued

class TrackedNoComZone:
    """
    CHATGPT GENERATED
    
    Stores encountered no communication zone informations. Positions are in WORLD COORDINATES"""
    def __init__(self, position):
        self.position = position

class ExplorationTracker:
    """
    Stores important information encountered during map exploration.
    Wounded persons, no_com_zones, etc...
    Positions are in WORLD COORDINATES
    """
    def __init__(self):
        self.wounded_persons = []
        self.no_com_zones = []

    def add_wounded(self, wounded_sighting_position):
        self.wounded_persons.append(TrackedWounded(wounded_sighting_position))

    def identify_wounded(self, wounded_sighting_position):
        """
        Links a wounded person tracked in self.wounded_persons to a wounded sighting position
        """
        if not self.wounded_persons:
            return None
    
        closest_wounded = min(self.wounded_persons, 
                            key=lambda w: w.position.distance_to(wounded_sighting_position))
        
        # Check if within threshold distance
        if closest_wounded.position.distance_to(wounded_sighting_position) < TrackingParams.wounded_id_distance_threshold:
            return closest_wounded

    def remove_wounded(self, wounded_sighting_position):
        
    
    def add_no_com_zone(self, position):
        """
        CHATGPT GENERATED

        Add a no communication zone position"""
        self.no_com_zones.append(position)

    def get_nearest_wounded(self, position):
        """
        CHATGPT GENERATED

        Find closest non-rescued wounded person"""
        if not self.wounded_persons:
            return None
        
        available_wounded = [w for w in self.wounded_persons if not w.rescued]
        if not available_wounded:
            return None
            
        return min(available_wounded, 
                  key=lambda w: w.position.distance_to(current_position))