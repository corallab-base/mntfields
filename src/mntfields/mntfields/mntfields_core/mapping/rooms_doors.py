import numpy as np
from typing import List, Set
from mntfields.utils.mntfields_utils import percent_overlap

class Room:
    def __init__(self, mask):
        mask_idx_i, mask_idx_j = np.where(mask)
        self.imin = mask_idx_i.min()
        self.imax = mask_idx_i.max()
        self.jmin = mask_idx_j.min()
        self.jmax = mask_idx_j.max()
        self.mask: np.ndarray = mask
        self.rect = (self.imin, self.jmin, self.imax, self.jmax)
    
    def has_overlap(self, other):
        other: Room = other
        return percent_overlap(self.rect, other.rect) > 0 #uses percent_overlap from utils
        
    def contains(self, point):
        return self.mask[point[0], point[1]]

class ConfirmedRoom(Room):
    n_confirmed_rooms = 0

    def __init__(self, mask):
        super().__init__(mask)
        self.doors: List[Door] = []
        ConfirmedRoom.n_confirmed_rooms += 1
        id = ConfirmedRoom.n_confirmed_rooms
        self.id = id

        self.region: Region = None
    
    def set_region(self, region):
        self.region = region

    def add_doors(self, doors):
        self.doors += doors
    
    def find_door(self, pos):
        for door in self.doors:
            if door.pos == pos:
                return door
        return None

    def serialize(self):
        return {
            'id': self.id,
            'doors': [door.serialize() for door in self.doors],
            'region': None if self.region is None else self.region.version
        }

class UnconfirmedRoom(Room):
    n_confirmed_rooms = 1000
    def __init__(self, mask, id=None):
        super().__init__(mask)
        self.doors: List[Door] = []
        self.region: Region = None
        UnconfirmedRoom.n_confirmed_rooms += 1
        if id is None:
            self.id = UnconfirmedRoom.n_confirmed_rooms
        else:
            self.id = int(id)
    
    def set_region(self, region):
        self.region = region

    def add_doors(self, doors):
        self.doors += doors
    
    def serialize(self):
        return {
            'id': int(self.id),
            'doors': [door.serialize() for door in self.doors],
            'region': None if self.region is None else self.region.version
        }

class Door:
    n_doors = 0
    def __init__(self, id, pos, room, mask=None, rooms_in_door=None):
        self.id = id
        self.pos = pos
        self.room = room
        self.mask = mask
        self.rooms_in_door: Set[int] = rooms_in_door
        Door.n_doors += 1
    
    def serialize(self):
        return {
            'id': self.id,
            'pos': [float(x) for x in self.pos],
            'room': int(self.room.id)
        }
