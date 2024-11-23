from config import Config


class BBUPool:
    def __init__(self, identifier, latitude, longitude, connected_rrh):
        config = Config()
        self.identifier = identifier
        self.latitude = latitude
        self.longitude = longitude
        self.connected_rrh = connected_rrh
