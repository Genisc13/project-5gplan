from config import Config


class BBUPool:
    def __init__(self, identification, rrh_map):
        config = Config()
        self.id = identification
        self.rrh_map = rrh_map
        self.connected_rrh = self.connect_rrh(config.num_RRHs_per_BBU)

    def connect_rrh_by_distance(self, num_rrh):
        connected_rrh_s = []
        return connected_rrh_s
