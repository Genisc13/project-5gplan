class Config:
    def __init__(self):
        self.num_users = 100
        self.coverage_radius = 9500  # in m
        self.distance_between_BBUs = 1000  # in m
        self.bbu_capacity = 2000  # in Mbps
        # Test_mode:
        # 0 For test taking into account distance
        # 1 For test taking into account number of RRHs
        # 2 For test taking into account traffic and distance
        # 3 For test having traffic load balancing
        self.test_mode = 3
        self.simulation_hour = -1
        self.traffic_csv_uri = "average_traffic_v2.csv"
        self.csv_uri = "antennas_kml_v2.csv"
        self.num_RRHs = 102
        self.num_RRHs_per_BBU = 25
        self.num_BBU_pools = 20
        self.percentage_RRH_mixed = 50
        self.percentage_RRH_office = 20
        self.percentage_RRH_residential = 30
