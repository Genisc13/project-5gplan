class Config:
    def __init__(self):
        self.num_users = 100
        self.coverage_radius = 1000  # en metros
        self.base_station_capacity = 10000  # in Mbps
        self.csv_uri = "antennas_kml_v2.csv"
        self.num_RRHs = 102
        self.num_RRHs_per_BBU = 3
        self.num_BBU_pools = 20
        self.percentage_RRH_mixed = 50
        self.percentage_RRH_office = 20
        self.percentage_RRH_residential = 30
