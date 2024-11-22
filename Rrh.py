import folium


class RRH:
    def __init__(self, latitude, longitude, altitude, r_type, identifier):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.r_type = r_type
        self.identifier = identifier
        self.marker = self.generate_marker()

    def generate_marker(self):
        return folium.Marker(
            icon=folium.Icon(icon="tower-cell", prefix="fa",
                             color=self.r_type, icon_color="black"),
            location=[self.latitude, self.longitude],
            popup=f"Name: {self.identifier}, Altitude: {self.altitude} m"
        )
