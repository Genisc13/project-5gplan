# main.py
from config import Config
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import folium


def main():
    # Load general config
    config = Config()
    # update the value of the antennas
    df = pd.read_csv(config.csv_uri)  # Use the correct csv

    # Create point geometry
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)

    # Create a map using the average location of the antennas
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    initial_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    # Add every antenna on the map
    for idx, row in gdf.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=row['antenna_type']
        ).add_to(initial_map)

    # Show the map
    initial_map.save("map_antennas.html")


if __name__ == "__main__":
    main()
