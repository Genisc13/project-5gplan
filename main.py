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

    # Remove rows with missing latitude or longitude
    df = df.dropna(subset=['ns1:latitude', 'ns1:longitude'])

    # Create point geometry
    geometry = [Point(xy) for xy in zip(df['ns1:longitude'], df['ns1:latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)

    # Create a map using the average location of the antennas
    center_lat = df['ns1:latitude'].mean()
    center_lon = df['ns1:longitude'].mean()
    initial_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    # Add every antenna on the map
    for idx, row in gdf.iterrows():
        folium.Marker(
            location=[row['ns1:latitude'], row['ns1:longitude']],
            popup=row['ns1:coordinates']
        ).add_to(initial_map)

    # Show the map
    initial_map.save("map_antennas.html")


if __name__ == "__main__":
    main()
