# main.py
from config import Config
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import folium


def main():
    # Load general config
    config = Config()
    # Load the CSV file
    df = pd.read_csv(config.csv_uri)  # Use the correct CSV

    # Remove rows with missing coordinates
    df = df.dropna(subset=['ns1:coordinates'])

    # Split 'ns1:coordinates' and extract longitude, latitude
    df[['longitude', 'latitude', 'altitude']] = (
        df['ns1:coordinates'].str.split(",", expand=True)[[0, 1, 2]].astype(float))

    # Create point geometry using longitude and latitude
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Calculate center for the map
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    initial_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    # Add every antenna to the map
    for idx, row in gdf.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Name: {row.get('ns1:name8', '')}, Altitude: {row['altitude']} m"
        ).add_to(initial_map)

    # Save the map
    initial_map.save("map_antennas.html")


if __name__ == "__main__":
    main()
