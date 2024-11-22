# main.py
from config import Config
from Bbu_pool import BBUPool
from Rrh import RRH
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import random
import folium


def randomize_residential_office(config):
    random_value = random.random()
    color = "orange"
    # For the mixed ones
    if random_value <= config.percentage_RRH_mixed / 100:
        color = "orange"
    # For the ones that are for office
    elif (config.percentage_RRH_mixed / 100) < random_value <= (
            (config.percentage_RRH_mixed / 100) + config.percentage_RRH_office / 100):
        color = "blue"
    # For the residential ones
    elif ((config.percentage_RRH_mixed / 100 + config.percentage_RRH_office / 100) <
          random_value <= 1):
        color = "green"
    return color


def main():
    # Load general config
    config = Config()
    # Load the CSV file
    df = pd.read_csv(config.csv_uri)  # Use the correct CSV
    bbu_pools = []
    # Remove rows with missing coordinates
    df = df.dropna(subset=['ns1:coordinates'])
    antennas_map = []
    # Split 'ns1:coordinates' and extract longitude, latitude
    df[['longitude', 'latitude', 'altitude']] = (
        df['ns1:coordinates'].str.split(",", expand=True)[[0, 1, 2]].astype(float))

    # Create point geometry using longitude and latitude
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Calculate center for the map
    center_lat = df['latitude'].mean()
    # print(center_lat)
    center_lon = df['longitude'].mean()
    # print(center_lon)
    initial_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    # Add every antenna to the map
    '''for idx, row in gdf.iterrows():
        antennas_map.append(RRH(row['latitude'], row['longitude'], row['altitude'],
                                randomize_residential_office(config), row['ns1:name9']))'''
    for idx, row in gdf.iterrows():
        # print([row['latitude'], row['longitude']])
        folium.Marker(
            icon=folium.Icon(icon="tower-cell", prefix="fa",
                             color=randomize_residential_office(config), icon_color="black"),
            location=[row['latitude'], row['longitude']],
            popup=f"Name: {row.get('ns1:name9', '')}, Altitude: {row['altitude']} m"
        ).add_to(initial_map)
    '''for rrh in antennas_map:
        rrh.marker.add_to(initial_map)'''

    # Save the map
    initial_map.save("map_antennas.html")


if __name__ == "__main__":
    main()
