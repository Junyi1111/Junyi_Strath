import geopandas as gpd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from shapely.ops import linemerge, unary_union
from shapely.geometry import LineString
import pandas as pd


class SubstationDistanceAnalyzer:
    def __init__(self, hv_cable_path, primary_sub_path, secondary_sub_path, address="18 BATTLEFIELD ROAD"):
        # Load and preprocess the data
        self.hv_cable = gpd.read_file(hv_cable_path).to_crs(epsg=27700)
        self.primary_sub = gpd.read_file(primary_sub_path).to_crs(epsg=27700)
        self.secondary_sub = gpd.read_file(secondary_sub_path).to_crs(epsg=27700)

        # Extract the point for the specified address
        self.sub_point = self.primary_sub[self.primary_sub['SPADDR1'] == address].geometry.values[0]

        # Compute necessary data
        self.merged_hv_cables = linemerge(unary_union(self.hv_cable.geometry))
        self.G = self._create_graph_from_multilines(self.merged_hv_cables)
        self.nearest_source_node = self._find_nearest_node(self.G, (self.sub_point.x, self.sub_point.y))

    def _create_graph_from_multilines(self, multilines):
        G = nx.Graph()
        lines = multilines.geoms if hasattr(multilines, 'geoms') else [multilines]
        for line in lines:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                start_point = coords[i]
                end_point = coords[i + 1]
                G.add_node(start_point, geometry=start_point)
                G.add_node(end_point, geometry=end_point)
                line_segment = LineString([start_point, end_point])
                G.add_edge(start_point, end_point, geometry=line_segment, length=line_segment.length)
        return G

    def _find_nearest_node(self, graph, point_coords):
        nodes = np.array(graph.nodes())
        dist = np.linalg.norm(nodes - point_coords, axis=1)
        nearest_idx = np.argmin(dist)
        return tuple(nodes[nearest_idx])

    def get_distances(self, num_substations=12):
        path_based_distances = []
        self.secondary_sub['distance_to_primary'] = self.secondary_sub.geometry.distance(self.sub_point)
        self.closest_substations = self.secondary_sub.nsmallest(num_substations, 'distance_to_primary')

        for idx, row in self.closest_substations.iterrows():
            nearest_target_node = self._find_nearest_node(self.G, (row['geometry'].x, row['geometry'].y))
            try:
                shortest_path_nodes = nx.shortest_path(self.G, source=self.nearest_source_node,
                                                       target=nearest_target_node, weight='length')
                path_distance = sum(
                    self.G[u][v]['length'] for u, v in zip(shortest_path_nodes[:-1], shortest_path_nodes[1:]))
                path_based_distances.append((row['SPADDR1'], path_distance))
            except nx.NetworkXNoPath:
                path_based_distances.append((row['SPADDR1'], None))
        return pd.DataFrame(path_based_distances, columns=['Address', 'Path Distance (m)'])

    def visualize_network(self, num_substations=22):
        fig, ax = plt.subplots(figsize=(6, 6))
        self.hv_cable.plot(ax=ax, color='lightgray', linewidth=1, zorder=1, label="11kV Cable")
        ax.scatter(self.sub_point.x, self.sub_point.y, color='red', s=150, edgecolor='k', zorder=5,
                   label='33kV Primary Substation', marker='o')
        secondary_label_added = False
        plotted_substations = 0
        for idx, row in self.closest_substations.iterrows():
            if plotted_substations >= num_substations:
                break
            nearest_target_node = self._find_nearest_node(self.G, (row['geometry'].x, row['geometry'].y))
            try:
                shortest_path_nodes = nx.shortest_path(self.G, source=self.nearest_source_node,
                                                       target=nearest_target_node, weight='length')
                ax.scatter(row['geometry'].x, row['geometry'].y, color='blue', s=50,
                           label='11kV Secondary Substation' if not secondary_label_added else "", zorder=4, marker='o')
                secondary_label_added = True
                path_lines = []
                for u, v in zip(shortest_path_nodes[:-1], shortest_path_nodes[1:]):
                    path_lines.append(self.G[u][v]['geometry'])
                gpd.GeoSeries(path_lines).plot(ax=ax, color='blue', linewidth=1.5, zorder=3)
                plotted_substations += 1
            except nx.NetworkXNoPath:
                continue
        ax.legend(loc="upper right", fontsize=10, prop={'family': 'Times New Roman'})
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.text(self.sub_point.x + 600, self.sub_point.y - 800, "|--- 100m ---|", ha='center', va='center',
                fontsize=10, fontweight='bold', color='black',fontname="Times New Roman")
        ax.plot([self.sub_point.x + 1100, self.sub_point.x + 1300], [self.sub_point.y - 1700, self.sub_point.y - 1700],
                color='black', linewidth=2)
        ax.set_xlim(self.sub_point.x - 1000, self.sub_point.x + 1000)
        ax.set_ylim(self.sub_point.y - 1000, self.sub_point.y + 1000)
        plt.savefig('network_visualization.svg', dpi=300, bbox_inches='tight')
        plt.show()
