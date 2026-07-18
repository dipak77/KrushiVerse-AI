import networkx as nx
from app.knowledge.dataset_loader import kb_loader

class GraphRAGStore:
    """Knowledge Graph RAG engine using NetworkX for multi-entity agricultural reasoning."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self._load_graph()

    def _load_graph(self):
        graph_data = kb_loader.graph_data
        if not graph_data:
            return

        for node in graph_data.get("nodes", []):
            self.graph.add_node(
                node["id"],
                label=node.get("label"),
                **node.get("properties", {})
            )

        for edge in graph_data.get("edges", []):
            self.graph.add_edge(
                edge["source"],
                edge["target"],
                relation=edge.get("relation")
            )

    def query_graph_for_entity(self, entity_name: str) -> dict:
        """Find graph neighbors and 1-hop / 2-hop subgraphs for a given crop/disease/soil entity."""
        matches = [n for n in self.graph.nodes if entity_name.lower() in str(n).lower()]
        if not matches:
            return {"entity": entity_name, "found": False, "neighbors": []}

        target = matches[0]
        neighbors = []

        # Outgoing edges
        for neighbor in self.graph.successors(target):
            edge_data = self.graph.get_edge_data(target, neighbor)
            neighbor_node = self.graph.nodes[neighbor]
            neighbors.append({
                "direction": "outgoing",
                "relation": edge_data.get("relation"),
                "target": neighbor,
                "label": neighbor_node.get("label"),
                "properties": {k: v for k, v in neighbor_node.items() if k != "label"}
            })

        # Incoming edges
        for predecessor in self.graph.predecessors(target):
            edge_data = self.graph.get_edge_data(predecessor, target)
            predecessor_node = self.graph.nodes[predecessor]
            neighbors.append({
                "direction": "incoming",
                "relation": edge_data.get("relation"),
                "source": predecessor,
                "label": predecessor_node.get("label"),
                "properties": {k: v for k, v in predecessor_node.items() if k != "label"}
            })

        return {
            "entity": target,
            "found": True,
            "label": self.graph.nodes[target].get("label"),
            "properties": {k: v for k, v in self.graph.nodes[target].items() if k != "label"},
            "neighbors": neighbors
        }

    def get_crop_ecosystem(self, crop_name: str) -> dict:
        """Traverse graph to retrieve pests, soil requirements, fertilizers, and eligible government schemes."""
        entity_info = self.query_graph_for_entity(crop_name)
        if not entity_info["found"]:
            return {"error": f"Crop {crop_name} not found in Knowledge Graph."}

        pests = []
        soils = []
        fertilizers = []
        schemes = []

        for rel in entity_info["neighbors"]:
            target = rel.get("target") or rel.get("source")
            relation = rel.get("relation")

            if relation == "AFFECTED_BY":
                pests.append(target)
            elif relation == "GROWS_IN":
                soils.append(target)
            elif relation == "REQUIRES_FERTILIZER":
                fertilizers.append(target)
            elif relation == "COVERED_BY" or relation == "BENEFITS_FROM":
                schemes.append(target)

        return {
            "crop": entity_info["entity"],
            "pests_and_diseases": pests,
            "soil_types": soils,
            "recommended_fertilizers": fertilizers,
            "applicable_schemes": schemes
        }

graph_rag = GraphRAGStore()
