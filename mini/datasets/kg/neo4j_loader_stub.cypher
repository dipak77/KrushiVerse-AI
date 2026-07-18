// Neo4j loader stub — Sprint 8 W-KGBUILD
// Import nodes from KG_LATEST.json then edges.
// Example (manual / APOC):
//
// CALL apoc.load.json('file:///KG_LATEST.json') YIELD value
// UNWIND value.nodes AS n
// MERGE (x:Entity {id: n.id})
// SET x.label = n.label, x += n.properties;
//
// CALL apoc.load.json('file:///KG_LATEST.json') YIELD value
// UNWIND value.edges AS e
// MATCH (a:Entity {id: e.source}), (b:Entity {id: e.target})
// CALL apoc.create.relationship(a, e.relation, e.properties, b) YIELD rel
// RETURN count(rel);
//
// Prefer offline NetworkX GraphML for local GraphRAG until Neo4j is provisioned.
