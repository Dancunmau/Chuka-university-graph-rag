import os
import neo4j
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PWD = os.getenv("NEO4J_PASSWORD")

driver = neo4j.GraphDatabase.driver(URI, auth=(USER, PWD))

def debug():
    with driver.session() as session:
        # 1. Investigate Faculty of Nursing -> BSc Agriculture paths
        print("=== Paths between Nursing and Agriculture ===")
        query1 = """
        MATCH p=shortestPath((f:Faculty)-[*1..4]-(prog:Programme))
        WHERE f.name CONTAINS 'Nursing' AND prog.name CONTAINS 'Agriculture'
        RETURN [n IN nodes(p) | coalesce(n.name, n.code, labels(n)[0])] as path,
               [r IN relationships(p) | type(r)] as rels
        LIMIT 5
        """
        for r in session.run(query1).data():
            print(f"Path: {r['path']}")
            print(f"Rels: {r['rels']}")

        # 2. Check if Department nodes exist and their relationships
        print("\n=== Sample Department Nodes ===")
        query2 = """
        MATCH (d:Department)
        MATCH (f:Faculty)-[r1]-(d)-[r2]-(p:Programme)
        RETURN f.name as fac, type(r1) as f_to_d, d.name as dep, type(r2) as d_to_p, p.name as prog
        LIMIT 5
        """
        deps = session.run(query2).data()
        for d in deps:
            print(f"{d['fac']} --[{d['f_to_d']}]-- {d['dep']} --[{d['d_to_p']}]-- {d['prog']}")
            
        if not deps:
            print("No Faculty-Department-Programme paths found. Let's look for any Department isolated:")
            iso = session.run("MATCH (d:Department) RETURN d.name LIMIT 5").data()
            print([i['name'] for i in iso])

if __name__ == "__main__":
    debug()
    driver.close()
