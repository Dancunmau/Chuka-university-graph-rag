import os
import neo4j
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PWD = os.getenv("NEO4J_PASSWORD")

driver = neo4j.GraphDatabase.driver(URI, auth=(USER, PWD))

def check_deps():
    with driver.session() as session:
        print("--- Programmes with NO Department ---")
        no_dep = session.run("MATCH (p:Programme) WHERE NOT (p)<-[:HAS_PROGRAMME]-(:Department) RETURN count(p) as cnt").data()
        print(no_dep)
        
        print("\n--- Direct Faculty->Programme links ---")
        direct = session.run("MATCH (f:Faculty)-[:HAS_PROGRAMME]->(p:Programme) RETURN count(p) as cnt").data()
        print(direct)
        
        print("\n--- Faculty->Department->Programme links ---")
        routed = session.run("MATCH (f:Faculty)-[:FACULTY_HAS_DEPARTMENT]->(d:Department)-[:HAS_PROGRAMME]->(p:Programme) RETURN count(p) as cnt").data()
        print(routed)
        
        print("\n--- Duplicate Programs in multiple Faculties ---")
        dups = session.run("""
        MATCH (f:Faculty)-[]-(p:Programme)
        WITH p, collect(distinct f.name) as facs
        WHERE size(facs) > 1
        RETURN p.name as prog, facs
        LIMIT 10
        """).data()
        for d in dups:
            print(f"{d['prog']} -> {d['facs']}")

if __name__ == "__main__":
    check_deps()
    driver.close()
