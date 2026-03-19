import os
import neo4j
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PWD = os.getenv("NEO4J_PASSWORD")

driver = neo4j.GraphDatabase.driver(URI, auth=(USER, PWD))

def check_structure():
    with driver.session() as session:
        # Check Faculty -> Programme relationships
        query = """
        MATCH (f)-[:HAS_DEPARTMENT|HAS_PROGRAMME|BELONGS_TO|IN_FACULTY|HAS_UNIT*1..3]->(p:Programme)
        WHERE any(label IN labels(f) WHERE label CONTAINS 'Faculty')
        RETURN DISTINCT f.name as fac, p.name as prog
        """
        results = session.run(query).data()
        print("FACULTY -> PROGRAMME MAPPINGS:")
        for r in results:
            print(f"  {r['fac']} -> {r['prog']}")
        
        # Check all unique faculties
        query_fac = "MATCH (f) WHERE any(label IN labels(f) WHERE label CONTAINS 'Faculty') RETURN DISTINCT f.name as name"
        facs = session.run(query_fac).data()
        print("\nALL FACULTIES FOUND:")
        for f in facs:
            print(f"  - {f['name']}")

if __name__ == "__main__":
    check_structure()
    driver.close()
