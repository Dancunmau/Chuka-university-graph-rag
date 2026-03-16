import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv('d:/Jupyter notebook/Graph rag/.env')
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PWD  = os.getenv("NEO4J_PASSWORD")

# KEYWORD MAPPING: Define core keywords for each department/faculty
# This acts as a semantic bridge for resources without course codes.
DEPARTMENT_KEYWORDS = {
    "Department of Agricultural Economics": ["agriculture", "agribusiness", "farm", "crop", "livestock", "soil", "tomato", "irrigation", "agri", "rural", "pigeon pea", "coffee", "banana"],
    "Department of Computer Science": ["computer", "software", "network", "algorithm", "data", "web", "internet", "wireless", "programming", "system", "information", "it", "digital"],
    "Department of Education": ["education", "teaching", "learning", "curriculum", "school", "student", "teacher", "pedagogy", "academic", "literacy"],
    "Department of Humanities": ["religion", "ethics", "society", "culture", "philosophy", "social", "history", "languages", "literature", "christian", "theology", "ethics"],
    "Department of Business": ["business", "management", "economic", "finance", "accounting", "marketing", "entrepreneurship", "corporate", "trade", "commerce", "stock"],
    "Department of Environmental Science": ["environment", "sustainable", "climate", "water", "pollution", "ecology", "nature", "conservation", "waste"],
    "Department of Plant Science": ["plant", "botany", "seed", "fruit", "physiology", "pathology", "horticulture"],
    "Department of Nursing": ["nursing", "health", "clinical", "patient", "medical", "disease", "medicine", "wellness"],
}

def link_resources():
    driver = GraphDatabase.driver(URI, auth=(USER, PWD))
    with driver.session() as session:
        print("Starting Semantic Community Linking...")
        
        total_created = 0
        for dept_name, keywords in DEPARTMENT_KEYWORDS.items():
            print(f"Processing {dept_name}...")
            
            # Cypher query to find unlinked items matching any keyword
            # We use toLower() for case-insensitive matching in Neo4j
            query = """
            UNWIND $keywords AS kw
            MATCH (ri:RepositoryItem)
            WHERE NOT (:CourseUnit)-[:HAS_RESOURCE]->(ri)
              AND toLower(ri.title) CONTAINS toLower(kw)
            
            // Match the Department node
            MATCH (d)
            WHERE (d:Department OR d:Faculty)
              AND toUpper(d.name) CONTAINS toUpper($dept_core)
            
            MERGE (ri)-[:BELONGS_TO_DEPARTMENT]->(d)
            RETURN count(ri) as count
            """
            
            # Simple heuristic: use the last part of the department name to find the node
            dept_core = dept_name.replace("Department of ", "").strip()
            
            res = session.run(query, keywords=keywords, dept_core=dept_core)
            created = res.single()['count']
            total_created += created
            print(f"  - Linked {created} items.")

        print(f"\nDone! Total [:BELONGS_TO_DEPARTMENT] relationships created: {total_created}")

    driver.close()

if __name__ == "__main__":
    link_resources()
