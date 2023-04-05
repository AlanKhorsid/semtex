from SPARQLWrapper import SPARQLWrapper, JSON
import re

sparql = SPARQLWrapper("http://dbpedia.org/sparql")
sparql.setReturnFormat(JSON)
queryStringPrefixes = "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n" \
                    "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n" \
                    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n" \
                    "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n" \
                    "PREFIX foaf: <http://xmlns.com/foaf/0.1/>\n" \
                    "PREFIX dc: <http://purl.org/dc/elements/1.1/>\n" \
                    "PREFIX : <http://dbpedia.org/resource/>\n" \
                    "PREFIX dbpedia2: <http://dbpedia.org/property/>\n" \
                    "PREFIX dbpedia: <http://dbpedia.org/>\n" \
                    "PREFIX skos: <http://www.w3.org/2004/02/skos/core#>\n"

def dbpedia_sparql_to_JSON(queryIdentifier:int , queryName:str) -> [str]:
    querystring = f"{queryStringPrefixes}SELECT * WHERE {{?wikidataquery owl:sameAs wikidata:Q{queryIdentifier} . ?wikidataquery rdf:type ?type . }}"

    sparql.setQuery(querystring)
    identifiers = []

    try:
        results = sparql.queryAndConvert()
        if len(results["results"]["bindings"]) == 0:
            results = dbpedia_queryname_lookup(queryName)

        for res in results["results"]["bindings"]:
            if "wikidata" in res["type"]["value"]:
                identifiers.append(re.search("Q[0-9]*", res["type"]["value"]).group())
    except Exception as e:
        print(e)

    return identifiers

def dbpedia_queryname_lookup(queryName):
    querystring = f"{queryStringPrefixes}SELECT ?type WHERE {{?wikidataquery rdfs:label \"{queryName}\"@en . ?wikidataquery rdf:type ?type .}}"
    sparql.setQuery(querystring)
    return sparql.queryAndConvert()

if __name__ == "__main__":
    test = dbpedia_sparql_to_JSON(1176, "Barack Obama")
    for v in test:
        print(v)
