import spacy
from spacy.matcher import PhraseMatcher
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor
import json
from pyvis.network import Network
import networkx as nx
import pandas as pd
from graphdatascience import GraphDataScience
from py2neo import Graph, Node, Relationship
import networkx as nx
import uuid
import matplotlib.pyplot as plt
graph = Graph('bolt://localhost:7687', auth = None)

def vis_network(nodes, edges, filename, physics=False):
    html = """
    <html>
    <head>
      <link href="/home/azureuser/datadrive/SmartCareer/KG_unsupervised/vis/dist/vis.css" rel="stylesheet" type="text/css">
    </head>
    <body>
    <div id="{id}" style="height:435px"></div>
    <script type="text/javascript">
      var nodes = {nodes};
      var edges = {edges};
      var container = document.getElementById("{id}");
      var data = {{
        nodes: nodes,
        edges: edges
      }};
      var options = {{
          nodes: {{
              shape: 'dot',
              size: 25,
              font: {{
                  size: 14
              }}
          }},
          edges: {{
              font: {{
                  size: 14,
                  align: 'middle'
              }},
              color: 'gray',
              arrows: {{
                  to: {{enabled: true, scaleFactor: 0.5}}
              }},
              smooth: {{enabled: false}}
          }},
          physics: {{
              enabled: {physics}
          }}
      }};
      var network = new vis.Network(container, data, options);
    </script>
    </body>
    </html>
    """

    unique_id = str(uuid.uuid4())
    print(unique_id)
    html = html.format(id=unique_id, nodes=json.dumps(nodes), edges=json.dumps(edges), physics=json.dumps(physics))

    with open("/home/azureuser/datadrive/SmartCareer/KG_unsupervised/vis/dist/vis.js","r",encoding='utf-8') as fp:
        vis_js = fp.readline()

    
    html = "<script>" + vis_js  + "</script>" + html

    file = open(filename, "w")
    file.write(html)
    file.close()

def gen_query(target, itemid, number):
    template ='''MATCH (n1:{})-[r1]->(n2)<-[r2]-(n3:{})-[r3]->(n4)
        WHERE n1.{} = "{}"
        RETURN n1,r1,n2,r2,n3,r3,n4,rand() as r
        ORDER BY r
        Limit 10
        '''
    if target == "Jobs":
        return template.format("Freelancer", "Job", "Id", str(itemid))
    if target == "Candidates":
        return template.format("Job", "Freelancer", "Title", str(itemid))
    
def draw(query, filename, physics=False):
    # The options argument should be a dictionary of node labels and property keys; it determines which property
    # is displayed for the node label. For example, in the movie graph, options = {"Movie": "title", "Person": "name"}.
    # Omitting a node label from the options dict will leave the node unlabeled in the visualization.
    # Setting physics = True makes the nodes bounce around when you touch them!
    """
    Args:
        data: py2neo result with source_node,  r, target_node
    """
    result = graph.run(query)
    print(query)
    data = result.to_subgraph()
    if data is None:
        raise ValueError('data must be provided')

    nodes = []
    edges = []

    def get_vis_info(node, *args):
        # print('node.labels:',node.labels)
        if str(node.labels) == ':Freelancer':
            node_label = [x for x in list(node.labels) if x != 'Hetio'][0]
            vis_label = node['Id']
            return {"id": vis_label, "label": vis_label, "group": node_label}
        elif str(node.labels) == ':Job':
            node_label = [x for x in list(node.labels) if x != 'Hetio'][0]
            vis_label = node['Title']
            return {"id": vis_label, "label": vis_label, "group": node_label}
        else:
            node_label = [x for x in list(node.labels) if x != 'Hetio'][0]
            vis_label = node['Name']
            return {"id": vis_label, "label": vis_label, "group": node_label}
            

    for row in data:
        source_node = row.start_node
        # print('source_node:',source_node)
        rel = row
        target_node = row.end_node
        # print('target_node:',target_node)
        source_info = get_vis_info(source_node)
        # print('source_info:',source_info)
        if source_info not in nodes:
            nodes.append(source_info)

        if rel is not None:
            target_info = get_vis_info(target_node)
            # print('target_info:',target_info)
            if target_info not in nodes:
                nodes.append(target_info)

            edges.append({"from": source_info["id"], "to": target_info["id"], "label": type(rel).__name__})

    return vis_network(nodes, edges, filename, physics=physics)

with open("skill_db_relax_20.json", "r") as f:
    skill_type_mapping = json.load(f)
skill_type_mapping = {jid:(skill_type_mapping[jid]["skill_name"], skill_type_mapping[jid]["skill_type"]) for jid in skill_type_mapping}

nlp = spacy.load("en_core_web_lg")
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
def desc_to_html(description, html_path):
    color_dict = {"Soft Skill":"#698269", "Certification":"#B99B6B", "Hard Skill":"#AA5656"}
    annotations = skill_extractor.annotate(description)
    ner_types = ['full_matches', 'ngram_scored'] 
    all_skill_and_types = [skill_type_mapping[jid['skill_id']] for ner_type in ner_types if ner_type in annotations['results'] 
                           for jid in annotations['results'][ner_type]]
    print(all_skill_and_types)
    G = nx.DiGraph()
    G.add_node("item", type = "Subject", size=25)
    for skill_type_tuple in all_skill_and_types:
        G.add_node(skill_type_tuple[0],type = skill_type_tuple[1], color = color_dict[skill_type_tuple[1]], size=25)
        G.add_edge("item", skill_type_tuple[0])

    demo_net = Network(height='465px', bgcolor='white', font_color='black')
    demo_net.from_nx(G)
    # Generate network with specific layout settings
    demo_net.repulsion(node_distance=420, central_gravity=0.33,
                       spring_length=110, spring_strength=0.10,
                       damping=0.95)
    demo_net.save_graph(html_path)

if __name__ == "__main__":
    jd = "I am a Python developer with a solid experience in web development and got Project Management Professional Certification. I quickly adapt to new environments and speak fluently English and French."
    desc_to_html(jd, "demo_net.html")
# docker run \
# --publish=7474:7474 --publish=7687:7687 \
# --volume=$HOME/datadrive/neo4j/data:/data \
# --env=NEO4J_AUTH=none \
# neo4j