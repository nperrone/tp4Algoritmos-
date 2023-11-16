import pickle
from graph import Graph
from tqdm import tqdm

"""
Lista de productos

Ejemplo de producto:
{'id': 2,
 'title': 'Candlemas: Feast of Flames',
 'group': 'Book',
 'categories': ['Books[283155]->Subjects[1000]->Religion & Spirituality[22]->Earth-Based Religions[12472]->Wicca[12484]',
  'Books[283155]->Subjects[1000]->Religion & Spirituality[22]->Earth-Based Religions[12472]->Witchcraft[12486]'],
 'reviewers': [('A11NCO6YTE4BTJ', 5),
  ('A9CQ3PLRNIR83', 4),
  ('A13SG9ACZ9O5IM', 5),
  ('A1BDAI6VEYMAZA', 5),
  ('A2P6KAWXJ16234', 4),
  ('AMACWC3M7PQFR', 4),
  ('A3GO7UV9XX14D8', 4),
  ('A1GIL64QK68WKL', 5),
  ('AEOBOF2ONQJWV', 5),
  ('A3IGHTES8ME05L', 5),
  ('A1CP26N8RHYVVO', 1),
  ('ANEIANH0WAT9D', 5)]}
"""
with open('products.pickle', 'rb') as file:
    products = pickle.load(file)

grafo = Graph()

print("Loading")
for p in tqdm(products):
    grafo.add_vertex(str(p["id"]), data={'title': p['title'],
                                         'group': p['group'],
                                         'categories': p['categories']})
    for reviewer, score in p['reviewers']:
        if not grafo.vertex_exists(reviewer):
            grafo.add_vertex(reviewer)
        grafo.add_edge(reviewer, str(p["id"]), score)
        grafo.add_edge(str(p["id"]), reviewer, score)

grafo.print_graph()

# Desarrolle el tp aquí