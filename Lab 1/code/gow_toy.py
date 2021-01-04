import string
from nltk.corpus import stopwords


#import os
#os.chdir() # to change working directory to where functions live
# import custom functions
from library import clean_text_simple, terms_to_graph, core_dec
from igraph import plot

stpwds = stopwords.words('english')
punct = string.punctuation.replace('-', '')

my_doc = """A method for solution of systems of linear algebraic equations \
with m-dimensional lambda matrices. A system of linear algebraic \
equations with m-dimensional lambda matrices is considered. \
The proposed method of searching for the solution of this system \
lies in reducing it to a numerical system of a special kind."""

my_doc = my_doc.replace('\n', '')

# pre-process document
my_tokens = clean_text_simple(my_doc,my_stopwords=stpwds,punct=punct)

print('my tokens: ', my_tokens)

g = terms_to_graph(my_tokens, 4) 

#### Task2 #####

visual_style = {}
visual_style["vertex_label"] = g.vs["name"]
visual_style["vertex_color"] = "white"
visual_style["vertex_label_dist"] = 3.5
visual_style["vertex_label_angle"] = 5
visual_style["edge_width"] = g.es['weight']
visual_style["layout"] = g.layout("kk")
visual_style["margin"] = 40
visual_style["vertex_size"] = 15
plot(g, **visual_style, autocurve = True)

#### End of Task 2 #####

# number of edges
print(len(g.es))

# the number of nodes should be equal to the number of unique terms
len(g.vs) == len(set(my_tokens))

edge_weights = []
for edge in g.es:
    source = g.vs[edge.source]['name']
    target = g.vs[edge.target]['name']
    weight = edge['weight']
    edge_weights.append([source, target, weight])

print(edge_weights)

for w in range(2, len(my_tokens)):
    g = terms_to_graph(my_tokens, w)
    ### fill the gap (print density of g) ###
    # Density using the density() function
    density = g.density();
    print('w = ', w, '\nDensity of the graph: ', density)
    # Density using the formula
    # print(len(g.es) / (len(g.vs) * (len(g.vs) - 1)))
    

# decompose g
g = terms_to_graph(my_tokens, 4)

# weighted
core_numbers = core_dec(g, True)
print(core_numbers)    
    
# unweighted
core_numbers = core_dec(g, False)
print(core_numbers)


### fill the gap (compare 'core_numbers' with the output of the .coreness() igraph method) ###
print('Core numbers igraph method: ', g.coreness()) # should be equal to the cores resulted from the unweighted k-core decomposition
  

# retain main core as keywords
max_c_n = max(core_numbers.values())
keywords = [kwd for kwd, c_n in core_numbers.items() if c_n == max_c_n]
print(keywords)
