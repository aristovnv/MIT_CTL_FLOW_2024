import stellargraph as sg
from stellargraph.layer import GraphSAGE
from stellargraph.mapper import GraphSAGENodeGenerator

# Assuming static_node_features, planned_node_features, actual_node_features are node feature matrices
# Assuming static_edge_features, planned_edge_features, actual_edge_features are edge feature matrices
# Assuming target_variables_static, target_variables_planned, target_variables_actual are target variables

# Create StellarGraph objects for each graph
static_graph = sg.StellarGraph(nodes=static_node_features, edges=static_edge_features)
planned_graph = sg.StellarGraph(nodes=planned_node_features, edges=planned_edge_features)
actual_graph = sg.StellarGraph(nodes=actual_node_features, edges=actual_edge_features)

# Create a list of StellarGraph objects for each route
graphs = [static_graph, planned_graph, actual_graph]

# Create a GraphSAGENodeGenerator for the list of graphs
generator = GraphSAGENodeGenerator(graphs)

# Define a GraphSAGE model
model = GraphSAGE(layer_sizes=[64, 32], generator=generator, bias=True, dropout=0.5)

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

# Train the model for each graph
for i in range(len(graphs)):
    train_gen = generator.flow(graphs[i].nodes(), target_variables_list[i], batch_size=32)
    model.fit(train_gen, epochs=10)

    
    
########################################################3
import stellargraph as sg
from stellargraph.layer import GCN, GraphSAGE
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.core.graph import StellarGraph
from tensorflow.keras.layers import GRUCell, GRU

# Assuming node_features_list is a list of node feature matrices for each graph
# Assuming edge_features_list is a list of edge feature matrices for each graph
# Assuming target_variables_list is a list of target variables for each graph

# Create a list of StellarGraph objects for each graph
graphs = [StellarGraph(nodes=node_features, edges=edge_features) for node_features, edge_features in zip(node_features_list, edge_features_list)]

# Create a FullBatchNodeGenerator for the list of graphs
generator = FullBatchNodeGenerator(graphs)

# Define a Graph Convolutional Network (GCN) layer
gcn_layer = GCN(layer_sizes=[64, 32], activations=["relu", "relu"], generator=generator)

# Define a GRUCell layer
gru_cell = GRUCell(64)

# Combine GCN layer and GRUCell layer
graph_convolution = gcn_layer.in_out_tensors()[0]  # Output tensor from GCN layer
gru_output, _ = GRU(64)(graph_convolution, initial_state=gru_cell.get_initial_state(graph_convolution))

# Additional layers and model compilation can be added based on your specific requirements

# Compile the model
# model.compile(...)

# Train the model
# model.fit(...)
