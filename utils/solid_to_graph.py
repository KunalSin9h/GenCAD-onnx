import argparse
import pathlib

import dgl
import numpy as np
import torch
from occwl.graph import face_adjacency
from occwl.io import load_step
from occwl.uvgrid import ugrid, uvgrid, uvgrid_new
from tqdm import tqdm
from multiprocessing.pool import Pool
from itertools import repeat
import signal


from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRep import BRep_Tool_Surface
from OCC.Core.TopoDS import topods_Face


def build_graph(solid, curv_num_u_samples, surf_num_u_samples, surf_num_v_samples):
    # Build face adjacency graph with B-rep entities as node and edge features
    graph = face_adjacency(solid)

    # Compute the UV-grids for faces
    graph_face_feat, uv_values = [], []
    for face_idx in graph.nodes:
        # Get the B-rep face
        face = graph.nodes[face_idx]["face"]
        # Compute UV-grids
        points, uvs = uvgrid(
            face, method="point", num_u=surf_num_u_samples, num_v=surf_num_v_samples, uvs=True 
        )
        normals = uvgrid(
            face, method="normal", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        visibility_status = uvgrid(
            face, method="visibility_status", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        mask = np.logical_or(visibility_status == 0, visibility_status == 2)  # 0: Inside, 1: Outside, 2: On boundary
        # Concatenate channel-wise to form face feature tensor
        face_feat = np.concatenate((points, normals, mask), axis=-1)
        graph_face_feat.append(face_feat)
        uv_values.append(uvs)
    graph_face_feat = np.asarray(graph_face_feat)
    uv_values = np.asarray(uv_values)
    
    # Compute the U-grids for edges
    graph_edge_feat = []
    for edge_idx in graph.edges:
        # Get the B-rep edge
        edge = graph.edges[edge_idx]["edge"]
        # Ignore dgenerate edges, e.g. at apex of cone
        if not edge.has_curve():
            continue
        # Compute U-grids
        points = ugrid(edge, method="point", num_u=curv_num_u_samples)
        tangents = ugrid(edge, method="tangent", num_u=curv_num_u_samples)
        # Concatenate channel-wise to form edge feature tensor
        edge_feat = np.concatenate((points, tangents), axis=-1)
        graph_edge_feat.append(edge_feat)
    graph_edge_feat = np.asarray(graph_edge_feat)

    # Convert face-adj graph to DGL format
    edges = list(graph.edges)
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    dgl_graph = dgl.graph((src, dst), num_nodes=len(graph.nodes))
    dgl_graph.ndata["x"] = torch.from_numpy(graph_face_feat)
    dgl_graph.ndata["uv"] = torch.from_numpy(uv_values)
    dgl_graph.edata["x"] = torch.from_numpy(graph_edge_feat)
    return dgl_graph


def update_fillet_edges(fillet_edges, node, edges2, graph1, graph2):
    n_neighbors = list(graph2.neighbors(node))

    for n in n_neighbors:
        if n in list(graph1.nodes()):
            m = list(graph1.edges(n))
            for k in m:
                if k not in edges2 and k not in fillet_edges:
                    fillet_edges.append(k)
                    fillet_edges.append(tuple(reversed(k)))

    return fillet_edges



def get_fillet_edges(graph1, graph2):

    # get nodes and edges
    nodes1, nodes2 = list(graph1.nodes()), list(graph2.nodes())
    edges1, edges2 = list(graph1.edges()), list(graph2.edges())

    new_nodes = list(set(nodes1) ^ set(nodes2))
    new_edges = list(set(edges1) ^ set(edges2))

    fillet_edges = []
    for node in new_nodes:
        fillet_edges = update_fillet_edges(fillet_edges, node, edges2, graph1, graph2)

    return fillet_edges



def build_graph_with_labels(solid_orig, solid_fillet, curv_num_u_samples, surf_num_u_samples, surf_num_v_samples):

    # Build face adjacency graph with B-rep entities as node and edge features
    graph = face_adjacency(solid_orig)
   
    # Compute the UV-grids for faces
    graph_face_feat, uv_values, face_areas = [], [], []
    for face_idx in graph.nodes:
        # Get the B-rep face
        face = graph.nodes[face_idx]["face"]
        # Compute UV-grids
        points, uvs = uvgrid(
            face, method="point", num_u=surf_num_u_samples, num_v=surf_num_v_samples, uvs=True 
        )
        normals = uvgrid(
            face, method="normal", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        visibility_status = uvgrid(
            face, method="visibility_status", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        mask = np.logical_or(visibility_status == 0, visibility_status == 2)  # 0: Inside, 1: Outside, 2: On boundary
        
        # Concatenate channel-wise to form face feature tensor
        face_feat = np.concatenate((points, normals, mask), axis=-1)
        graph_face_feat.append(face_feat)
        uv_values.append(uvs)

        # calculate face area
        props = GProp_GProps()
        brepgprop_SurfaceProperties(face.topods_shape(), props)
        area = props.Mass()  # face area
        com = props.CentreOfMass().Coord()   # face center of mass



        face_areas.append(area)

    face_areas = np.asarray(face_areas)

    graph_face_feat = np.asarray(graph_face_feat)
    uv_values = np.asarray(uv_values)



    # Compute the U-grids for edges
    graph_edge_feat = []
    for edge_idx in graph.edges:
        # Get the B-rep edge
        edge = graph.edges[edge_idx]["edge"]
        # Ignore dgenerate edges, e.g. at apex of cone
        if not edge.has_curve():
            continue
        # Compute U-grids
        points = ugrid(edge, method="point", num_u=curv_num_u_samples)
        tangents = ugrid(edge, method="tangent", num_u=curv_num_u_samples)
        # Concatenate channel-wise to form edge feature tensor
        edge_feat = np.concatenate((points, tangents), axis=-1)
        graph_edge_feat.append(edge_feat)
    graph_edge_feat = np.asarray(graph_edge_feat)

    # Convert face-adj graph to DGL format
    edges = list(graph.edges)


    # find unique edges
    unique_edges = set(frozenset(edge) for edge in edges)
    unique_edges = [tuple(sorted(edge)) for edge in unique_edges]
    unique_edges = sorted(unique_edges)

    graph_fillet = face_adjacency(solid_fillet)    
    fillet_edges = get_fillet_edges(graph, graph_fillet)

    unique_fillet_edges = set(frozenset(edge) for edge in fillet_edges)
    unique_fillet_edges = [tuple(sorted(edge)) for edge in unique_fillet_edges]
    unique_fillet_edges = sorted(unique_fillet_edges)

    fillet_indices = []
    for f in unique_fillet_edges: 
        fillet_indices.append(unique_edges.index(f))


    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    dgl_graph = dgl.graph((src, dst), num_nodes=len(graph.nodes))
    dgl_graph.ndata["x"] = torch.from_numpy(graph_face_feat)
    dgl_graph.ndata["uv"] = torch.from_numpy(uv_values)
    dgl_graph.edata["x"] = torch.from_numpy(graph_edge_feat)
    dgl_graph.ndata["area"] = torch.from_numpy(face_areas)
    

    val = np.zeros(len(unique_edges))
    val[fillet_indices] = 1
    val = np.tile(val, 2)

    # new_val = []
    # for k in val: 
    #     if k == 0: 
    #         new_val.append([1, 0])
    #     else:
    #         new_val.append([0, 1])
    # new_val = np.array(new_val)
    # dgl_graph.edata["y"] = torch.from_numpy(new_val)

    dgl_graph.edata["y"] = torch.from_numpy(val)

    return dgl_graph
