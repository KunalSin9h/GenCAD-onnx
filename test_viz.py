# put your code here
from visualizations.viz import VizMesh, VizPointCloud, VizSTL, VizVoxel, VizImage


def viz_mesh(mesh_path):
    viz = VizMesh()
    return 


def viz_pc(mesh_path):
    viz = VizPointCloud()
    return 


def viz_stl(mesh_path):
    viz = VizSTL()
    return 


def viz_voxel(mesh_path):
    viz = VizVoxel()
    return 


def viz_image(mesh_path):
    viz = VizImage()
    return 


if __name__ == "__main__":
    
    mesh_path = 'path/to/mesh'
    pc_path = 'path/to/pc'
    stl_path = 'path/to/stl'
    voxel_path = 'path/to/voxel'
    
    viz_mesh(mesh_path)
    viz_pc(pc_path)
    viz_stl(stl_path)
    viz_voxel(voxel_path)
    viz_image(voxel_path)
