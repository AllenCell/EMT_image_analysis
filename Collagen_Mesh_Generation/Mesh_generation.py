from aicsimageio import AICSImage
import numpy as np
import pyvista as pv
from skimage.transform import resize
from skimage.exposure import rescale_intensity
import pymeshfix as mf
import trimesh
from pathlib import Path
from argparse import ArgumentParser
import open3d as o3d
import pyacvd

######---------Main code---------######

def mesh_generation(
        segmentation_fn: str,
        output_directory:str,
        start_timepoint: int=0,
        end_timepoint: int=90,
    ):
    '''
        Generate collagen membrane mesh for a colony timelapse segmentation.
        Saves the meshes as a pyvista MultiBlock object in a .vtm file.
        
        Parameters:
            segmentation_fn: str
                Filepath to the timelapse segmentation.
            start_timepoint: int
                The first timepoint to process.
            end_timepoint: int
                The last timepoint to process.
    '''
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    segmentations = AICSImage(segmentation_fn)
    
    num_timepoints = segmentations.shape[0]
    if end_timepoint < 0 or end_timepoint >= num_timepoints:
        end_timepoint = num_timepoints
    
    meshes = {}
    for timepoint in range(start_timepoint, end_timepoint):
        seg_fn = segmentations.get_image_data(timepoint)
        out_fn = f"mesh_{timepoint}.obj"
        
        mesh = process_seg(seg_fn, out_fn)
        meshes[timepoint] = mesh
    
    mesh_block = pv.MultiBlock(meshes)

######---------Per-timepoint code---------######

def process_seg(seg_fn):
    
    seg = AICSImage(seg_fn).data.squeeze()
    
    seg = resize(
        seg, 
        (int(seg.shape[0] * 2.88/0.271), seg.shape[1], seg.shape[2]), 
        order=0, 
        preserve_range=False
    )
    
    mesh = seg_to_mesh(seg)
    
    return pv.wrap(mesh)
    
    
######---------Helper functions---------######


def apply_local_outlier_factor_removal_to_pc(pc, n_neighbors=20):
    cl, _ = pc.remove_statistical_outlier(nb_neighbors=n_neighbors, std_ratio=2.0)
    cl, _ = cl.remove_radius_outlier(nb_points=n_neighbors, radius=40)
    return cl

def ensure_outward_normals(mesh):
    centroid = list(mesh.centroid)
    centroid[-1] = mesh.bounds[1, -1]
    
    mesh.fix_normals()
    
    rayCaster = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    hitFace = rayCaster.intersects_first([centroid], [[0,0,-1]])
    assert hitFace is not None, "Could not find a face to flip"
    
    faceNorm = mesh.face_normals[hitFace]
    if faceNorm[0,-1] > 0:
        mesh.invert()
        
    return mesh

def sample_segmentation(seg, n_samples=30000):
    seg = seg.astype(np.float32) / seg.max()

    sample_probs = np.clip(seg, 0.3, 1)
    sample_probs = rescale_intensity(sample_probs, out_range=(0,1))
    sample_probs = sample_probs / np.sum(sample_probs)

    samples = np.random.choice(
        np.arange(sample_probs.size), 
        size=n_samples, 
        p=sample_probs.flatten()
    )
    z,y,x = np.unravel_index(samples, sample_probs.shape)
    
    return np.stack([x,y,z], axis=1)

def init_mesh(pCloud):
    bounds = pCloud.bounds
    radius = min([
        (bounds[1] - bounds[0])/2,
        (bounds[3] - bounds[2])/2,
    ])
    height = bounds[5] - bounds[4]
    height_scale = height / radius
    scale = 0.8

    samples_points = pCloud.points
    center = (
        np.mean(samples_points[:,0]),
        np.mean(samples_points[:,1]), 
        bounds[5]
    )
    transform = trimesh.transformations.scale_and_translate(
        scale= [
            scale,
            scale,
            scale * height_scale    
        ], 
        translate=center
    )

    sphereMesh = trimesh.creation.icosphere(
        subdivisions=3,
        radius=radius)
    sphereMesh.apply_transform(transform)
    
    faces = sphereMesh.vertex_faces[sphereMesh.vertices[:,2] < np.amax(samples_points[:,2])]
    bot_faces = np.unique(faces.flatten()).tolist()
    sphereMesh = sphereMesh.submesh([bot_faces], only_watertight=False, append=True)
    
    return sphereMesh

def seg_to_mesh(seg):
    seg_sample = sample_segmentation(seg)
    
    center = np.mean(seg_sample, axis=0)
    bounds = np.array(
        [
            seg_sample[:,0].max() - seg_sample[:,0].min(),
            seg_sample[:,1].max() - seg_sample[:,1].min(),
            seg_sample[:,2].max() - seg_sample[:,2].min()
        ]
    )
    scale_points = [
        750 / bounds[0],
        750 / bounds[1],
        1
    ]
    seg_scaled = np.array(
        [
            (seg_sample[:,0] - center[0]) * scale_points[0] + center[0],
            (seg_sample[:,1] - center[1]) * scale_points[1] + center[1],
            (seg_sample[:,2] - center[2]) * scale_points[2] + center[2]
        ]
    ).T
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(seg_scaled)
    downpcd = pcd.voxel_down_sample(voxel_size=10)
    downpcd = apply_local_outlier_factor_removal_to_pc(downpcd, 10)
    dPoly = pv.PolyData(np.asarray(downpcd.points))
    
    mesh = init_mesh(dPoly)
    
    steps = [
        # [wc, wi, ws, wl, wn],
        [10, 0.001, 0.5, 2000, 0],
        [100, 0.001, 0.75, 2000, 0],
        [200, 0.01, 1.0, 5000, 0],
    ]
    vox_size = [60, 30, 10]
    for vox in vox_size:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(dPoly.points)
        downpcd = pcd.voxel_down_sample(voxel_size=vox)
        dPoly = pv.PolyData(np.asarray(downpcd.points))
        
        target = trimesh.PointCloud(dPoly.points)
        register = trimesh.registration.nricp_sumner(
            source_mesh=mesh, 
            target_geometry=target,
            steps=steps,
            distance_threshold=vox*3,
            use_faces=False,
            use_vertex_normals=False,
            neighbors_count=5,
        )
        mesh.vertices = register
        mesh = mesh.subdivide()
    
    trimesh.repair.fill_holes(mesh)
    mesh = trimesh.smoothing.filter_humphrey(mesh)
    mesh = ensure_outward_normals(mesh)
    
    V, F = mesh.vertices, mesh.faces
    V = np.array(
        [
            (V[:,0] - center[0]) / scale_points[0] + center[0],
            (V[:,1] - center[1]) / scale_points[1] + center[1],
            (V[:,2] - center[2]) / scale_points[2] + center[2],
        ]
    ).T
    mesh = trimesh.Trimesh(V, F)
    
    pSurf = pv.wrap(mesh)
    mfix = mf.MeshFix(pSurf)
    mf_holes = pv.wrap(mfix.extract_holes())
    outline_verts = mf_holes.points
    top = np.percentile(outline_verts[:,2], 99)
    for i in range(outline_verts.shape[0]):
        vert = outline_verts[i]
        new_vert = np.array([vert[0], vert[1], max([vert[2], top])])
        
        v_idx = pSurf.find_closest_point(vert)
        pSurf.points[v_idx] = new_vert
    
    pSurf.subdivide_adaptive(max_edge_len=5, inplace=True)
    clus = pyacvd.Clustering(pSurf)
    clus.subdivide(2)
    clus.cluster(10000)
    mesh = clus.create_mesh()

    mesh = trimesh.Trimesh(
        vertices=mesh.points, 
        faces=mesh.faces.reshape(mesh.n_faces, 4)[:,1:]
    )
    trimesh.repair.fill_holes(mesh)
    
    return mesh
