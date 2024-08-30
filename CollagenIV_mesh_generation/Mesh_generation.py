from bioio import BioImage
import numpy as np
import pyvista as pv
from skimage.transform import resize
from skimage.exposure import rescale_intensity
import trimesh
from pathlib import Path
import open3d as o3d
import pyacvd

from argparse import ArgumentParser

######---------Main code---------######

def mesh_generation(
        manifest_path: str,
        movie_id:str,
        output_directory: str,
        start_timepoint: int=0,
        end_timepoint: int=97,
    ):
    '''
        Generate collagen membrane mesh for a colony timelapse segmentation.
        Saves the meshes as a pyvista MultiBlock object in a .vtm file.
        
        Parameters:
            manifest_path: str
                Path to the dataset manifest
            movie_id: str
                Movie Unique ID of the timelapse to process
            output_directory: str
                Directory to save the mesh.
            start_timepoint: int
                The first timepoint to process.
            end_timepoint: int
                The last timepoint to process.
    '''
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True   )
    
    # load the segmentation
    df = pd.read_csv(manifest_path)
    df = df[df['Movie Unique ID'] == movie_id]
    segmentations = BioImage(df['CollagenIV Segmentation Probability URL'].values[0])
    
    # set the timepoints to process
    num_timepoints = int(df['Image Size T'].values[0])
    if end_timepoint < 0 or end_timepoint >= num_timepoints:
        end_timepoint = num_timepoints
    
    # process each timepoint
    meshes = {}
    for timepoint in range(start_timepoint, end_timepoint):
        mesh = process_seg(segmentations.get_image_data(T=timepoint).squeeze())
        meshes[f'{timepoint}'] = mesh
    
    # save the meshes
    mesh_block = pv.MultiBlock(meshes)
    out_fn = Path(df['CollagenIV Segmentation Probability URL'].values[0]).stem.replace("_probability", "_mesh") + ".vtm"
    mesh_block.save(out_dir / out_fn)

######---------Per-timepoint code---------######

def process_seg(
        segmentation: np.ndarray,
    ) -> pv.PolyData:
    '''
        Generate a collagen membrane mesh for a single timepoint segmentation.
        
        Parameters:
            seg_fn: str
                Filepath to the segmentation.
                
        Output:
            mesh: pv.PolyData
                The generated mesh.
    '''
    # resize the segmentation to isometric voxels
    seg = resize(
        seg, 
        (int(seg.shape[0] * 2.88/0.271), seg.shape[1], seg.shape[2]), 
        order=0, 
        preserve_range=False
    )
    
    # sample point cloud from the segmentation
    seg_sample = sample_segmentation(seg)
    
    # scale the point cloud to a standard size
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
    
    # convert point cloud in pyvista object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(seg_scaled)
    downpcd = pcd.voxel_down_sample(voxel_size=10)
    downpcd = apply_local_outlier_factor_removal_to_pc(downpcd, 10)
    dPoly = pv.PolyData(np.asarray(downpcd.points))
    
    # generate the initial mesh
    mesh = init_mesh(dPoly)
    
    # register initial mesh to point cloud
    steps = [
        # [wc, wi, ws, wl, wn],
        [10, 0.001, 0.5, 2000, 0],
        [100, 0.001, 0.75, 2000, 0],
        [200, 0.01, 1.0, 5000, 0],
    ]
    vox_size = [60, 30, 10]
    for vox in vox_size:
        # downsample the point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(dPoly.points)
        downpcd = pcd.voxel_down_sample(voxel_size=vox)
        dPoly = pv.PolyData(np.asarray(downpcd.points))
        
        # register the mesh to the point cloud
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
    
    # mesh post processing
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
    
    # mesh cleanup
    pSurf = pv.wrap(mesh)
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
    
    return pv.wrap(mesh)
    
    
######---------Helper functions---------######


def apply_local_outlier_factor_removal_to_pc(
        pc:o3d.geometry.PointCloud, 
        n_neighbors:int=20
    ) -> o3d.geometry.PointCloud:
    '''
        Apply local outlier factor removal to a point cloud.
        
        Parameters:
            pc: o3d.geometry.PointCloud
                The point cloud to remove outliers from.
            n_neighbors: int
                The number of neighbors to consider for outlier removal.
                
        Output:
            cl: o3d.geometry.PointCloud
                The point cloud with outliers removed.
    '''
    cl, _ = pc.remove_statistical_outlier(nb_neighbors=n_neighbors, std_ratio=2.0)
    cl, _ = cl.remove_radius_outlier(nb_points=n_neighbors, radius=40)
    return cl

def ensure_outward_normals(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    '''
        Ensure that the normals of a mesh are pointing outwards.
        
        Parameters:
            mesh: trimesh.Trimesh
                The mesh to ensure outward normals for.
                
        Output:
            mesh: trimesh.Trimesh
                The mesh with outward normals.
    '''
    centroid = list(mesh.centroid)
    centroid[-1] = mesh.bounds[1, -1]
    
    mesh.fix_normals()
    
    # check if the normals are pointing outwards
    rayCaster = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    hitFace = rayCaster.intersects_first([centroid], [[0,0,-1]])
    assert hitFace is not None, "Could not find a face to flip"
    
    # flip the normals if they are not pointing outwards
    faceNorm = mesh.face_normals[hitFace]
    if faceNorm[0,-1] > 0:
        mesh.invert()
        
    return mesh

def sample_segmentation(
        seg: np.ndarray, 
        n_samples: int=30000,
        probability_threshold: float=0.3
    ) -> np.ndarray:
    '''
        Sample a point cloud from a segmentation.
        
        Parameters:
            seg: np.ndarray
                The segmentation to sample from.
            n_samples: int
                The number of samples to take.
            probability_threshold: float
                The segmentation probability threshold for sampling.
    '''
    seg = seg.astype(np.float32) / seg.max()

    # sample points from the segmentation
    sample_probs = np.clip(seg, probability_threshold, 1)
    sample_probs = rescale_intensity(sample_probs, out_range=(0,1))
    sample_probs = sample_probs / np.sum(sample_probs)

    samples = np.random.choice(
        np.arange(sample_probs.size), 
        size=n_samples, 
        p=sample_probs.flatten()
    )
    z,y,x = np.unravel_index(samples, sample_probs.shape)
    
    return np.stack([x,y,z], axis=1)

def init_mesh(
        pCloud: pv.PolyData
    ) -> trimesh.Trimesh:
    '''
        Generate an initial mesh from a point cloud.
        
        Parameters:
            pCloud: pv.PolyData
                The point cloud to generate the mesh from.
                
        Output:
            mesh: trimesh.Trimesh
                The generated mesh.
    '''
    # calculate bounds of the point cloud
    bounds = pCloud.bounds
    radius = min([
        (bounds[1] - bounds[0])/2,
        (bounds[3] - bounds[2])/2,
    ])
    height = bounds[5] - bounds[4]
    height_scale = height / radius
    scale = 0.8

    # set the transform scale and center the mesh
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

    # generate a sphere mesh
    sphereMesh = trimesh.creation.icosphere(
        subdivisions=3,
        radius=radius)
    sphereMesh.apply_transform(transform)
    
    # remove top half of the sphere
    faces = sphereMesh.vertex_faces[sphereMesh.vertices[:,2] < np.amax(samples_points[:,2])]
    bot_faces = np.unique(faces.flatten()).tolist()
    sphereMesh = sphereMesh.submesh([bot_faces], only_watertight=False, append=True)
    
    return sphereMesh

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--manifest_path",
        type=str,
        required=True,
        help="Filepath to the dataset manifest.",
    )
    parser.add_argument(
        "--movie_id",
        type=str,
        required=True,
        help="Movie Unique ID of timelapse to process."
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        required=True,
        help="Directory to save the mesh.",
    )
    parser.add_argument(
        "--start_timepoint",
        type=int,
        default=0,
        help="The first timepoint to process.",
    )
    parser.add_argument(
        "--end_timepoint",
        type=int,
        default=97,
        help="The last timepoint to process.",
    )
    args = parser.parse_args()
    
    mesh_generation(
        args.manifest_path,
        args.movie_id,
        args.output_directory,
        args.start_timepoint,
        args.end_time
    )