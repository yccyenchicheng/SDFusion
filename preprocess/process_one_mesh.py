
from random import choices
import h5py
import os
import argparse
import numpy as np
import trimesh
from scipy.interpolate import RegularGridInterpolator
import time
import pdb
# import pymesh


def get_sdf(sdf_file, sdf_res):
    intsize = 4
    floatsize = 8

    sdf = {
        "param": [],
        "value": []
    }
    with open(sdf_file, "rb") as f:
        try:
            bytes = f.read()
            ress = np.fromstring(bytes[:intsize * 3], dtype=np.int32)
            if -1 * ress[0] != sdf_res or ress[1] != sdf_res or ress[2] != sdf_res:
                raise Exception(sdf_file, "sdf_res not consistent with ", str(sdf_res))
            positions = np.fromstring(bytes[intsize * 3:intsize * 3 + floatsize * 6], dtype=np.float64)
            # bottom left corner, x,y,z and top right corner, x, y, z
            sdf["param"] = [positions[0], positions[1], positions[2],
                            positions[3], positions[4], positions[5]]
            sdf["param"] = np.float32(sdf["param"])
            sdf["value"] = np.fromstring(bytes[intsize * 3 + floatsize * 6:], dtype=np.float32)
            sdf["value"] = np.reshape(sdf["value"], (sdf_res + 1, sdf_res + 1, sdf_res + 1)) # somehow the cube is sdf_res+1 rather than sdf_res... need to investigate why
        finally:
            f.close()
    return sdf

def get_offset_ball(num, bandwidth):
    u = np.random.normal(0, 1, size=(num,1))
    v = np.random.normal(0, 1, size=(num,1))
    w = np.random.normal(0, 1, size=(num,1))
    r = np.random.uniform(0, 1, size=(num,1)) ** (1. / 3) * bandwidth
    norm = np.linalg.norm(np.concatenate([u, v, w], axis=1),axis=1, keepdims=1)
    # print("u.shape",u.shape)
    # print("norm.shape",norm.shape)
    # print("r.shape",r.shape)
    (x, y, z) = r * (u, v, w) / norm
    return np.concatenate([x,y,z],axis=1)

def get_offset_cube(num, bandwidth):
    u = np.random.normal(0, 1, size=(num,1))
    v = np.random.normal(0, 1, size=(num,1))
    w = np.random.normal(0, 1, size=(num,1))
    r = np.random.uniform(0, 1, size=(num,1)) ** (1. / 3) * bandwidth
    norm = np.linalg.norm(np.concatenate([u, v, w], axis=1),axis=1, keepdims=1)
    # print("u.shape",u.shape)
    # print("norm.shape",norm.shape)
    # print("r.shape",r.shape)
    (x, y, z) = r * (u, v, w) / norm
    return np.concatenate([x,y,z],axis=1)

def sample_sdf(num_sample, bandwidth, iso_val, sdf_dict, sdf_res, reduce):
    start = time.time()
    params = sdf_dict["param"]
    sdf_values = sdf_dict["value"].flatten()
    # print("np.min(sdf_values), np.mean(sdf_values), np.max(sdf_values)",
    #       np.min(sdf_values), np.mean(sdf_values), np.max(sdf_values))

    # n_sample = sdf_res // reduce + 1
    # 
    n_sample = sdf_res // reduce # want 64 * 64 * 64

    x = np.linspace(params[0], params[3], num=n_sample).astype(np.float32)
    y = np.linspace(params[1], params[4], num=n_sample).astype(np.float32)
    z = np.linspace(params[2], params[5], num=n_sample).astype(np.float32)
    z_vals, y_vals, x_vals = np.meshgrid(z, y, x, indexing='ij')
    # print("x_vals", x_vals[0, 0, sdf_res // reduce - 1])
    # x_original = np.linspace(params[0], params[3], num=sdf_res + 1).astype(np.float32)
    # y_original = np.linspace(params[1], params[4], num=sdf_res + 1).astype(np.float32)
    # z_original = np.linspace(params[2], params[5], num=sdf_res + 1).astype(np.float32)
    x_original = np.linspace(params[0], params[3], num=sdf_res+1).astype(np.float32)
    y_original = np.linspace(params[1], params[4], num=sdf_res+1).astype(np.float32)
    z_original = np.linspace(params[2], params[5], num=sdf_res+1).astype(np.float32)
    x_ind = np.arange(n_sample).astype(np.int32)
    y_ind = np.arange(n_sample).astype(np.int32)
    z_ind = np.arange(n_sample).astype(np.int32)
    zv, yv, xv = np.meshgrid(z_ind, y_ind, x_ind, indexing='ij')
    choosen_ind = xv * reduce + yv * (sdf_res+1) * reduce + zv * (sdf_res+1)**2 * reduce
    choosen_ind = np.asarray(choosen_ind, dtype=np.int32).reshape(-1)
    vals = sdf_values[choosen_ind]
    x_vals = x[xv.reshape(-1)]
    y_vals = y[yv.reshape(-1)]
    z_vals = z[zv.reshape(-1)]

    # pdb.set_trace()
    # sdf_pt_val = np.stack((x_vals, y_vals, z_vals, vals), axis = -1)
    sdf_pt_val = np.expand_dims(vals, axis= -1 )
    # print("np.min(vals), np.mean(vals), np.max(vals)", np.min(vals), np.mean(vals), np.max(vals))
    # print("sdf_pt_val.shape", sdf_pt_val.shape)
    print("[*] sample_sdf: {} s".format(time.time()-start))
    return sdf_pt_val, check_insideout(sdf_values, sdf_res, x_original,y_original,z_original)

def check_insideout(sdf_val, sdf_res, x, y, z):
    x_ind = np.argmin(np.absolute(x))
    y_ind = np.argmin(np.absolute(y))
    z_ind = np.argmin(np.absolute(z))
    all_val = sdf_val.flatten()
    # num_val = all_val[x_ind+y_ind*(sdf_res+1)+z_ind*(sdf_res+1)**2]
    num_val = all_val[x_ind+y_ind*(sdf_res)+z_ind*(sdf_res)**2]
    return num_val > 0.0

def create_h5_sdf_pt(h5_file, sdf_file, norm_obj_file,
         centroid, m, sdf_res, num_sample, bandwidth, iso_val, max_verts, normalize, reduce=8):
    sdf_dict = get_sdf(sdf_file, sdf_res)
    ori_verts = np.asarray([0.0,0.0,0.0], dtype=np.float32).reshape((1,3))
    # Nx3(x,y,z)
    # print("ori_verts", ori_verts.shape)
    samplesdf, is_insideout = sample_sdf(num_sample, bandwidth, iso_val, sdf_dict, sdf_res, reduce)  # (N*8)x4 (x,y,z)
    # print("samplesdf", samplesdf.shape)
    print("[*] start writing: ", h5_file)
    norm_params = np.concatenate((centroid, np.asarray([m]).astype(np.float32)))
    f1 = h5py.File(h5_file, 'w')
    f1.create_dataset('pc_sdf_original', data=ori_verts.astype(np.float32), compression='gzip', compression_opts=4)
    f1.create_dataset('pc_sdf_sample', data=samplesdf.astype(np.float32), compression='gzip', compression_opts=4)
    f1.create_dataset('norm_params', data=norm_params, compression='gzip', compression_opts=4)
    f1.create_dataset('sdf_params', data=sdf_dict["param"], compression='gzip', compression_opts=4)
    f1.close()
    print("[*] end writing: ", h5_file)
    command_str = "rm -rf " + norm_obj_file
    print("[*] command:", command_str)
    os.system(command_str)
    command_str = "rm -rf " + sdf_file
    print("[*] command:", command_str)
    os.system(command_str)

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = trimesh.Trimesh(vertices=scene_or_mesh.vertices, faces=scene_or_mesh.faces)
    return mesh

# from DISN create_point_sdf_grid
def get_normalize_mesh(model_file, norm_mesh_sub_dir):
    total = 16384
    print("[*] loading model with trimesh...", model_file)
    mesh_list = trimesh.load_mesh(model_file, process=False)
    print("[*] done!", model_file)

    mesh = as_mesh(mesh_list) # from s2s
    if not isinstance(mesh, list):
        mesh_list = [mesh]

    area_sum = 0
    area_lst = []
    for idx, mesh in enumerate(mesh_list):
        area = np.sum(mesh.area_faces)
        area_lst.append(area)
        area_sum+=area
    area_lst = np.asarray(area_lst)
    amount_lst = (area_lst * total / area_sum).astype(np.int32)
    points_all=np.zeros((0,3), dtype=np.float32)
    for i in range(amount_lst.shape[0]):
        mesh = mesh_list[i]
        # print("start sample surface of ", mesh.faces.shape[0])
        points, index = trimesh.sample.sample_surface(mesh, amount_lst[i])
        # print("end sample surface")
        points_all = np.concatenate([points_all,points], axis=0)
    centroid = np.mean(points_all, axis=0)
    points_all = points_all - centroid
    m = np.max(np.sqrt(np.sum(points_all ** 2, axis=1)))
    # print("centroid, m", centroid, m)
    obj_file = os.path.join(norm_mesh_sub_dir, "pc_norm.obj")
    # ori_mesh = pymesh.load_mesh(model_file)
    # pymesh.save_mesh_raw(obj_file, (ori_mesh.vertices - centroid) / float(m), ori_mesh.faces)
    
    ori_mesh_list = trimesh.load_mesh(model_file, process=False)
    ori_mesh = as_mesh(ori_mesh_list)
    ori_mesh.vertices = (ori_mesh.vertices - centroid) / float(m)
    ori_mesh.export(obj_file)

    print("[*] export_mesh: ", obj_file)
    return obj_file, centroid, m


def create_one_sdf(sdfcommand, sdf_res, expand_rate, sdf_file, obj_file, indx, g=0.0):

    # command_str = ". " + sdfcommand + " " + obj_file + " " + str(sdf_res) + " " + str(sdf_res) + \
    command_str = sdfcommand + " " + obj_file + " " + str(sdf_res) + " " + str(sdf_res) + \
       " " + str(sdf_res) + " -s " + " -e " + str(expand_rate) + " -o " + str(indx) + ".dist -m 1"
    command_str += ' -c'
    if g > 0.0:
        command_str += " -g " + str(g)

    print("[*] command:", command_str)
    os.system(command_str)
    command_str2 = "mv " + str(indx)+".dist " + sdf_file
    print("[*] command:", command_str2)
    os.system(command_str2)

def create_sdf_obj(sdfcommand, marching_cube_command, norm_mesh_dir, sdf_dir, obj,
       sdf_res, iso_val, expand_rate, indx, ish5, normalize, num_sample, bandwidth, max_verts, g, reduce, h5_file=None):

    model_id = os.path.basename(obj).replace('.obj', '')

    norm_mesh_sub_dir = os.path.join(norm_mesh_dir, model_id)
    sdf_sub_dir = os.path.join(sdf_dir, model_id)

    if not os.path.exists(norm_mesh_sub_dir): os.makedirs(norm_mesh_sub_dir)
    if not os.path.exists(sdf_sub_dir): os.makedirs(sdf_sub_dir)
    
    sdf_file = os.path.join(sdf_sub_dir, "isosurf.sdf")

    if h5_file is None:
        h5_file = obj.replace('.obj', '_sdf.h5')
    # h5_file = os.path.join(sdf_sub_dir, "ori_sample_grid.h5")

        # model_file = os.path.join(cat_mesh_dir, obj, "models", "model_normalized.obj")
        # model_file = os.path.join(cat_mesh_dir, obj, "model.obj")
    model_file = obj
    print("[*] creating", sdf_file)
    if normalize:
        norm_obj_file, centroid, m = get_normalize_mesh(model_file, norm_mesh_sub_dir)

    create_one_sdf(sdfcommand, sdf_res, expand_rate, sdf_file, norm_obj_file, indx, g=g)
    # save to h5
    create_h5_sdf_pt(h5_file, sdf_file, norm_obj_file,
            centroid, m, sdf_res, num_sample, bandwidth, iso_val, max_verts, normalize, reduce=reduce)

def process_one_obj(sdfcommand,
                    marching_cube_command,
                    LIB_command,
                    num_sample,
                    bandwidth,
                    sdf_res,
                    expand_rate,
                    obj_file,
                    iso_val,
                    max_verts,
                    ish5=True, normalize=True, g=0.00, reduce=4):
    '''
    Usage: SDFGen <filename> <dx> <padding>
    Where:
        res is number of grids on xyz dimension
        w is narrowband width
        expand_rate is sdf range of max x,y,z
    '''
    #cats_init = cats
    #cats = cats_init
    #cats['airplane'] = cats_init['airplane']
    #print("command:", LIB_command)
    os.system(LIB_command)
    start=0

    dataroot = '../../data'

    tmp_dir = f'tmp/for_sdf'
    model_dir = f'{tmp_dir}/model'
    norm_mesh_dir = f'{tmp_dir}/norm_mesh'
    sdf_dir = f'{tmp_dir}/sdf'

    if not os.path.exists(norm_mesh_dir): os.makedirs(norm_mesh_dir)
    if not os.path.exists(sdf_dir): os.makedirs(sdf_dir)

    # list_obj = os.listdir(model_dir)
    list_obj = [obj_file]

    repeat = len(list_obj)
    sdfcommand_lst=[sdfcommand for i in range(repeat)]
    marching_cube_command_lst=[marching_cube_command for i in range(repeat)]
    norm_mesh_dir_lst=[norm_mesh_dir for i in range(repeat)] # by yc
    sdf_dir_lst=[sdf_dir for i in range(repeat)]
    res_lst=[sdf_res for i in range(repeat)]
    iso_val_lst=[iso_val for i in range(repeat)]
    expand_rate_lst=[expand_rate for i in range(repeat)]
    indx_lst = [i for i in range(start, start+repeat)]
    ish5_lst=[ish5 for i in range(repeat)]
    normalize_lst=[normalize for i in range(repeat)]
    num_sample_lst=[num_sample for i in range(repeat)]
    bandwidth_lst=[bandwidth for i in range(repeat)]
    max_verts_lst=[max_verts for i in range(repeat)]
    g_lst=[g for i in range(repeat)]
    reduce_lst=[reduce for i in range(repeat)]

    # parallel
    # with Parallel(n_jobs=5) as parallel:
    #     parallel(delayed(create_sdf_obj)
    #     (sdfcommand, marching_cube_command, norm_mesh_dir, sdf_dir, obj, res, iso_val, expand_rate, indx, ish5, norm, num_sample, bandwidth, max_verts, g, reduce)
    #     for sdfcommand, marching_cube_command, norm_mesh_dir, sdf_dir, obj, res, iso_val, 
    #         expand_rate, indx, ish5, norm, num_sample, bandwidth, max_verts, g, reduce
    #         in zip(sdfcommand_lst,
    #                marching_cube_command_lst,
    #                norm_mesh_dir_lst,
    #                sdf_dir_lst,
    #                list_obj,
    #                res_lst,
    #                iso_val_lst,
    #                expand_rate_lst,
    #                indx_lst, ish5_lst, normalize_lst, num_sample_lst,
    #                bandwidth_lst, max_verts_lst, g_lst, reduce_lst))

    # no parallel
    for (sdfcommand, marching_cube_command, norm_mesh_dir, sdf_dir, obj, sdf_res, iso_val, 
        expand_rate, indx, ish5, norm, num_sample, bandwidth, max_verts, g, reduce) in \
        zip(sdfcommand_lst, marching_cube_command_lst, norm_mesh_dir_lst, sdf_dir_lst, list_obj,
            res_lst, iso_val_lst, expand_rate_lst, indx_lst, ish5_lst, normalize_lst, num_sample_lst,
            bandwidth_lst, max_verts_lst, g_lst, reduce_lst):
            create_sdf_obj(
                sdfcommand, marching_cube_command, norm_mesh_dir, sdf_dir, obj, sdf_res, 
                iso_val, expand_rate, indx, ish5, norm, num_sample, bandwidth, max_verts, g, reduce)

    print("[*] finished!")



def process_obj(obj_file, isosurface_dir, sdf_cmd, mcube_cmd, reduce=1, h5_file=None):

    # sdf_cmd = './preprocess/isosurface/computeDistanceField'
    # sdfcommand = './isosurface/computeDistanceField'
    # mcube_cmd = 'preprocess/isosurface/computeMarchingCubes'
    # lib_cmd = 'preprocess/isosurface/LIB_PATH'

    # output sdf_file, save as .h5

    num_sample = 65 ** 3
    bandwidth = 0.1 # snet
    sdf_res = 256
    expand_rate = 1.3
    iso_val = 0.003 # snet
    max_verts = 16384
    g=0.0 # snet

    indx = 0
    ish5 = True
    norm = True
    # reduce = 4
    # reduce = 1

    if h5_file is None:
        h5_file = obj_file.replace(".obj", f"_sdf_{sdf_res//reduce}.h5")

    # os.environ['LD_LIBRARY_PATH'] = f'$LD_LIBRARY_PATH:{isosurface_dir}:./preprocess/isosurface/:./preprocess/isosurface/tbb/tbb2018_20180822oss/lib/intel64/gcc4.7:/opt/intel/lib/intel64:/opt/intel/mkl/lib/intel64:/usr/local/lib64:/usr/local/lib:/usr/local/cuda/lib64'
    # os.environ['LD_LIBRARY_PATH'] = f'$LD_LIBRARY_PATH:./isosurface/:./preprocess/isosurface/:./preprocess/isosurface/tbb/tbb2018_20180822oss/lib/intel64/gcc4.7:/opt/intel/lib/intel64:/opt/intel/mkl/lib/intel64:/usr/local/lib64:/usr/local/lib:/usr/local/cuda/lib64'
    os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:./isosurface/:./isosurface/tbb/tbb2018_20180822oss/lib/intel64/gcc4.7:/opt/intel/lib/intel64:/opt/intel/mkl/lib/intel64:/usr/local/lib64:/usr/local/lib:/usr/local/cuda/lib64'

    tmp_dir = f'tmp/for_sdf'
    model_dir = f'{tmp_dir}/model'
    norm_mesh_dir = f'{tmp_dir}/norm_mesh'
    sdf_dir = f'{tmp_dir}/sdf'

    if not os.path.exists(norm_mesh_dir): os.makedirs(norm_mesh_dir)
    if not os.path.exists(sdf_dir): os.makedirs(sdf_dir)

    create_sdf_obj(sdf_cmd, mcube_cmd, norm_mesh_dir, sdf_dir, obj_file, sdf_res, 
                iso_val, expand_rate, indx, ish5, norm, num_sample, bandwidth, max_verts, g, reduce, h5_file)

    print(f'[*] successfully extract sdf and save to: {h5_file}')
    return h5_file

if __name__ == "__main__":

    # nohup python -u create_point_sdf_fullgrid.py &> createfull.log &
    # lst_dir, cats, all_cats, raw_dirs = create_file_lst_abc.get_all_info()
    
    # dset = FLAGS.dset
    # cat = FLAGS.category

    # lst_dir, cats, all_cats, raw_dirs = create_sdf_file_lst.get_all_info(dset)

    # # if dset != 'abc':
    #     if cat == 'all':
    #         FLAGS.cats = all_cats
    #     else:
    #         FLAGS.cats = [cat]


    # obj_file = FLAGS.obj_file

    # print("# <<< REMEMBER to run: ")
    # print("#     source ./isosurface/LIB_PATH")
    # print("#     !!!! <<<<")

    print("REMEMBER to replace the following line: ")
    print("with the path that works in your environment ")
    os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:./isosurface/:./isosurface/tbb/tbb2018_20180822oss/lib/intel64/gcc4.7:/opt/intel/lib/intel64:/opt/intel/mkl/lib/intel64:/usr/local/lib64:/usr/local/lib:/usr/local/cuda/lib64'


    # obj_file = '../demo_data/chair_model.obj'
    sdf_cmd = './isosurface/computeDistanceField'
    mcube_cmd = './isosurface/computeMarchingCubes'
    lib_cmd = './isosurface/LIB_PATH'
    isosurface_dir = './isosurface/'
    # os.system(f'source {lib_cmd}')

    # obj_file = '../demo_data/1006be65e7bc937e9141f9b58470d646.obj'
    obj_file = '../demo_data/COMMERCIALcastle_mesh0365.obj'

    process_obj(obj_file, isosurface_dir, sdf_cmd, mcube_cmd)
    print('mesh to sdf: done!')


    

    # num_sample = 64 ** 3
    # bandwidth = 0.1 # snet
    # sdf_res = 256
    # expand_rate = 1.3
    # iso_val = 0.003 # snet
    # max_verts = 16384
    # g=0.0 # snet
    # # g=0.1 # bunny

    # #  full set
    # # create_sdf(sdfcommand,
    # #            mcube_cmd,
    # #            "source %s" % lib_cmd, 274625, 0.1,
    # #            256, 1.3, all_cats, cats, raw_dirs,
    # #            lst_dir, 0.003, 16384, ish5=True, normalize=True, g=0.00, reduce=4)
    # # create_sdf_obj(sdfcommand,
    # #            mcube_cmd,
    # #            "source %s" % lib_cmd, num_sample, bandwidth,
    # #            res, expand_rate, raw_dirs, iso_val, max_verts, ish5=True, normalize=True, g=0.00, reduce=4)

    # process_one_obj(sdfcommand, mcube_cmd, "source %s" % lib_cmd,
    #             num_sample, bandwidth, sdf_res, expand_rate, obj_file, iso_val,
    #             max_verts, ish5=True, normalize=True, g=g, reduce=4)
    
