# modify from https://github.com/Xharlie/DISN/blob/master/preprocessing/create_point_sdf_grid.py

from random import choices
import h5py
import os
import argparse
import numpy as np
import glob

from joblib import Parallel, delayed
import trimesh
from scipy.interpolate import RegularGridInterpolator
import time
import json

CUR_PATH = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--dset', type=str, choices=['shapenet', 'abc', 'pix3d', 'building'], default='shapenet', help='which dataset to extract sdf')
parser.add_argument('--thread_num', type=int, default='9', help='how many objs are creating at the same time')
parser.add_argument('--reduce', type=int, default=4, help='define resolution. res=256//reduce')
parser.add_argument('--category', type=str, default="all", help='Which single class to generate on [default: all, can '
                                                                'be chair or plane, etc.]')
FLAGS = parser.parse_args()

def get_sdf_value(sdf_pt, sdf_params_ph, sdf_ph, sdf_res):
    num_point = sdf_pt.shape[0]
    # x = np.linspace(sdf_params_ph[0], sdf_params_ph[3], num=sdf_res+1)
    # y = np.linspace(sdf_params_ph[1], sdf_params_ph[4], num=sdf_res+1)
    # z = np.linspace(sdf_params_ph[2], sdf_params_ph[5], num=sdf_res+1)
    x = np.linspace(sdf_params_ph[0], sdf_params_ph[3], num=sdf_res)
    y = np.linspace(sdf_params_ph[1], sdf_params_ph[4], num=sdf_res)
    z = np.linspace(sdf_params_ph[2], sdf_params_ph[5], num=sdf_res)
    my_interpolating_function = RegularGridInterpolator((z, y, x), sdf_ph)
    sdf_value = my_interpolating_function(sdf_pt)
    print("sdf_value:", sdf_value.shape)
    return np.expand_dims(sdf_value, axis=1)

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
                raise Exception(sdf_file, "res not consistent with ", str(sdf_res))
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
    n_sample = sdf_res // reduce # want 64 * 64 * 64

    x = np.linspace(params[0], params[3], num=n_sample).astype(np.float32)
    y = np.linspace(params[1], params[4], num=n_sample).astype(np.float32)
    z = np.linspace(params[2], params[5], num=n_sample).astype(np.float32)
    z_vals, y_vals, x_vals = np.meshgrid(z, y, x, indexing='ij')
    print("x_vals", x_vals[0, 0, sdf_res // reduce - 1])
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
    print("sdf_pt_val.shape", sdf_pt_val.shape)
    print("sample_sdf: {} s".format(time.time()-start))
    return sdf_pt_val, check_insideout(sdf_values, sdf_res, x_original,y_original,z_original)

def check_insideout(sdf_val, sdf_res, x, y, z):
    # "chair": "03001627",
    # "bench": "02828884",
    # "cabinet": "02933112",
    # "car": "02958343",
    # "airplane": "02691156",
    # "display": "03211117",
    # "lamp": "03636649",
    # "speaker": "03691459",
    # "rifle": "04090263",
    # "sofa": "04256520",
    # "table": "04379243",
    # "phone": "04401088",
    # "watercraft": "04530566"

    # if cat_id in ["02958343", "02691156", "04530566"]:
    x_ind = np.argmin(np.absolute(x))
    y_ind = np.argmin(np.absolute(y))
    z_ind = np.argmin(np.absolute(z))
    all_val = sdf_val.flatten()
    # num_val = all_val[x_ind+y_ind*(sdf_res+1)+z_ind*(sdf_res+1)**2]
    num_val = all_val[x_ind+y_ind*(sdf_res)+z_ind*(sdf_res)**2]
    return num_val > 0.0
    # else:
        # return False

def create_h5_sdf_pt(h5_file, sdf_file, flag_file, norm_obj_file,
         centroid, m, sdf_res, num_sample, bandwidth, iso_val, max_verts, normalize, reduce=8):
    sdf_dict = get_sdf(sdf_file, sdf_res)
    ori_verts = np.asarray([0.0,0.0,0.0], dtype=np.float32).reshape((1,3))
    # Nx3(x,y,z)
    print("ori_verts", ori_verts.shape)
    samplesdf, is_insideout = sample_sdf(num_sample, bandwidth, iso_val, sdf_dict, sdf_res, reduce)  # (N*8)x4 (x,y,z)
    if is_insideout:
        with open(flag_file, "w") as f:
            f.write("mid point sdf val > 0")
        print("insideout !!:", sdf_file)
    else:
        os.remove(flag_file) if os.path.exists(flag_file) else None
    print("samplesdf", samplesdf.shape)
    print("start to write",h5_file)
    norm_params = np.concatenate((centroid, np.asarray([m]).astype(np.float32)))
    f1 = h5py.File(h5_file, 'w')
    f1.create_dataset('pc_sdf_original', data=ori_verts.astype(np.float32), compression='gzip', compression_opts=4)
    f1.create_dataset('pc_sdf_sample', data=samplesdf.astype(np.float32), compression='gzip', compression_opts=4)
    f1.create_dataset('norm_params', data=norm_params, compression='gzip', compression_opts=4)
    f1.create_dataset('sdf_params', data=sdf_dict["param"], compression='gzip', compression_opts=4)
    f1.close()
    print("end writing",h5_file)
    command_str = "rm -rf " + norm_obj_file
    print("command:", command_str)
    os.system(command_str)
    command_str = "rm -rf " + sdf_file
    print("command:", command_str)
    os.system(command_str)

def get_param_from_h5(sdf_h5_file, cat_id, obj):
    h5_f = h5py.File(sdf_h5_file, 'r')
    try:
        if 'norm_params' in h5_f.keys():
            norm_params = h5_f['norm_params'][:]
        else:
            raise Exception(cat_id, obj, "no sdf and sample")
    finally:
        h5_f.close()
    return norm_params[:3], norm_params[3]


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


# def get_normalize_mesh(model_file, norm_sdf_file, cat_id, obj, sdf_sub_dir):

#     print("load mesh from ", model_file)
#     mesh_list = trimesh.load_mesh(model_file, process=False)
#     #if not isinstance(mesh_list, list):
#     #    mesh_list = [mesh_list]
#     #largest_ind = 0
#     #largest_sur = 0
#     #for idx, mesh in enumerate(mesh_list):
#     #    area = np.sum(mesh.area_faces)
#     #    if largest_sur < area:
#     #        largest_ind = idx
#     #        largest_sur = area
#     #mesh = mesh_list[largest_ind]
#     mesh = as_mesh(mesh_list)
#     centroid, m = get_param_from_h5(norm_sdf_file, cat_id, obj)
#     mesh.vertices = (mesh.vertices - centroid) / float(m)
#     obj_file = os.path.join(sdf_sub_dir,"pc_norm.obj")
#     print("exporting", obj_file)
#     trimesh.exchange.export.export_mesh(mesh, obj_file, file_type="obj")
#     print("export_mesh", obj_file)
#     return obj_file, centroid, m

# from DISN create_point_sdf_grid
def get_normalize_mesh(model_file, norm_mesh_sub_dir):
    total = 16384
    print("trimesh_load:", model_file)
    mesh_list = trimesh.load_mesh(model_file, process=False)
    print("[*] done!", model_file)

    # NOTE: used to load with pymesh!
    #       change to trimesh
    # pymesh.load_mesh(model_file)

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
        print("start sample surface of ", mesh.faces.shape[0])
        points, index = trimesh.sample.sample_surface(mesh, amount_lst[i])
        print("end sample surface")
        points_all = np.concatenate([points_all,points], axis=0)
    centroid = np.mean(points_all, axis=0)
    points_all = points_all - centroid
    m = np.max(np.sqrt(np.sum(points_all ** 2, axis=1)))
    if '/pix3d/' in model_file:
        model_basename = os.path.basename(model_file)
        pc_norm_name = model_basename.replace('model', 'pc_norm')
        obj_file = os.path.join(norm_mesh_sub_dir, pc_norm_name)
    else:
        obj_file = os.path.join(norm_mesh_sub_dir, "pc_norm.obj")

    # NOTE: used to load with pymesh!
    #       change to trimesh
    # ori_mesh = pymesh.load_mesh(model_file)
    # print("centroid, m", centroid, m)
    # try:
        # pymesh.save_mesh_raw(obj_file, (ori_mesh.vertices - centroid) / float(m), ori_mesh.faces)
    # except:
        # import pdb; pdb.set_trace()

    ori_mesh_list = trimesh.load_mesh(model_file, process=False)
    ori_mesh = as_mesh(ori_mesh_list)
    ori_mesh.vertices = (ori_mesh.vertices - centroid) / float(m)
    ori_mesh.export(obj_file)
    
    # print("export_mesh", obj_file)
    # print('EXIST?????', os.path.exists(obj_file), obj_file)
    return obj_file, centroid, m


def create_one_sdf(sdfcommand, res, expand_rate, sdf_file, obj_file, indx, g=0.0):

    command_str = sdfcommand + " " + obj_file + " " + str(res) + " " + str(res) + \
       " " + str(res) + " -s " + " -e " + str(expand_rate) + " -o " + str(indx) + ".dist -m 1"
    command_str += ' -c'
    if g > 0.0:
        command_str += " -g " + str(g)
    print("command:", command_str)
    os.system(command_str)
    command_str2 = "mv " + str(indx)+".dist " + sdf_file
    print("command:", command_str2)
    os.system(command_str2)


# s2s
# def create_sdf_obj(sdfcommand, marching_cube_command, cat_mesh_dir, cat_norm_mesh_dir, cat_norm_sdf_dir, cat_sdf_dir, obj,
#        res, iso_val, expand_rate, indx, ish5, normalize, num_sample, bandwidth, max_verts, cat_id, g, reduce):

def create_sdf_obj(sdfcommand, marching_cube_command, norm_mesh_dir, sdf_dir, obj,
       res, iso_val, expand_rate, indx, ish5, normalize, num_sample, bandwidth, max_verts, g, reduce):
    # obj = obj.rstrip('\r\n')
    # model_file = obj

    # norm_mesh_sub_dir = os.path.join(norm_mesh_dir, os.path.basename(obj).replace('.obj', ''))
    # sdf_sub_dir = os.path.join(sdf_dir, os.path.basename(obj).replace('.obj', ''))

    if FLAGS.dset == 'abc':
        model_id = os.path.basename(obj).replace('.obj', '')
    elif FLAGS.dset == 'pix3d':
        model_id = obj.split('/')[-2]
    elif FLAGS.dset == 'building':
        model_id = os.path.basename(obj).replace('.obj', '')
    elif FLAGS.dset == 'shapenet':
        model_id = obj.split('/')[-2]
    norm_mesh_sub_dir = os.path.join(norm_mesh_dir, model_id)
    sdf_sub_dir = os.path.join(sdf_dir, model_id)

    if not os.path.exists(norm_mesh_sub_dir): os.makedirs(norm_mesh_sub_dir)
    if not os.path.exists(sdf_sub_dir): os.makedirs(sdf_sub_dir)

    if FLAGS.dset == 'pix3d':
        obj_basename = os.path.basename(obj).replace('.obj', '')
        sdf_name = obj_basename.replace('model', 'isosurf')
        flag_name = obj_basename.replace('model', 'isinsideout')
        h5_name = obj_basename.replace('model', 'ori_sample_grid')

        sdf_file = os.path.join(sdf_sub_dir, f"{sdf_name}.sdf")
        flag_file = os.path.join(sdf_sub_dir, f"{flag_name}.txt")
        h5_file = os.path.join(sdf_sub_dir, f"{h5_name}.h5")
    else:
        sdf_file = os.path.join(sdf_sub_dir, "isosurf.sdf")
        flag_file = os.path.join(sdf_sub_dir, "isinsideout.txt")
        h5_file = os.path.join(sdf_sub_dir, "ori_sample_grid.h5")

    if ish5 and os.path.exists(h5_file) and not os.path.exists(flag_file):
        print("skip existed: ", h5_file)
    elif not ish5 and os.path.exists(sdf_file):
        print("skip existed: ", sdf_file)
    else:
        # model_file = os.path.join(cat_mesh_dir, obj, "models", "model_normalized.obj")
        # model_file = os.path.join(cat_mesh_dir, obj, "model.obj")
        model_file = os.path.join(obj)
        print("creating", sdf_file)
        if normalize:
            norm_obj_file, centroid, m = get_normalize_mesh(model_file, norm_mesh_sub_dir)

        create_one_sdf(sdfcommand, res, expand_rate, sdf_file, norm_obj_file, indx, g=g)
        # create_one_cube_obj(marching_cube_command, iso_val, sdf_file, cube_obj_file)
        # change to h5
        if ish5:
            create_h5_sdf_pt(h5_file, sdf_file, flag_file, norm_obj_file,
                 centroid, m, res, num_sample, bandwidth, iso_val, max_verts, normalize, reduce=reduce)
        # except:
        #     print("%%%%%%%%%%%%%%%%%%%%%%%% fail to process ", model_file)

def create_one_cube_obj(marching_cube_command, i, sdf_file, cube_obj_file):
    command_str = marching_cube_command + " " + sdf_file + " " + cube_obj_file + " -i " + str(i)
    print("command:", command_str)
    os.system(command_str)
    return cube_obj_file

def create_sdf_abc(sdfcommand, marching_cube_command, LIB_command,
               num_sample, bandwidth, res, expand_rate, raw_dirs, iso_val,
               max_verts, ish5=True, normalize=True, g=0.00, reduce=4):
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
    for split in ['train', 'test']:
        
        model_dir = os.path.join(raw_dirs['mesh_dir'], split, '2048')
        norm_mesh_dir = os.path.join(raw_dirs["norm_mesh_dir"], split)
        sdf_dir = os.path.join(raw_dirs["sdf_dir"], split)

        if not os.path.exists(sdf_dir): os.makedirs(sdf_dir)
        if not os.path.exists(norm_mesh_dir): os.makedirs(norm_mesh_dir)

        # list_obj = os.listdir(model_dir)
        list_obj = [os.path.join(model_dir, f) for f in os.listdir(model_dir)]
        repeat = len(list_obj)
        sdfcommand_lst=[sdfcommand for i in range(repeat)]
        marching_cube_command_lst=[marching_cube_command for i in range(repeat)]
        norm_mesh_dir_lst=[norm_mesh_dir for i in range(repeat)] # by yc
        sdf_dir_lst=[sdf_dir for i in range(repeat)]
        res_lst=[res for i in range(repeat)]
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

        with Parallel(n_jobs=5) as parallel:
            parallel(delayed(create_sdf_obj)
            (sdfcommand, marching_cube_command, norm_mesh_dir, sdf_dir, obj, res, iso_val, expand_rate, indx, ish5, norm, num_sample, bandwidth, max_verts, g, reduce)
            for sdfcommand, marching_cube_command, norm_mesh_dir, sdf_dir, obj, res, iso_val, 
                expand_rate, indx, ish5, norm, num_sample, bandwidth, max_verts, g, reduce
                in zip(sdfcommand_lst,
                       marching_cube_command_lst,
                       norm_mesh_dir_lst,
                       sdf_dir_lst,
                       list_obj,
                       res_lst,
                       iso_val_lst,
                       expand_rate_lst,
                       indx_lst, ish5_lst, normalize_lst, num_sample_lst,
                       bandwidth_lst, max_verts_lst, g_lst, reduce_lst))

        # debug
        # for (sdfcommand, marching_cube_command, norm_mesh_dir, sdf_dir, obj, res, iso_val, 
        #     expand_rate, indx, ish5, norm, num_sample, bandwidth, max_verts, g, reduce) in \
        #     zip(sdfcommand_lst, marching_cube_command_lst, norm_mesh_dir_lst, sdf_dir_lst, list_obj,
        #         res_lst, iso_val_lst, expand_rate_lst, indx_lst, ish5_lst, normalize_lst, num_sample_lst,
        #         bandwidth_lst, max_verts_lst, g_lst, reduce_lst):
        #         create_sdf_obj(
        #             sdfcommand, marching_cube_command, norm_mesh_dir, sdf_dir, obj, res, 
        #             iso_val, expand_rate, indx, ish5, norm, num_sample, bandwidth, max_verts, g, reduce)
        start+=repeat
    print("finish all")


def create_sdf_pix3d(sdfcommand, marching_cube_command, LIB_command,
               num_sample, bandwidth, res, expand_rate,
               lst_dir, all_cats, raw_dirs, iso_val,
               max_verts, ish5=True, normalize=True, g=0.00, reduce=4):
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

    """ load input text. this is for chair. """
    # input_txt = '../../data/pix3d/input.txt'
    # with open(input_txt, 'r') as f:
    #     lines = [l.strip('\n')[3:] for l in f.readlines()] # no ../

    # gt_txt = input_txt.replace('input', 'gt')
    # with open(gt_txt, 'r') as f:
    #     gt_lines = [l.strip('\n')[3:] for l in f.readlines()] # no ../

    lst_dir
    dataroot = lst_dir.split('/pix3d/filelists')[0]
    with open(f'{dataroot}/pix3d/pix3d.json', 'r') as f:
        pix3d_info = json.load(f)
    pix3d_root = f'{dataroot}/pix3d'

    # map_input_to_info = {}
    # for d in pix3d_info:
    #     img_name = d['img']
    #     img_name = os.path.splitext(img_name)[0]
    #     map_input_to_info[img_name] = d


    for cat in all_cats:

        # if cat != 'chair':
            # continue

        model_dir = os.path.join(raw_dirs['mesh_dir'], cat)
        norm_mesh_dir = os.path.join(raw_dirs["norm_mesh_dir"], cat)
        sdf_dir = os.path.join(raw_dirs["sdf_dir"], cat)

        if not os.path.exists(norm_mesh_dir): os.makedirs(norm_mesh_dir)
        if not os.path.exists(sdf_dir): os.makedirs(sdf_dir)

        # list_obj = os.listdir(model_dir)
        list_model_id = []
        train_lst = f'{lst_dir}/{cat}_train.lst'
        test_lst = f'{lst_dir}/{cat}_test.lst'
        with open(train_lst, 'r') as f:
            list_model_id = f.readlines()
        with open(test_lst, 'r') as f:
            list_model_id += f.readlines()

        # get all obj file
        list_obj = []
        for model_id in list_model_id:  

            # again, different case for chair
            if cat == 'chair':

                # here basically just copy from 'pix3d_align_shapenet'
                p = model_id
                p = p.rstrip('\n')
                model_id = p.split('/')[-2]

                # # find gt voxel file. 
                # img_name = os.path.basename(p)
                # img_name = os.path.splitext(img_name)[0]
                # key = f'img/{cat}/{img_name}'
                # pix3d_img_name = map_input_to_info[key]['img']

                # # test file
                # if pix3d_img_name in lines:
                #     ix = lines.index(pix3d_img_name)
                #     gt_voxel_name = gt_lines[ix]
                #     gt_voxel_bn = os.path.basename(gt_voxel_name)
                #     obj_bn = gt_voxel_bn.replace('voxel', 'model')
                #     obj_bn = obj_bn.replace('.mat', '.obj')
            
                #     obj_f = f'{pix3d_root}/model_align/{cat}/{model_id}/{obj_bn}'

                #     if 'IKEA_JULES_1' in obj_f:
                #         kkk = obj_f
                #         import pdb; pdb.set_trace()

                # # train file.
                # else:
                #     # train file.
                #     # take the first one. for some model, there are multiple obj files
                #     obj_f = glob.glob(f'{pix3d_root}/model_align/{cat}/{model_id}/*.obj')[0]

                obj_files = glob.glob(f'{model_dir}/{model_id}/*.obj')#[0]

                # list_obj += obj_files
                # if obj_f not in list_obj:
                #     list_obj.append(obj_f)
            else:
                model_id = model_id.rstrip('\n')
                obj_files = glob.glob(f'{model_dir}/{model_id}/*.obj')#[0]
                # list_obj.append(obj_f)
            
            for obj_f in obj_files:
                if obj_f not in list_obj:
                    list_obj.append(obj_f)

        # list_obj = [kkk]
        # import pdb; pdb.set_trace()

        repeat = len(list_obj)
        sdfcommand_lst=[sdfcommand for i in range(repeat)]
        marching_cube_command_lst=[marching_cube_command for i in range(repeat)]
        norm_mesh_dir_lst=[norm_mesh_dir for i in range(repeat)] # by yc
        sdf_dir_lst=[sdf_dir for i in range(repeat)]
        res_lst=[res for i in range(repeat)]
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

        # not sure why, n_jobs=2 fails...
        with Parallel(n_jobs=5) as parallel:
            parallel(delayed(create_sdf_obj)
            (sdfcommand, marching_cube_command, norm_mesh_dir, sdf_dir, obj, res, iso_val, expand_rate, indx, ish5, norm, num_sample, bandwidth, max_verts, g, reduce)
            for sdfcommand, marching_cube_command, norm_mesh_dir, sdf_dir, obj, res, iso_val, 
                expand_rate, indx, ish5, norm, num_sample, bandwidth, max_verts, g, reduce
                in zip(sdfcommand_lst,
                       marching_cube_command_lst,
                       norm_mesh_dir_lst,
                       sdf_dir_lst,
                       list_obj,
                       res_lst,
                       iso_val_lst,
                       expand_rate_lst,
                       indx_lst, ish5_lst, normalize_lst, num_sample_lst,
                       bandwidth_lst, max_verts_lst, g_lst, reduce_lst))

        # debug
        # for (sdfcommand, marching_cube_command, norm_mesh_dir, sdf_dir, obj, res, iso_val, 
        #     expand_rate, indx, ish5, norm, num_sample, bandwidth, max_verts, g, reduce) in \
        #     zip(sdfcommand_lst, marching_cube_command_lst, norm_mesh_dir_lst, sdf_dir_lst, list_obj,
        #         res_lst, iso_val_lst, expand_rate_lst, indx_lst, ish5_lst, normalize_lst, num_sample_lst,
        #         bandwidth_lst, max_verts_lst, g_lst, reduce_lst):
        #         create_sdf_obj(
        #             sdfcommand, marching_cube_command, norm_mesh_dir, sdf_dir, obj, res, 
        #             iso_val, expand_rate, indx, ish5, norm, num_sample, bandwidth, max_verts, g, reduce)

        start+=repeat
    print("finish all")



def create_sdf_building(sdfcommand, marching_cube_command, LIB_command,
               num_sample, bandwidth, res, expand_rate, raw_dirs, iso_val,
               max_verts, ish5=True, normalize=True, g=0.00, reduce=4):
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

    resolution = int(res // reduce)

    model_dir = os.path.join(raw_dirs['mesh_dir'], 'OBJ_MODELS')
    norm_mesh_dir = os.path.join(raw_dirs["norm_mesh_dir"])
    sdf_dir = os.path.join(raw_dirs["sdf_dir"], f'resolution_{resolution}')

    if not os.path.exists(sdf_dir): os.makedirs(sdf_dir)
    if not os.path.exists(norm_mesh_dir): os.makedirs(norm_mesh_dir)

    # list_obj = os.listdir(model_dir)
    list_obj = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if '.obj' in f]
    
    repeat = len(list_obj)
    sdfcommand_lst=[sdfcommand for i in range(repeat)]
    marching_cube_command_lst=[marching_cube_command for i in range(repeat)]
    norm_mesh_dir_lst=[norm_mesh_dir for i in range(repeat)] # by yc
    sdf_dir_lst=[sdf_dir for i in range(repeat)]
    res_lst=[res for i in range(repeat)]
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

    with Parallel(n_jobs=5) as parallel:
        parallel(delayed(create_sdf_obj)
        (sdfcommand, marching_cube_command, norm_mesh_dir, sdf_dir, obj, res, iso_val, expand_rate, indx, ish5, norm, num_sample, bandwidth, max_verts, g, reduce)
        for sdfcommand, marching_cube_command, norm_mesh_dir, sdf_dir, obj, res, iso_val, 
            expand_rate, indx, ish5, norm, num_sample, bandwidth, max_verts, g, reduce
            in zip(sdfcommand_lst,
                    marching_cube_command_lst,
                    norm_mesh_dir_lst,
                    sdf_dir_lst,
                    list_obj,
                    res_lst,
                    iso_val_lst,
                    expand_rate_lst,
                    indx_lst, ish5_lst, normalize_lst, num_sample_lst,
                    bandwidth_lst, max_verts_lst, g_lst, reduce_lst))

    # debug
    # for (sdfcommand, marching_cube_command, norm_mesh_dir, sdf_dir, obj, res, iso_val, 
    #     expand_rate, indx, ish5, norm, num_sample, bandwidth, max_verts, g, reduce) in \
    #     zip(sdfcommand_lst, marching_cube_command_lst, norm_mesh_dir_lst, sdf_dir_lst, list_obj,
    #         res_lst, iso_val_lst, expand_rate_lst, indx_lst, ish5_lst, normalize_lst, num_sample_lst,
    #         bandwidth_lst, max_verts_lst, g_lst, reduce_lst):
    #         create_sdf_obj(
    #             sdfcommand, marching_cube_command, norm_mesh_dir, sdf_dir, obj, res, 
    #             iso_val, expand_rate, indx, ish5, norm, num_sample, bandwidth, max_verts, g, reduce)
    #         import pdb; pdb.set_trace()
    start+=repeat
    print("finish all")



def create_sdf_shapenet(sdfcommand, marching_cube_command, LIB_command,
               num_sample, bandwidth, res, expand_rate,
               lst_dir, all_cats, raw_dirs, iso_val,
               max_verts, ish5=True, normalize=True, g=0.00, reduce=4):
    '''
    Usage: SDFGen <filename> <dx> <padding>
    Where:
        res is number of grids on xyz dimension
        w is narrowband width
        expand_rate is sdf range of max x,y,z
    '''
    os.system(LIB_command)
    start=0

    """ load input text. this is for chair. """

    print("command:", LIB_command)
    # import subprocess
    # subprocess.run(LIB_command)
    os.system(LIB_command)
    # import pdb; pdb.set_trace()

    resolution = int(res // reduce)
    sdf_root = raw_dirs["sdf_dir"]
    sdf_dir = os.path.join(sdf_root, f'resolution_{resolution}') # .../ShapeNet/SDF_v1
    if not os.path.exists(sdf_dir): os.makedirs(sdf_dir)

    # map_input_to_info = {}
    # for d in pix3d_info:
    #     img_name = d['img']
    #     img_name = os.path.splitext(img_name)[0]
    #     map_input_to_info[img_name] = d

    # sanity check: all files exists
    for catnm in all_cats:

        print(f'[*] checking obj files in {catnm} ({cats[catnm]})')

        cat_id = cats[catnm]
        cat_mesh_dir = os.path.join(raw_dirs["mesh_dir"], cat_id)
        with open(lst_dir+"/"+str(cat_id)+"_test.lst", "r") as f:
            list_obj = f.readlines()

        with open(lst_dir+"/"+str(cat_id)+"_train.lst", "r") as f:
            list_obj += f.readlines()

        list_obj = [f.rstrip() for f in list_obj]
        list_obj = [f'{cat_mesh_dir}/{f}/model.obj' for f in list_obj]

        for f in list_obj:
            if not os.path.exists(f):
                print(f)
                import pdb; pdb.set_trace()
            assert os.path.exists(f)

        print(f'[*] all files exist for {catnm} ({cats[catnm]})!')


    for catnm in all_cats:

        cat_id = cats[catnm]
        cat_sdf_dir = os.path.join(sdf_dir, cat_id)
        if not os.path.exists(cat_sdf_dir): os.makedirs(cat_sdf_dir)
        cat_mesh_dir = os.path.join(raw_dirs["mesh_dir"], cat_id)
        cat_norm_mesh_dir = os.path.join(raw_dirs["norm_mesh_dir"], cat_id)

        with open(lst_dir+"/"+str(cat_id)+"_test.lst", "r") as f:
            list_obj = f.readlines()
        with open(lst_dir+"/"+str(cat_id)+"_train.lst", "r") as f:
            list_obj += f.readlines()

        list_obj = [f.rstrip() for f in list_obj]
        list_obj = [f'{cat_mesh_dir}/{f}/model.obj' for f in list_obj]

        # model_dir = os.path.join(raw_dirs['mesh_dir'], cat)
        # norm_mesh_dir = os.path.join(raw_dirs["norm_mesh_dir"], cat)
        # cat_sdf_dir = os.path.join(raw_dirs["sdf_dir"], cat)
        # if not os.path.exists(norm_mesh_dir): os.makedirs(norm_mesh_dir)
        # if not os.path.exists(cat_sdf_dir): os.makedirs(cat_sdf_dir)

        repeat = len(list_obj)
        sdfcommand_lst=[sdfcommand for i in range(repeat)]
        marching_cube_command_lst=[marching_cube_command for i in range(repeat)]
        # norm_mesh_dir_lst=[norm_mesh_dir for i in range(repeat)] # by yc
        # sdf_dir_lst=[sdf_dir for i in range(repeat)]
        norm_mesh_dir_lst=[cat_norm_mesh_dir for i in range(repeat)] # by yc
        sdf_dir_lst=[cat_sdf_dir for i in range(repeat)] # by yc
        res_lst=[res for i in range(repeat)]
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

        # not sure why, n_jobs=2 fails...
        with Parallel(n_jobs=5) as parallel:
            parallel(delayed(create_sdf_obj)
            (sdfcommand, marching_cube_command, norm_mesh_dir, sdf_dir, obj, res, iso_val, expand_rate, indx, ish5, norm, num_sample, bandwidth, max_verts, g, reduce)
            for sdfcommand, marching_cube_command, norm_mesh_dir, sdf_dir, obj, res, iso_val, 
                expand_rate, indx, ish5, norm, num_sample, bandwidth, max_verts, g, reduce
                in zip(sdfcommand_lst,
                       marching_cube_command_lst,
                       norm_mesh_dir_lst,
                       sdf_dir_lst,
                       list_obj,
                       res_lst,
                       iso_val_lst,
                       expand_rate_lst,
                       indx_lst, ish5_lst, normalize_lst, num_sample_lst,
                       bandwidth_lst, max_verts_lst, g_lst, reduce_lst))

        # debug
        # for (sdfcommand, marching_cube_command, norm_mesh_dir, sdf_dir, obj, res, iso_val, 
        #     expand_rate, indx, ish5, norm, num_sample, bandwidth, max_verts, g, reduce) in \
        #     zip(sdfcommand_lst, marching_cube_command_lst, norm_mesh_dir_lst, sdf_dir_lst, list_obj,
        #         res_lst, iso_val_lst, expand_rate_lst, indx_lst, ish5_lst, normalize_lst, num_sample_lst,
        #         bandwidth_lst, max_verts_lst, g_lst, reduce_lst):
        #         create_sdf_obj(
        #             sdfcommand, marching_cube_command, norm_mesh_dir, sdf_dir, obj, res, 
        #             iso_val, expand_rate, indx, ish5, norm, num_sample, bandwidth, max_verts, g, reduce)

        start+=repeat
    print("finish all")


# def test_sdf(sdf_h5_file):
#     h5_f = h5py.File(sdf_h5_file, 'r')
#     red = np.asarray([255.0, 0, 0]).astype(np.float32)
#     blue = np.asarray([0, 0, 255.0]).astype(np.float32)
#     try:
#         if ('pc_sdf_original' in h5_f.keys() and 'pc_sdf_sample' in h5_f.keys()):
#             ori_sdf = h5_f['pc_sdf_original'][:]
#             sample_sdf = np.reshape(h5_f['pc_sdf_sample'][:], (-1, 4))
#             ori_pt, ori_sdf_val = ori_sdf[:, :3], ori_sdf[:, 3]
#             sample_pt, sample_sdf_val = sample_sdf[:, :3], sample_sdf[:, 3]
#             minval, maxval = np.min(ori_sdf_val), np.max(ori_sdf_val)
#             sdf_pt_color = np.zeros([ori_pt.shape[0], 6], dtype=np.float32)
#             sdf_pt_color[:, :3] = ori_pt
#             for i in range(sdf_pt_color.shape[0]):
#                 sdf_pt_color[i, 3:] = red + (blue - red) * (
#                             float(ori_sdf_val[i] - minval) / float(maxval - minval))
#             np.savetxt("./ori.txt", sdf_pt_color)
#
#             sample_pt_color = np.zeros([sample_pt.shape[0], 6], dtype=np.float32)
#             sample_pt_color[:, :3] = sample_pt
#             minval, maxval = np.min(sample_sdf_val), np.max(sample_sdf_val)
#             for i in range(sample_pt_color.shape[0]):
#                 sample_pt_color[i, 3:] = red + (blue - red) * (
#                         float(sample_sdf_val[i] - minval) / float(maxval - minval))
#             np.savetxt("./sample.txt", sample_pt_color)
#     finally:
#         h5_f.close()



if __name__ == "__main__":

    # nohup python -u create_point_sdf_fullgrid.py &> createfull.log &
    # lst_dir, cats, all_cats, raw_dirs = create_file_lst_abc.get_all_info()
    
    dset = FLAGS.dset
    cat = FLAGS.category

    # lst_dir, cats, all_cats, raw_dirs = get_sdf_file_lst.get_all_info(dset)

    info_file = '../dataset_info_files/info-shapenet.json'
    with open(info_file) as json_file:
        info_data = json.load(json_file)
        lst_dir, cats, all_cats, raw_dirs = info_data["lst_dir"], info_data['cats'], info_data['all_cats'], info_data['raw_dirs_v1']

    if dset == 'shapenet':
        if cat != 'all':
            cats = {cat: cats[cat]}
    elif dset != 'abc':
        if cat == 'all':
            FLAGS.cats = all_cats
        else:
            FLAGS.cats = [cat]
    
    isosurface_dir = './isosurface'
    sdf_cmd = f'{isosurface_dir}/computeDistanceField'
    mcube_cmd = f'{isosurface_dir}/computeMarchingCubes'
    lib_cmd = f'{isosurface_dir}/LIB_PATH'

    # set env variable
    os.environ['LD_LIBRARY_PATH'] = f'$LD_LIBRARY_PATH:{isosurface_dir}:./isosurface/tbb/tbb2018_20180822oss/lib/intel64/gcc4.7/'

    num_sample = 65 ** 3
    bandwidth = 0.1
    res = 256
    reduce = FLAGS.reduce
    expand_rate = 1.3
    iso_val = 0.003
    max_verts = 16384

    #  full set
    # create_sdf(sdfcommand,
    #            mcube_cmd,
    #            "source %s" % lib_cmd, 274625, 0.1,
    #            256, 1.3, all_cats, cats, raw_dirs,
    #            lst_dir, 0.003, 16384, ish5=True, normalize=True, g=0.00, reduce=4)
    # create_sdf_obj(sdfcommand,
    #            mcube_cmd,
    #            "source %s" % lib_cmd, num_sample, bandwidth,
    #            res, expand_rate, raw_dirs, iso_val, max_verts, ish5=True, normalize=True, g=0.00, reduce=4)

    if dset == 'abc':
        create_sdf_abc(sdf_cmd, mcube_cmd, "source %s" % lib_cmd,
                   num_sample, bandwidth, res, expand_rate, raw_dirs, iso_val,
                   max_verts, ish5=True, normalize=True, g=0.00, reduce=reduce)
    elif dset == 'pix3d':
        create_sdf_pix3d(sdf_cmd, mcube_cmd, "source %s" % lib_cmd,
                   num_sample, bandwidth, res, expand_rate,
                   lst_dir, FLAGS.cats, raw_dirs,
                   iso_val, max_verts, ish5=True, normalize=True, g=0.00, reduce=reduce)
    elif dset == 'building':
        create_sdf_building(sdf_cmd, mcube_cmd, "source %s" % lib_cmd,
                   num_sample, bandwidth, res, expand_rate, raw_dirs, iso_val,
                   max_verts, ish5=True, normalize=True, g=0.00, reduce=reduce)
    elif dset == 'shapenet':
        # cats: synset_name: synset_id
        create_sdf_shapenet(sdf_cmd, mcube_cmd, "source %s" % lib_cmd,
                   num_sample, bandwidth, res, expand_rate,
                   lst_dir, cats, raw_dirs,
                   iso_val, max_verts, ish5=True, normalize=True, g=0.00, reduce=reduce)

    