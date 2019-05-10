import glob
import os
import numpy as np
import random
import time

import menpo3d.io as m3io

from menpo.shape import TriMesh

from copy import deepcopy
from sklearn.decomposition import PCA
from tqdm import tqdm


class FaceData(object):
    def __init__(self, nVal, train_file, test_file, reference_mesh_file, pca_n_comp=8, fitpca=False):
        self.nVal = nVal
        self.train_file = train_file
        self.test_file = test_file
        self.vertices_train = None
        self.vertices_val = None
        self.vertices_test = None
        self.N = None
        self.n_vertex = None
        self.fitpca = fitpca
        self.mean = None
        self.std = None

        self.load()
        self.reference_mesh = m3io.import_mesh(filepath=reference_mesh_file)

        # self.mean = np.mean(self.vertices_train, axis=0)
        # self.std = np.std(self.vertices_train, axis=0)
        self.pca = PCA(n_components=pca_n_comp)
        self.pcaMatrix = None
        self.normalize()

    def load(self):
        vertices_train = np.load(self.train_file)
        self.mean = np.mean(vertices_train, axis=0)
        self.std = np.std(vertices_train, axis=0)

        if self.nVal > 0.1*vertices_train.shape[0]:
            self.nVal = np.int(np.floor(0.1*vertices_train.shape[0]))

        self.vertices_train = vertices_train[:-self.nVal]
        self.vertices_val = vertices_train[-self.nVal:]

        self.n_vertex = self.vertices_train.shape[1]

        self.vertices_test = np.load(self.test_file)
        self.vertices_test = self.vertices_test

    def normalize(self):
        self.vertices_train = self.vertices_train - self.mean
        self.vertices_train = self.vertices_train/self.std

        self.vertices_val = self.vertices_val - self.mean
        self.vertices_val = self.vertices_val/self.std

        self.vertices_test = self.vertices_test - self.mean
        self.vertices_test = self.vertices_test/self.std

        self.N = self.vertices_train.shape[0]

        if self.fitpca:
            self.pca.fit(np.reshape(self.vertices_train, (self.N, self.n_vertex*3)))
        # eigenVals = np.sqrt(self.pca.explained_variance_)
        # self.pcaMatrix = np.dot(np.diag(eigenVals), self.pca.components_)
        print('Vertices normalized')

    def vec2mesh(self, vec):
        vec = vec.reshape((self.n_vertex, 3))*self.std + self.mean
        return TriMesh(points=vec, trilist=self.reference_mesh.trilist)

    def show(self, ids):
        # ids: list of ids to play
        if max(ids) >= self.N:
            raise ValueError('id: out of bounds')

        mesh = TriMesh(points=self.vertices_train[ids[0]], trilist=self.reference_mesh.f)
        time.sleep(0.5)    # pause 0.5 seconds
        viewer = mesh.show()
        for i in range(len(ids)-1):
            viewer.dynamic_meshes = [TriMesh(points=self.vertices_train[ids[i+1]], trilist=self.reference_mesh.f)]
            time.sleep(0.5)    # pause 0.5 seconds
        return 0

    def sample(self, batch_size=64):
        datasamples = np.zeros((batch_size, self.vertices_train.shape[1]*self.vertices_train.shape[2]))
        for i in range(batch_size):
            _randid = random.randint(0, self.N-1)
            # print _randid
            datasamples[i] = ((deepcopy(self.vertices_train[_randid]) - self.mean)/self.std).reshape(-1)

        return datasamples

    def save_meshes(self, filename, meshes):
        for i in range(meshes.shape[0]):
            # Is next line required? Why?
            vertices = meshes[i].reshape((self.n_vertex, 3))*self.std + self.mean
            mesh = TriMesh(points=vertices, trilist=self.reference_mesh.f)
            m3io.export_mesh(mesh, filename+'-'+str(i).zfill(3)+'.ply')
        return 0

    def show_mesh(self, viewer, mesh_vecs, figsize):
        for i in range(figsize[0]):
            for j in range(figsize[1]):
                mesh_vec = mesh_vecs[i*(figsize[0]-1) + j]
                mesh_mesh = self.vec2mesh(mesh_vec)
                viewer[i][j].set_dynamic_meshes([mesh_mesh])
        time.sleep(0.1)  # pause 0.5 seconds
        return 0

    def get_normalized_meshes(self, mesh_paths):
        meshes = []
        for mesh_path in mesh_paths:
            mesh = m3io.import_mesh(filepath=mesh_path)
            mesh_v = (mesh.points - self.mean)/self.std
            meshes.append(mesh_v)
        return np.array(meshes)


# Commenting out for the moment, not nesessary
# def meshPlay(folder, every=100, wait=0.05):
#   files = glob.glob(folder+'/*')
#   files.sort()
#   files = files[-1000:]
#   view = TriMeshViewer()
#   for i in range(0,len(files),every):
#       mesh = TriMesh(filename=files[i])
#       view.dynamic_meshes = [mesh]
#       time.sleep(wait)


class MakeSlicedTimeDataset(object):
    """docstring for FaceTriMesh"""
    def __init__(self, folders, dataset_name):
        self.facial_motion_dirs = folders
        self.dataset_name = dataset_name

        self.train_datapaths = self.gather_paths("train")
        self.test_datapaths = self.gather_paths("test")

        self.train_vertices = self.gather_data(self.train_datapaths)
        self.test_vertices = self.gather_data(self.test_datapaths)
        self.save_vertices()

    def gather_paths(self, opt):
        datapaths = []
        for i in range(len(self.facial_motion_dirs)):
            print(self.facial_motion_dirs[i])
            datapaths += glob.glob(self.facial_motion_dirs[i]+'/*/*/*.ply')

        trainpaths = []
        testpaths = []
        for i in range(len(datapaths)):
            if (i % 100) < 10:
                testpaths += [datapaths[i]]
            else:
                trainpaths += [datapaths[i]]

        if opt == "train":
            print("{} data of size: {}".format(opt, len(trainpaths)))
            return trainpaths
        if opt == "test":
            print("{} data of size: {}".format(opt, len(testpaths)))
            return testpaths

    @staticmethod
    def gather_data(datapaths):
        vertices = []
        for p in tqdm(datapaths):
            mesh_file = p
            face_mesh = m3io.import_mesh(filepath=mesh_file)
            vertices.append(face_mesh.points)
        return np.array(vertices)

    def save_vertices(self):
        if not os.path.exists(self.dataset_name):
            os.makedirs(self.dataset_name)
        np.save(self.dataset_name+'/train', self.train_vertices)
        np.save(self.dataset_name+'/test', self.test_vertices)

        print("Saving {}...".format(self.dataset_name+'/train'))
        print("Saving {}...".format(self.dataset_name+'/test'))
        return 0


class MakeIdentityExpressionDataset(object):
    # docstring for FaceTriMesh
    def __init__(self, folders, test_exp, dataset_name, crossval="expression", use_templates=0):
        self.facial_motion_dirs = folders
        self.dataset_name = dataset_name
        self.test_exp = test_exp
        self.crossval = crossval

        self.train_datapaths = self.gather_paths("train")
        self.test_datapaths = self.gather_paths("test")

        self.train_vertices = self.gather_data(self.train_datapaths)
        self.test_vertices = self.gather_data(self.test_datapaths)
        self.save_vertices()

    def gather_paths(self, opt):
        datapaths = []
        for i in range(len(self.facial_motion_dirs)):
            print(self.facial_motion_dirs[i])
            datapaths += glob.glob(self.facial_motion_dirs[i]+'/*/*/*.ply')

        trainpaths = []
        testpaths = []

        if self.crossval == "expression":
            path_flagger = -2
        if self.crossval == "identity":
            path_flagger = -3

        for i in range(len(datapaths)):
            p = datapaths[i]
            if p.split('/')[path_flagger] == self.test_exp:
                testpaths += [datapaths[i]]
            else:
                trainpaths += [datapaths[i]]

        if opt == "train":
            print("{} data of size: {}".format(opt, len(trainpaths)))
            return trainpaths
        if opt == "test":
            print("{} data of size: {}".format(opt, len(testpaths)))
            return testpaths

    @staticmethod
    def gather_data(datapaths):
        vertices = []
        for p in tqdm(datapaths):
            mesh_file = p
            face_mesh = m3io.import_mesh(filepath=mesh_file)
            vertices.append(face_mesh.points)
        return np.array(vertices)

    def save_vertices(self):
        if not os.path.exists(self.dataset_name):
            os.makedirs(self.dataset_name)
        np.save(self.dataset_name+'/train', self.train_vertices)
        np.save(self.dataset_name+'/test', self.test_vertices)

        print("Saving {}...".format(self.dataset_name+'/train'))
        print("Saving {}...".format(self.dataset_name+'/test'))
        return 0


class MakeDataset(object):
    """Prepare data from folder shape_nicp for CoMA processing"""

    def __init__(self, folders, dataset_name):
        self.facial_motion_dirs = folders
        self.dataset_name = dataset_name

        self.train_datapaths = self.gather_paths("train")
        self.test_datapaths = self.gather_paths("test")
        print(len(self.train_datapaths), len(self.test_datapaths))

        self.train_vertices = self.gather_data(self.train_datapaths)
        self.test_vertices = self.gather_data(self.test_datapaths)
        print('Train: ', self.train_vertices.shape)
        print('Test: ', self.test_vertices.shape)
        self.save_vertices()

    def gather_paths(self, opt):
        datapaths = []
        for i in range(len(self.facial_motion_dirs)):
            print(self.facial_motion_dirs[i])
            datapaths += glob.glob(self.facial_motion_dirs[i] + '/*.obj')

        trainpaths = []
        testpaths = []
        for i in range(len(datapaths)):
            if (i % 100) < 10:
                testpaths += [datapaths[i]]
            else:
                trainpaths += [datapaths[i]]

        if opt == "train":
            print(opt + " data of size: ", len(trainpaths))
            return trainpaths
        if opt == "test":
            print(opt + " data of size: ", len(testpaths))
            return testpaths

    def gather_data(self, datapaths):
        vertices = []
        for mesh_file in tqdm(datapaths):
            vertices.append(m3io.import_mesh(mesh_file).points)
        return np.array(vertices)

    def save_vertices(self):
        if not os.path.exists(self.dataset_name):
            os.makedirs(self.dataset_name)
        np.save(self.dataset_name + '/train', self.train_vertices)
        np.save(self.dataset_name + '/test', self.test_vertices)

        print("Saving ... ", self.dataset_name + '/train')
        print("Saving ... ", self.dataset_name + '/test')
        return 0


class MakeEarDataset(object):
    """Prepare data from correspondence folders at path forshape_nicp for CoMA processing"""

    def __init__(self, folders, dataset_name):
        self.data_path_dirs = folders
        self.dataset_name = dataset_name

        self.train_datapaths = self.gather_paths("train")
        self.test_datapaths = self.gather_paths("test")

        self.train_vertices = self.gather_data(self.train_datapaths)
        self.test_vertices = self.gather_data(self.test_datapaths)
        print('Train: ', self.train_vertices.shape)
        print('Test: ', self.test_vertices.shape)
        self.save_vertices()


    def gather_paths(self, opt):
        datapaths = []
        for dpd in self.data_path_dirs:
            path_dirs = os.listdir(dpd)

            for d in path_dirs:
                if not os.path.isdir(dpd+'/'+d): continue
                print(dpd+d)
                datapaths += glob.glob(dpd+'/'+d+'/correspondence/*.obj')

        trainpaths = []
        testpaths = []
        for i in range(len(datapaths)):
            if (i % 100) < 10:
                testpaths += [datapaths[i]]
            else:
                trainpaths += [datapaths[i]]

        if opt == "train":
            print(opt + " data of size: ", len(trainpaths))
            return trainpaths
        if opt == "test":
            print(opt + " data of size: ", len(testpaths))
            return testpaths

    @staticmethod
    def gather_data(datapaths):
        vertices = []
        for mesh_file in tqdm(datapaths):
            vertices.append(m3io.import_mesh(mesh_file).points)
        return np.array(vertices)

    def save_vertices(self):
        if not os.path.exists(self.dataset_name):
            os.makedirs(self.dataset_name)
        np.save(self.dataset_name + '/train', self.train_vertices)
        np.save(self.dataset_name + '/test', self.test_vertices)

        print("Saving ... ", self.dataset_name + '/train')
        print("Saving ... ", self.dataset_name + '/test')
        return 0


def generateSlicedTimeDataSet(data_path, save_path):
    MakeSlicedTimeDataset(folders=[data_path], dataset_name=os.path.join(save_path, 'sliced'))
    return 0


def generateExpressionDataSet(data_path, save_path):
    test_exps = ['bareteeth', 'cheeks_in', 'eyebrow', 'high_smile', 'lips_back', 'lips_up', 'mouth_down',
                 'mouth_extreme', 'mouth_middle', 'mouth_open', 'mouth_side', 'mouth_up']

    for exp in test_exps:
        fm = MakeIdentityExpressionDataset(folders=[data_path], test_exp=exp,
                                           dataset_name=os.path.join(save_path, exp), use_templates=0)


def generateIdentityDataset(data_path, save_path):
    test_ids = ['FaceTalk_170725_00137_TA',  'FaceTalk_170731_00024_TA',  'FaceTalk_170811_03274_TA',
                'FaceTalk_170904_00128_TA',  'FaceTalk_170908_03277_TA',  'FaceTalk_170913_03279_TA',
                'FaceTalk_170728_03272_TA',  'FaceTalk_170809_00138_TA',  'FaceTalk_170811_03275_TA',
                'FaceTalk_170904_03276_TA',  'FaceTalk_170912_03278_TA',  'FaceTalk_170915_00223_TA']

    for ids in test_ids:
        fm = MakeIdentityExpressionDataset(folders=[data_path], test_exp=ids,
                                           dataset_name=os.path.join(save_path, ids),
                                           crossval="identity", use_templates=0)


def generateDataset(data_path, save_path):
    MakeEarDataset(folders=[data_path], dataset_name=save_path)
    return 0


def generateEarDataset(data_path, save_path):
    MakeEarDataset(folders=[data_path], dataset_name=save_path)
    return 0