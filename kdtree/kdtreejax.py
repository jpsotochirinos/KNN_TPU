import re
from string import printable
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import numpy as np
from jax import device_put

def kdtree( data, leafsize=10 ):

    ndim = data.shape[0]
    ndata = data.shape[1]

    # find bounding hyper-rectangle
    hrect = np.zeros((2,data.shape[0]))
    hrect[0,:] = data.min(axis=1)
    hrect[1,:] = data.max(axis=1)

    # create root of kd-tree
    idx = np.argsort(data[0,:], kind='mergesort')
    data[:,:] = data[:,idx]
    splitval = data[0,int(ndata/2)]

    left_hrect = hrect.copy()
    right_hrect = hrect.copy()
    left_hrect[1, 0] = splitval
    right_hrect[0, 0] = splitval

    tree = [(None, None, left_hrect, right_hrect, None, None)]

    stack = [(data[:,:int(ndata/2)], idx[:int(ndata/2)], 1, 0, True),
             (data[:,int(ndata/2):], idx[int(ndata/2):], 1, 0, False)]

    # recursively split data in halves using hyper-rectangles:
    while stack:

        # pop data off stack
        data, didx, depth, parent, leftbranch = stack.pop()
        ndata = data.shape[1]
        nodeptr = len(tree)

        # update parent node

        _didx, _data, _left_hrect, _right_hrect, left, right = tree[parent]

        tree[parent] = (_didx, _data, _left_hrect, _right_hrect, nodeptr, right) if leftbranch \
            else (_didx, _data, _left_hrect, _right_hrect, left, nodeptr)

        # insert node in kd-tree

        # leaf node?
        if ndata <= leafsize:
            _didx = didx.copy()
            _data = data.copy()
            leaf = (_didx, _data, None, None, 0, 0)
            tree.append(leaf)

        # not a leaf, split the data in two      
        else:
            splitdim = depth % ndim
            idx = np.argsort(data[splitdim,:], kind='mergesort')
            data[:,:] = data[:,idx]
            didx = didx[idx]
            nodeptr = len(tree)
            stack.append((data[:,:int(ndata/2)], didx[:int(ndata/2)], depth+1, nodeptr, True))
            stack.append((data[:,int(ndata/2):], didx[int(ndata/2):], depth+1, nodeptr, False))
            splitval = data[splitdim,int(ndata/2)]
            if leftbranch:
                left_hrect = _left_hrect.copy()
                right_hrect = _left_hrect.copy()
            else:
                left_hrect = _right_hrect.copy()
                right_hrect = _right_hrect.copy()
            left_hrect[1, splitdim] = splitval
            right_hrect[0, splitdim] = splitval
            # append node to tree
            tree.append((None, None, left_hrect, right_hrect, None, None))

    return tree
    #!python numbers=disable

def intersect(hrect, r2, centroid):
    """
    checks if the hyperrectangle hrect intersects with the
    hypersphere defined by centroid and r2
    """
    maxval = hrect[1,:]
    minval = hrect[0,:]
    p = centroid.copy()
    idx = p < minval
    p[idx] = minval[idx]
    idx = p > maxval
    p[idx] = maxval[idx]
    return ((p-centroid)**2).sum() < r2

def quadratic_knn_search(data, lidx, ldata, K,p):
    """ find K nearest neighbours of data among ldata """
    ndata = ldata.shape[1]
    param = ldata.shape[0]
    K = K if K < ndata else ndata
    retval = []
    data = np.resize(data,ldata.shape)
    sqd = ((ldata - data)**p).sum(axis=0) # data.reshape((param,1)).repeat(ndata, axis=1);
    idx = np.argsort(sqd, kind='mergesort')
    idx = idx[:K]
    return zip(sqd[idx], lidx[idx])

# def minkowsky_knn_search(data, lidx, ldata, K,p):
#     """ find K nearest neighbours of data among ldata """
#     ndata = ldata.shape[1]
#     param = ldata.shape[0]
#     K = K if K < ndata else ndata
#     retval = []
#     data = jnp.resize(data,ldata.shape)
#     sqd = ((ldata - data)**p).sum(axis=0) # data.reshape((param,1)).repeat(ndata, axis=1);
#     idx = jnp.argsort(sqd, kind='mergesort')
#     idx = idx[:K]
#     return zip(sqd[idx], lidx[idx])

# def pearson_knn_search(data, lidx, ldata, K,p):
#     """ find K nearest neighbours of data among ldata """
#     ndata = ldata.shape[1]
#     param = ldata.shape[0]
#     K = K if K < ndata else ndata
#     retval = []
#     data = jnp.resize(data,ldata.shape)
#     sqd = ((jnp.multiply(ldata,data)).sum(axis=0) - (jnp.multiply((data).sum(axis=0),(ldata).sum(axis=0)))/ndata)/jnp.multiply(jnp.sqrt(jnp.multiply(data,data).sum(axis=0)-jnp.power((data).sum(axis=0),2)/ndata),(jnp.sqrt(jnp.multiply(ldata,ldata).sum(axis=0)-jnp.power((ldata).sum(axis=0),2)/ndata)))
#     idx = jnp.argsort(sqd, kind='mergesort')[::-1]
#     idx = idx[:K]
#     return zip(sqd[idx], lidx[idx])

# def coseno_knn_search(data, lidx, ldata, K,p):
#     """ find K nearest neighbours of data among ldata """
#     ndata = ldata.shape[1]
#     param = ldata.shape[0]
#     K = K if K < ndata else ndata
#     retval = []
#     data = jnp.resize(data,ldata.shape)
#     sqd = jnp.divide((jnp.multiply(ldata,data).sum(axis=0)),(jnp.multiply(jnp.sqrt(jnp.multiply(data,data).sum(axis=0)),jnp.sqrt(jnp.multiply(ldata,ldata).sum(axis=0)))))
#     idx = jnp.argsort(sqd, kind='mergesort')[::-1]
#     idx = idx[:K]
#     return zip(sqd[idx], lidx[idx])
def minkowsky_knn_search(data, ldata,p):
    filter_ceros = ldata*data
    ldata = jnp.where(filter_ceros==0,0,ldata)
    data = jnp.where(filter_ceros==0,0,data)
    return ((ldata - data)**p).sum(axis=0)

def pearson_knn_search(data, ldata,ndata):
    filter_ceros = ldata*data
    ldata = jnp.where(filter_ceros==0,0,ldata)
    data = jnp.where(filter_ceros==0,0,data)
    return ((jnp.multiply(ldata,data)).sum(axis=0) - (jnp.multiply((data).sum(axis=0),(ldata).sum(axis=0)))/ndata)/jnp.multiply(jnp.sqrt(jnp.multiply(data,data).sum(axis=0)-jnp.power((data).sum(axis=0),2)/ndata),(jnp.sqrt(jnp.multiply(ldata,ldata).sum(axis=0)-jnp.power((ldata).sum(axis=0),2)/ndata)))

def coseno_knn_search(data, ldata):
    filter_ceros = ldata*data
    ldata = jnp.where(filter_ceros==0,0,ldata)
    data = jnp.where(filter_ceros==0,0,data)
    return jnp.divide((jnp.multiply(ldata,data).sum(axis=0)),(jnp.multiply(jnp.sqrt(jnp.multiply(data,data).sum(axis=0)),jnp.sqrt(jnp.multiply(ldata,ldata).sum(axis=0)))))

def get_matrix(data):
    ndata = data.shape[1]
    param = data.shape[0]
    matrix = np.zeros((param,param))
    zeros = jnp.count_nonzero(data,axis=1)
    summ = data.sum(axis=1)
    avrg = (summ/zeros).reshape(param,1).repeat(ndata, axis=1)
    _data = jnp.where(data>0,data-avrg,0)
    _data = _data.transpose()
    np.set_printoptions(precision=4)
    for i in range(param):
        for j in range(param):
            filter_ceros = _data[i]*_data[j]
            _data_i = jnp.where(filter_ceros==0,0,_data[i])
            _data_j = jnp.where(filter_ceros==0,0,_data[j])
            matrix[i][j]=((_data_i*_data_j).sum(axis=0))/(np.sqrt((_data_i*_data_i).sum(axis=0))*np.sqrt((_data_j*_data_j).sum(axis=0)))
    return matrix

def normalizar(data):
    ndata = data.shape[1]
    param = data.shape[0]
    n = data.sum()
    _min = jnp.where(data==0,n,data)
    mins = jnp.amin(_min,axis=1).reshape(param,1).repeat(ndata, axis=1)
    maxs = jnp.amax(data,axis=1).reshape(param,1).repeat(ndata, axis=1)
    _data = (2*(data-mins)-(maxs-mins))/((maxs-mins))
    filter_ceros = _data * data
    _data = jnp.where(filter_ceros==0,0,_data)
    return _data

def de_normalizar(data,res):
    ndata = data.shape[1]
    param = data.shape[0]
    n = data.sum()
    _min = jnp.where(data==0,n,data)
    mins = jnp.amin(_min,axis=1).reshape(param,1).repeat(ndata, axis=1)
    maxs = jnp.amax(data,axis=1).reshape(param,1).repeat(ndata, axis=1)
    new = (((res+1)*((maxs-mins)))/2)+mins
    #new = new.reshape(param,1).repeat(ndata, axis=1)
    filter_ceros = new * data
    new = jnp.where(filter_ceros==0,new,data)
    return new

def coseno_2_knn_search(matrix,normalize):
    return ((matrix*normalize).sum(axis=1))/(np.abs(matrix).sum(axis=1)-1)

def distance_knn_search(data, lidx, ldata, K,p,d=0):
    """ find K nearest neighbours of data among ldata """
    ndata = ldata.shape[1]
    param = ldata.shape[0]
    K = K if K < ndata else ndata
    retval = []
    data =data[:,0].reshape((param,1)).repeat(ndata, axis=1)
    jit_minkowsky_knn_search = jit(minkowsky_knn_search)
    jit_pearson_knn_search= jit(pearson_knn_search)
    jit_coseno_knn_search = jit(coseno_knn_search)
    switch = {
        0:jit_minkowsky_knn_search(data,ldata,p).block_until_ready(),
        1:jit_pearson_knn_search(data,ldata,ndata).block_until_ready(),
        2:jit_coseno_knn_search(data,ldata).block_until_ready(),
    }
    distance = jnp.array(switch.get(d,"Invalid input"))
    if(d == 0):
        idx = jnp.argsort(distance, kind='stable')
        idx = idx[:K]
    elif(d == 1):
        idx = jnp.argsort(distance, kind='stable')[::-1]
        idx = idx[:K]
    elif(d == 2):
        idx = jnp.argsort(distance, kind='stable')[::1]
        idx = idx[:K]
    lidx = jnp.array(lidx)
    return zip(distance[idx], lidx[idx])

def search_kdtree(tree, datapoint, K,p,d=0):
    """ find the k nearest neighbours of datapoint in a kdtree """
    stack = [tree[0]]
    knn = [(np.inf, None)]*K
    _datapt = datapoint[:,0]
    while stack:

        leaf_idx, leaf_data, left_hrect, \
                  right_hrect, left, right = stack.pop()

        # leaf
        if leaf_idx is not None:
            _knn = distance_knn_search(datapoint, leaf_idx, leaf_data, K,p,d)
            _knn = list(_knn)
            if _knn[0][0] < knn[-1][0]:
                knn = sorted(knn + _knn)[:K]

        # not a leaf
        else:

            # check left branch
            if intersect(left_hrect, knn[-1][0], _datapt):
                stack.append(tree[left])

            # chech right branch
            if intersect(right_hrect, knn[-1][0], _datapt):
                stack.append(tree[right])
    if(d==0):
        return knn
    elif(d==1):
        return _knn
    elif(d==2):
        return _knn

def knn_search_all( data, K,p,d=0, leafsize=1028):

    """ find the K nearest neighbours for data points in data,
        using an O(n log n) kd-tree """

    ndata = data.shape[1]
    param = data.shape[0]

    # build kdtree
    tree = kdtree(data.copy(), leafsize=ndata)

    # search kdtree
    knn = []
    for i in np.arange(ndata):
        _data = data[:,i].reshape((param,1)).repeat(ndata, axis=1)
        _knn = search_kdtree(tree, _data, K+1,p,d)
        knn.append(_knn[1:])
    return knn

def knn_search_by_point( data,point, K,p,d=0):

    """ find the K nearest neighbours for data points in data,
        using an O(n log n) kd-tree """

    ndata = data.shape[1]
    param = data.shape[0]

    # build kdtree
    tree = kdtree(data.copy(), leafsize=ndata)

    # search kdtree
    knn = []
    _data = point.reshape((param,1)).repeat(ndata, axis=1)
    _knn = search_kdtree(tree, _data, K+1,p,d)
    knn.append(_knn[1:])
    return knn
# def knn_search( data, K ,p,d=0):
#     """ find the K nearest neighbours for data points in data,
#         using O(n**2) search """
#     ndata = data.shape[1]
#     knn = []
#     idx = jnp.arange(ndata)
#     for i in jnp.arange(ndata):
#         _knn = distance_knn_search(data[:,i], idx, data, K+1,p,d) # see above
#         aux = list(_knn)
#         knn.append(aux)
#     return knn

try:
    import multiprocessing as processing
except:
    import processing

import ctypes, os

def __num_processors():
    if os.name == 'nt': # Windows
        return int(os.getenv('NUMBER_OF_PROCESSORS'))
    else: # glibc (Linux, *BSD, Apple)
        get_nprocs = ctypes.cdll.LoadLibrary("libc.so.6").get_nprocs
        get_nprocs.restype = ctypes.c_int
        get_nprocs.argtypes = []
        return get_nprocs()

def __search_kdtree(tree, data, K, leafsize,p):
    knn = []
    param = data.shape[0]
    ndata = data.shape[1]
    for i in np.arange(ndata):
        _data = data[:,i].reshape((param,1)).repeat(leafsize, axis=1)
        _knn = search_kdtree(tree, _data, K+1,p)
        knn.append(_knn[1:])      
    return knn

def __remote_process(rank, qin, qout, tree, K, leafsize,p):
    while 1:
        # read input queue (block until data arrives)
        nc, data = qin.get()
        
        # process data
        knn = __search_kdtree(tree, data, K, leafsize,p)
        np.save('knn.npy',knn)
        # write to output queue
        qout.put((nc,knn))

def knn_search_parallel(data, K,p, leafsize=1028):

    """ find the K nearest neighbours for data points in data,
        using an O(n log n) kd-tree, exploiting all logical
        processors on the computer """

    ndata = data.shape[1]
    param = data.shape[0]
    nproc = __num_processors()
    # build kdtree
    tree = kdtree(data.copy(), leafsize=leafsize)
    # compute chunk size
    chunk_size = data.shape[1] / (4*nproc)
    chunk_size = 100 if chunk_size < 100 else chunk_size
    # set up a pool of processes
    qin = processing.Queue(maxsize=int(ndata/chunk_size))
    qout = processing.Queue(maxsize=int(ndata/chunk_size))
    pool = [processing.Process(target=__remote_process,
                args=(rank, qin, qout, tree, K, leafsize,p))
                    for rank in range(nproc)]
    for p in pool: p.start()
    # put data chunks in input queue
    cur, nc = 0, 0
    while 1:
        _data = data[:,cur:cur+int(chunk_size)]
        if _data.shape[1] == 0: break
        qin.put((nc,_data))
        cur += int(chunk_size)
        nc += 1
    # read output queue
    knn = []
    while len(knn) < nc:
        knn += [qout.get()]
    # avoid race condition
    _knn = [n for i,n in sorted(knn)]
    knn = []
    for tmp in _knn:
        knn += tmp
    # terminate workers
    for p in pool: p.terminate()
    return knn

from time import process_time
import csv
key = random.PRNGKey(0)

def test():
    K = 6
    ndata = 610
    ndim = 193609
    #data =  10 * np.random.rand(ndata*ndim).reshape((ndim,ndata) )
    data = np.empty((ndim,ndata))
    #data = random.normal(key, (ndata,ndim))
    #data = np.random.rand(ndata*ndim).reshape((ndim,ndata))
    #data = []
    with open('/home/lordcocoro2004/ml-latest-small/ratings.csv', newline='') as File:  
        reader = csv.reader(File)
        for row in reader:
            data[int(row[1])-1][int(row[0])-1]=float(row[2])
    names = []
    # with open('/home/lordcocoro2004/maestria/Recuperaciondelainfo/ml-latest-small/movie_rating.csv', newline='') as File:  
    #     reader = csv.reader(File)
    #     for row in reader:
    #         names = row[1:]
    #         break
    #     for row in reader:
    #         data.append([0 if value=='' else int(value) for value in row[1:]])
    
    # nor = normalizar(data)
    # mat = get_matrix(data)
    # res = coseno_2_knn_search(mat,nor)
    # print(res)
    # regular_data=de_normalizar(data,res)
    # regular_data = np.array(regular_data)

    # print(regular_data[0])
    #print(names[0])
    list_knn = knn_search_by_point(data.transpose(),data[0], K,p=1,d=1)
    for k in list_knn[0]:
       print(k[0],'____',k[1])
    #print(regular_data)
    
    #print(names[0])
    # list_knn_ = knn_search_by_point(regular_data.transpose(),regular_data[0], K,p=2,d=0)
    # for k in list_knn_[0]:
    #    print(k[0],'____',k[1])


if __name__ == '__main__':
    t0 = process_time()
    test()
    t1 = process_time()
    print ("Elapsed time seconds:", t1-t0)