#!python numbers=disable
#import psyco

from string import printable
import numpy

def kdtree( data, leafsize=10 ):
    """
    build a kd-tree for O(n log n) nearest neighbour search

    input:
        data:       2D ndarray, shape =(ndim,ndata), preferentially C order
        leafsize:   max. number of data points to leave in a leaf

    output:
        kd-tree:    list of tuples
    """

    ndim = data.shape[0]
    ndata = data.shape[1]

    # find bounding hyper-rectangle
    hrect = numpy.zeros((2,data.shape[0]))
    hrect[0,:] = data.min(axis=1)
    hrect[1,:] = data.max(axis=1)

    # create root of kd-tree
    idx = numpy.argsort(data[0,:], kind='mergesort')
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
            idx = numpy.argsort(data[splitdim,:], kind='mergesort')
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
    data = numpy.resize(data,ldata.shape)
    sqd = ((ldata - data)**p).sum(axis=0) # data.reshape((param,1)).repeat(ndata, axis=1);
    idx = numpy.argsort(sqd, kind='mergesort')
    idx = idx[:K]
    return zip(sqd[idx], lidx[idx])

def minkowsky_knn_search(data, lidx, ldata, K,p):
    """ find K nearest neighbours of data among ldata """
    ndata = ldata.shape[1]
    param = ldata.shape[0]
    K = K if K < ndata else ndata
    retval = []
    data = numpy.resize(data,ldata.shape)
    sqd = ((ldata - data)**p).sum(axis=0) # data.reshape((param,1)).repeat(ndata, axis=1);
    idx = numpy.argsort(sqd, kind='mergesort')
    idx = idx[:K]
    return zip(sqd[idx], lidx[idx])

def pearson_knn_search(data, lidx, ldata, K,p):
    """ find K nearest neighbours of data among ldata """
    ndata = ldata.shape[1]
    param = ldata.shape[0]
    K = K if K < ndata else ndata
    retval = []
    data = numpy.resize(data,ldata.shape)
    sqd = ((numpy.multiply(ldata,data)).sum(axis=0) - (numpy.multiply((data).sum(axis=0),(ldata).sum(axis=0)))/ndata)/numpy.multiply(numpy.sqrt(numpy.multiply(data,data).sum(axis=0)-numpy.power((data).sum(axis=0),2)/ndata),(numpy.sqrt(numpy.multiply(ldata,ldata).sum(axis=0)-numpy.power((ldata).sum(axis=0),2)/ndata)))
    idx = numpy.argsort(sqd, kind='mergesort')[::-1]
    idx = idx[:K]
    return zip(sqd[idx], lidx[idx])

def coseno_knn_search(data, lidx, ldata, K,p):
    """ find K nearest neighbours of data among ldata """
    ndata = ldata.shape[1]
    param = ldata.shape[0]
    K = K if K < ndata else ndata
    retval = []
    data = numpy.resize(data,ldata.shape)
    sqd = numpy.divide((numpy.multiply(ldata,data).sum(axis=0)),(numpy.multiply(numpy.sqrt(numpy.multiply(data,data).sum(axis=0)),numpy.sqrt(numpy.multiply(ldata,ldata).sum(axis=0)))))
    idx = numpy.argsort(sqd, kind='mergesort')[::-1]
    idx = idx[:K]
    return zip(sqd[idx], lidx[idx])

def search_kdtree(tree, datapoint, K,p):
    """ find the k nearest neighbours of datapoint in a kdtree """
    stack = [tree[0]]
    knn = [(numpy.inf, None)]*K
    _datapt = datapoint[:,0]
    while stack:

        leaf_idx, leaf_data, left_hrect, \
                  right_hrect, left, right = stack.pop()

        # leaf
        if leaf_idx is not None:
            _knn = coseno_knn_search(datapoint, leaf_idx, leaf_data, K,p)
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
    return knn


def knn_search( data, K,p, leafsize=1028):

    """ find the K nearest neighbours for data points in data,
        using an O(n log n) kd-tree """

    ndata = data.shape[1]
    param = data.shape[0]

    # build kdtree
    tree = kdtree(data.copy(), leafsize=leafsize)

    # search kdtree
    knn = []
    for i in numpy.arange(ndata):
        _data = data[:,i].reshape((param,1)).repeat(leafsize, axis=1)
        _knn = search_kdtree(tree, _data, K+1,p)
        knn.append(_knn[1:])
    return knn
def knn_search_by_point( data,point, K,p):

    """ find the K nearest neighbours for data points in data,
        using an O(n log n) kd-tree """

    ndata = data.shape[1]
    param = data.shape[0]

    # build kdtree
    tree = kdtree(data.copy(), leafsize=ndata)

    # search kdtree
    knn = []
    
    _data = point.reshape((param,1)).repeat(ndata, axis=1)
    _knn = search_kdtree(tree, _data, K+1,p)
    knn.append(_knn[1:])
    return knn

# def radius_search(tree, datapoint, radius):
#     """ find all points within radius of datapoint """
#     stack = [tree[0]]
#     inside = []
#     while stack:

#         leaf_idx, leaf_data, left_hrect, \
#                   right_hrect, left, right = stack.pop()

#         # leaf
#         if leaf_idx is not None:
#             param=leaf_data.shape[0]
#             distance = numpy.sqrt(((leaf_data - datapoint.reshape((param,1)))**2).sum(axis=0))
#             near = numpy.where(distance<=radius)
#             if len(near[0]):
#                 idx = leaf_idx[near]
#                 distance = distance[near]
#                 inside += (zip(distance, idx))

#         else:

#             if intersect(left_hrect, radius, datapoint):
#                 stack.append(tree[left])

#             if intersect(right_hrect, radius, datapoint):
#                 stack.append(tree[right])

#     return inside


# def knn_search( data, K ,p):
#     """ find the K nearest neighbours for data points in data,
#         using O(n**2) search """
#     ndata = data.shape[1]
#     knn = []
#     idx = numpy.arange(ndata)
#     for i in numpy.arange(ndata):
#         _knn = coseno_knn_search(data[:,i], idx, data, K+1,p) # see above
#         aux = list(_knn)
#         knn.append(aux)
#     return knn

#!python numbers=disable

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


def __search_kdtree(tree, data,point, K, leafsize,p):
    knn = []
    param = data.shape[0]
    ndata = data.shape[1]
    #print(ndata)
    #for i in numpy.arange(ndata):
    #print(i)
    _data = point.reshape((param,1)).repeat(leafsize, axis=1)
    _knn = search_kdtree(tree, _data, K+1,p)
    #print(_knn[0])
    knn.append(_knn[1:])      
    return knn

def __remote_process(rank, qin, qout, tree,point, K, leafsize,p):
    while 1:
        # read input queue (block until data arrives)
        nc, data = qin.get()
        
        # process data
        knn = __search_kdtree(tree, data,point, K, leafsize,p)
        print(knn)
        numpy.save('knn.npy',knn)
        # write to output queue
        qout.put((nc,knn))

def knn_search_parallel(data,point, K,p, leafsize=1028):

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
                args=(rank, qin, qout, tree,point, K, leafsize,p))
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

def test():
    K = 6
    ndata = 193609
    ndim = 611
    data =  10 * numpy.random.rand(ndata*ndim).reshape((ndim,ndata) )
    point = numpy.random.rand(ndim)
    # data = numpy.empty((ndim,ndata))
    # with open('/home/lordcocoro2004/ml-latest-small/ratings.csv', newline='') as File:  
    #    reader = csv.reader(File)
    #    for row in reader:
    #        data[int(row[0])][int(row[1])]=float(row[2])
    #knn_search_parallel(data,point, K,p=1)
    list_knn = knn_search_by_point(data,point, K,p=1)
    print(list_knn)

if __name__ == '__main__':
    t0 = process_time()
    test()
    t1 = process_time()
    print ("Elapsed time seconds:", t1-t0)