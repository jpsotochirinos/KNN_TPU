from asyncio.windows_events import NULL
import math
import numpy as np

k = 3
c = [0,1,2,3,4,5]
class Node:
    def __init__(self,point,axis):
        self.point = point
        self.axis = axis
        self.left = None
        self.right = None
        self.visited = False

def getHeight(node):
    if(node is None):
        return 0
    return max(1 + getHeight(node.left), 1 + getHeight(node.right))

def generate_dot(node):
    if(node is not None):
        if(node.left is not None):
            print("\"" + node.point + "\"" + "->" + "\"" + node.left.point + "\"")
        if(node.right is not None):
            print("\"" + node.point + "\"" + "->" + "\"" + node.right.point + "\"")
        generate_dot(node.left)
        generate_dot(node.right)
    
def distanceMinkowski(point1, point2,p):
    distance = 0
    for i in range(k):
        distance = distance + np.power(np.absolute(point1[i] - point2[i]),p)
        if(p is 0):
            return np.power(distance,p)
        else:
            return np.power(distance,1/p)