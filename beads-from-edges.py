# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:41:53 2024

@author: Natalie.Buchanan
"""

import numpy as np
import random
import os
import matplotlib.pyplot as plt


def load_files(name):
    name: str
    network = "auelp"
    core_x = np.loadtxt(os.path.join(os.getcwd(), "Data",
                        network, name + '-core_x.dat'))
    core_y = np.loadtxt(os.path.join(os.getcwd(), "Data",
                        network, name + '-core_y.dat'))
    edges = np.loadtxt(os.path.join(os.getcwd(), "Data",
                       network, name + '-conn_core_edges.dat'))
    return [core_x, core_y, edges]


def constrainedWalk(n=15, start=(0, 0), end=(5, 2)):
    x0, xtarg = start[0], end[0]
    y0, ytarg = start[1], end[1]
    x = x0
    y = y0
    res = np.ones(n+1)*x0
    res[-1] = xtarg
    resY = np.ones(n+1)*y0
    resY[-1] = ytarg

    i = 0; crycounter = 0
    while i <= n/2 - 1:
        thetaX = 0.5*(1-((xtarg-x)/(n-(i*2))))
        thetaY = 0.5*(1-((ytarg-y)/(n-(i*2))))
        if random.random() <= thetaX:
            x2 = x-1
        else:
            x2 = x+1
        if random.random() <= thetaY:
            y2 = y - 1
        else:
            y2 = y + 1
        checkx = np.where(res == x2)[0]
        checky = np.where(resY == y2)[0]
        if np.isin(checkx, checky).any() == True:
            if random.random() > 0.5:
                x3 = x
                y3 = y2
                flag ='x'
            else:
                y3 = y
                x3 = x2
                flag='y'
            checkx = np.where(res == x3)[0]
            checky = np.where(resY == y3)[0]
            if np.isin(checkx, checky).any() == True:
                if flag=='x':
                    y3 = y
                    x3 = x2
                else:
                    y3 = y2
                    x3 = x
                checkx = np.where(res == x3)[0]
                checky = np.where(resY == y3)[0]
                if np.isin(checkx, checky).any()==True:
                    print("Cry", x3, y3, i+1)
                    crycounter +=1
            x2 = x3; y2 = y3
        x = x2
        y = y2
        res[i+1] = x
        resY[i+1] = y

        thetaX = 0.5*(1-((x-xtarg)/(n-(i*2)-1)))
        thetaY = 0.5*(1-((y-ytarg)/(n-(i*2)-1)))
        if random.random() <= thetaX:
            xtarg2 = xtarg-1
        else:
            xtarg2 = xtarg+1
        if random.random() <= thetaY:
            ytarg2 = ytarg - 1
        else:
            ytarg2 = ytarg + 1
        checkx = np.where(res == xtarg2)[0]
        checky = np.where(resY == ytarg2)[0]
        if np.isin(checkx, checky).any() == True:
            if random.random() > 0.5:
                xtarg3 = xtarg
                ytarg3 = ytarg2
                flag = 'x'
            else:
                ytarg3 = ytarg
                xtarg3 = xtarg2
                flag = 'y'
            checkx = np.where(res == xtarg3)[0]
            checky = np.where(resY == ytarg3)[0]
            if np.isin(checkx, checky).any() == True:
                if flag == 'x':
                    ytarg3 = ytarg
                    xtarg3 = xtarg2
                else:
                    xtarg3 = xtarg
                    ytarg3 = ytarg2
                checkx = np.where(res == xtarg3)[0]
                checky = np.where(resY == ytarg3)[0]
                if np.isin(checkx, checky).any() == True:
                    print("Cry", xtarg3, ytarg3, n-i-1)
                    crycounter += 1
            xtarg2 = xtarg3
            ytarg2 = ytarg3
        xtarg = xtarg2
        ytarg = ytarg2
        res[n-i-1] = xtarg
        resY[n-i-1] = ytarg

        i += 1
    print("Cry counter:" + str(crycounter))
    return [res, resY]


studyName = '20241016A1C0'
[NodeX, NodeY, Edges] = load_files(studyName)


point0 = (NodeX[int(Edges[0][0])], NodeY[int(Edges[0][0])])
pointN = (NodeX[int(Edges[0][1])], NodeY[int(Edges[0][1])])
[pathX, pathY] = constrainedWalk(n=51, start=point0, end=pointN)
plt.scatter(pathX, pathY, c=range(len(pathX)), cmap='rainbow')
plt.plot(pathX, pathY, 'k:')
plt.scatter(point0[0], point0[1], color='k')
plt.scatter(pointN[0], pointN[1], color='k')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Path")
plt.gca().set_aspect('equal')
plt.show()
