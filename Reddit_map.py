# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 23:15:59 2020

@author: divisionbyzero1
"""

import sys
#import matplotlib.pyplot as plt
#import numpy as np
import pandas as pd

sys.path.append('C:\\Users\\Mike\\Documents\\code\\PythonWork\\TravellerMap\\starMapGenerator\\')

#from mapGen_class import starSystem
#from mapGen_class import mapProcess
import mapGen_class as MGC


b = MGC.mapProcess('Reddit_Example.png')
'''
To make these maps, I utilize GIMP and layers.  The layers are ful 100% white
and drawn by hand/filled/modified.  There's about 5 layers and then each layer
is set to 12.5% transparency.  Once it is flattened, it makes a good map.

Note: there is some bug and I'm not sure how to better handle the bit-depth.  You
need to do some "flatten" images and make sure it's grayscale, otherwise it will not be 
processed correctly.

'''
sectX = 0
sectY = 0
writeFile = 'testMap_seed12345.txt'

b.setMapSeed(seed=12345)
b.createStarMap()
b.showImage(filename='Reddit_baseImage')
b.showDensity(filename='Reddit_density')

b.showMap(filename='Reddit_starmap')
b.createSectorSets(default = False) #this step does most of the work and "rolls the most die"

b.fixLowTechAtmos()

#the following two stars are just examples of importing system data

KarfuData = ['1324', \
             'Karfu', \
             'A230867-E', \
             'N', \
             'De Na Ri Ht', \
             ' ', \
             '212', \
             'Redd', \
             'K2 V', \
             '{ 3 }', \
             '(A46+2)', \
             '[1716]', \
             'C', \
             '6']
SeasahnData = ['1227', \
               'Seasahn', \
               'C687663-9', \
               ' ', \
               'Ag Ga Ri', \
               ' ', \
               '813', \
               'Redd', \
               'G1 V K7 V', \
               '{ 1 }', \
               '(A46+2)', \
               '[1716]', \
               'B', \
               '5']

addList = [KarfuData, SeasahnData]
for i in range(len(addList)):
    b.insertStar(sectX, sectY, addList[i])

'''
The createEmpireStage method was created to simulate colonization from a central
world, including those colonists conducting some level of terraforming.
'''

b.createEmpireStage(sectX, sectY, '1324', terraformLevel=3, jump=1, \
                    maxJumps=4, empireCode='Redd', terraformLimit=30, \
                    habLimit = 30, \
                    minPop = 7, \
                    maxTech=14)
b.createEmpireStage(sectX, sectY, '1227', terraformLevel=2, jump=2, \
                    maxJumps=4, empireCode='Redd', terraformLimit=30, \
                    habLimit = 30, \
                    minPop = 6, \
                    maxTech=14)

'''
The following import and renaming method is just one example of how one can do this.
The filterEmpire method does extract all the stars belonging to a given empire
and then you can use that list to rename or modify as appropriate.
'''

sysList = b.filterEmpire(sectX, sectY, 'Redd')
Redd_names = pd.read_csv('Redd_starNames.txt', names=['Name'])
for i in range( min([Redd_names.Name.count(), len(sysList)])):
    b.setSystemName(sectX, sectY, sysList[i], Redd_names.Name[i])

'''
writeSectorData method should create a text file that is ready for putting into
the Traveller Poster Maker.  You'll have to supply the metadata portion, though.
'''
b.writeSectorData(sectX, sectY, filename='Reddit_sector.txt')