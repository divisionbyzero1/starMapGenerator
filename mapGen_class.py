# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 22:10:08 2020

@author: divisionbyzero1
Traveller map generator
This class will generate a star system following the Traveller rules
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class starSystem():
    '''
    This class describes a star system created with die rolls
    for Traveller.
    The systems generated are based on the GURPS Traveller rules found in "First In".
    '''
    randSeed = 123456
    import numpy as np

    import pandas as pd
    import os

    LOCALPATH = 'C:\\Users\\Mike\\Documents\\code\\PythonWork\\TravellerMap\\starMapGenerator\\'
    #return the local directory where this file is located

    numStars = 1
    starType = []
    companionOrbits = []
    
    stellarChars = [] #temp, lum, mass, radius, and age (unused)
    
    orbitZones = [] #inner, life, and snow radii for each star in system
    
    forbiddenZones = [] #if there is more than one star in system
    
    planetOrbitList = [] #provides the list of planetary orbit designations and distances
    planetTypes = [] #0 = gas giant, 1=belt, 2=terrestrial
       
    planetChars = [] #size, density, moons, eccentricity, period, tidal effects, rotation period, axial tilt table, 
    
    habScores = []
    systemPopulation = [] #population, government, law, and tech
    
    systemDataList = []
    
    def __init__(self):
        '''
        create the system
        '''
        
    def setSeed(self, seed=randSeed):
        '''
        sets the random seed to be used
        '''
        self.randSeed = seed
        self.np.random.seed(seed=self.randSeed)

    def createTotalSystem(self, \
                          habLimit = 40, \
                          empireCode = 'None', \
                          noNameChange = False):
        '''
        this enacts all the steps necessary to create a complete 
        system
        '''
        self.generateStars()
        self.populateCharacteristics()
        self.populateOrbitZones()
        self.processPlanetZones()
        self.fillPlanetOrbits()
        self.genPlanetCharacteristics()
        self.habScores = self.systemHabitabilityScore()
        self.populateSystem(habLimit = habLimit)
        self.createSystemSectorData(empireCode = empireCode, \
                                    noNameChange = noNameChange)

    def generateStars(self):
        '''
        passes through star generation steps 1-n...
        Generates and sorts the stars in the system
        '''
        self.starType = []
        self.companionOrbits = []
        
        #step 1: number of stars
        numStarRoll = self.Nd6Gen(numDie=3, size=1, dieSize=6)
        #print('numStarRoll Value: ', numStarRoll[0])
        if numStarRoll[0] <= 10:
            self.numStars = 1
        elif numStarRoll[0] <= 15:
            self.numStars = 2
        else:
            self.numStars = 3
            
        #print('number of Stars: ', self.numStars)
        #step 2 and 3: stellar types
        #find star types
        lumType = []
        spectralType = []
        spectralsubType = []
        
        luminosityOrder = ['I','III', 'V', 'VII']
        spectralOrder = ['O', 'B', 'A', 'F', 'G', 'K', 'M', 'D']
        
        for i in range( self.numStars ):
            lumClassRoll = self.Nd6Gen(numDie=3, size=1, dieSize=6)
            
            #print('lumClassRoll Value:', lumClassRoll[0])
            
            if lumClassRoll[0] <= 3:
                lumType.append(1)
            elif lumClassRoll[0] <= 14:
                lumType.append(2)
            else:
                lumType.append(3)
           
                            
                
                
            if not (lumType[-1] == 3):
                spectTypeRoll = self.Nd6Gen(numDie=3, size=1, dieSize=6)
                
                if spectTypeRoll[0] <= 4:
                    subRoll = self.Nd6Gen(numDie=1, size=1, dieSize=6)
                    if subRoll[0] == 1:
                        spectralType.append(1)
                    else:
                        spectralType.append(2)
                elif spectTypeRoll[0] <= 6:
                    spectralType.append(3)
                elif spectTypeRoll[0] <= 8:
                    spectralType.append(4)
                elif spectTypeRoll[0] <= 10:
                    spectralType.append(5)
                else:
                    spectralType.append(6)
        
                #now determine subType
                roll1 = self.Nd6Gen(numDie=1, size=1, dieSize=10)
                #if roll1[0] == 6:
                #    roll2 = self.Nd6Gen(numDie=1, size=1, dieSize=6)
                #    spectralsubType.append(roll1[0]+roll2[0]-2)
                #else:
                spectralsubType.append(roll1[0]-1)
                
            else:
                spectralType.append(7)
                spectralsubType.append(0)
            
                
        #sort by luminosity first
        isrt1 = self.np.argsort(lumType)
        lumType = self.np.asarray(lumType)[isrt1]
        spectralType = self.np.asarray(spectralType)[isrt1]
        spectralsubType = self.np.asarray(spectralsubType)[isrt1]
        
        #sort by spectral type
        for i in range( self.numStars ):
            sub = lumType == lumType[0]
            if len( lumType[sub] ) > 1:
                isrt2 = self.np.argsort( spectralType[sub] )
                spectralType[sub] = spectralType[sub][isrt2]
                spectralsubType[sub] = spectralsubType[sub][isrt2]
        
        #sort by spectral subtype
        for i in range( self.numStars ):
            sub1 = lumType == lumType[0]
            if len( lumType[sub1] ) > 1:
                sub2 = self.np.asarray( [ ((lumType[j] == lumType[i]) and \
                                           (spectralType[j] == spectralType[i])) for j in range( self.numStars)])
                if len( spectralsubType[sub2] ) > 1:
                    isrt3 = self.np.argsort( spectralsubType[sub2])
                    spectralsubType[sub2] = spectralsubType[sub2][isrt3]
    
        
        for i in range(self.numStars):
            starString = spectralOrder[spectralType[i]] + \
                        str(int(spectralsubType[i])).zfill(1) + \
                        ' ' + luminosityOrder[lumType[i]]
            self.starType.append(starString)
            
        #enter Step 4: companion orbit determination
        separationValues = [0.05, 0.5, 2, 10, 50]
        eccentricityValues = [0.05, 0.1, 0.2, 0.3, 0.4, \
                              0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        
        orbitRadius = []
        orbitEccen = []
        
        for i in range(1, self.numStars):
            multRoll = self.Nd6Gen(numDie=3, size=1, dieSize=6) + 6*(i-1)
            compRoll = self.Nd6Gen(numDie=2, size=1, dieSize=6)
            
            multIndex = 0
            #compIndex = 0
            
            eccenMod = 0
            
            if multRoll[0] <= 6:
                #very close
                multIndex = 0
                eccenMod = -6
            elif multRoll[0] <=9:
                #close
                multIndex = 1
                eccenMod = -4
            elif multRoll[0] <=11:
                #moderate
                multIndex = 2
                eccenMod = -2
            elif multRoll[0] <= 14:
                #wide
                multIndex = 3
            else:
                multIndex = 4
                
            orbitRadius.append( separationValues[multIndex]*compRoll[0])
            
            eccRoll = self.Nd6Gen(numDie=3, size=1, dieSize=6) + eccenMod
            
            eccValue = 0            
            if eccRoll[0] <= 3:
                eccValue = 0.05
            elif eccRoll[0] == 4:
                eccValue = 0.1
            elif eccRoll[0] == 5:
                eccValue = 0.2
            elif eccRoll[0] == 6:
                eccValue = 0.3
            elif eccRoll[0] <= 8:
                eccValue = 0.4
            elif eccRoll[0] <= 11:
                eccValue = 0.5
            elif eccRoll[0] <= 13:
                eccValue = 0.6
            elif eccRoll[0] <= 15:
                eccValue = 0.7
            elif eccRoll[0] == 16:
                eccValue = 0.8
            elif eccRoll[0] == 17:
                eccValue = 0.9
            elif eccRoll[0] >= 18:
                eccValue = 0.95
            else:
                eccValue = 0
                
            orbitEccen.append( eccValue )
                
            self.companionOrbits.append([orbitRadius[-1], orbitEccen[-1]])
            
    
    def minMaxOrbits(self, orbitR, orbitEcc):
        '''
        returns a tuple giving the minimum and maximum orbital separations
        for a given average separation and eccentricity
        '''
        
        minSep = (1 - orbitEcc)*orbitR
        maxSep = (1 + orbitEcc)*orbitR
        
        return (minSep, maxSep)
    
    def generateStellarChars(self, starString):
        '''
        processes the spectral and luminosity classes and provides surface tmep, 
        luminosity, mass, radius and lifespan values for each star
        uses starString which is something like K8 V or D0 VII to
        generate values
        
        returns: (Temperature, Luminosity, Mass, Radius, Age)
        '''
    
        #process starString first, then conduct table lookups
        classValue = starString[0]
        subTypeValue = int(starString[1])
        
        lumType = starString[3:]
        lumValue = 0
        if lumType == 'I':
            lumValue = 0
        elif lumType == 'III':
            lumValue = 1
        elif lumType == 'V':
            lumValue = 2
        else:
            lumValue = 3
        
        classSet = ['O', 'B', 'A', 'F', \
                    'G', 'K', 'M', 'D']
        sub = np.asarray([ classSet[i] == classValue for i in range(len(classSet))])
        
        #print(sub)
        classInt = np.arange(len(classSet))[sub][0]
        
        #print('classSet')
        #print('classInt Value: ', classInt)
        
        #import values from file
        filenames = ['classI_starChars.txt', \
                     'classIII_starChars.txt', \
                     'classV_starChars.txt']
        
        for i in range(len(filenames)):
            filenames[i] = self.LOCALPATH + filenames[i]
        
        #print(filenames)
        
        Temp = 0
        Lum = 0
        Mass = 0
        Radius = 0
        Age = 0
        Lifespan = 0
        
        if lumValue == 3:
            #then it is a white dwarf
            Temperature = 45000
            Mass = 0.14 + self.Nd6Gen(numDie=3, size=1, dieSize=6)[0]*0.072
            Lum = self.Nd6Gen(numDie=3, size=1, dieSize=6)[0]*0.01
            
            LumInterp = 0
            
            rmax = 0.002
            rmin = 0.0008
            Mmax = 1.44
            Mmin = 0.356
            Radius = (rmax-rmin)/(Mmin-Mmax)*(Mass - Mmin) + rmax
            
            Lifespan = 6
            deltaRange = 0.95
            rollMax = 36
            rollMin = 6
            deltaConst = deltaRange/(rollMax-rollMin)
            startDelta = 0.5 - (deltaRange/2 + deltaConst*rollMin)
            variance = startDelta + deltaConst*self.Nd6Gen(numDie=6, size=1, dieSize=6)[0]
            
            Age = Lifespan*variance
            
            
            #Age = 5
            
        else:
            #load the files and interpolate
            data = self.pd.read_csv(filenames[lumValue], \
                                    delim_whitespace=True)
            
            subtype = [ int(x[1]) for x in data.Type.values]
            data['subtype'] = subtype
            
            dataClass = [ data.Type.values[i][0] for i in range( data.Type.count() )]
            data['ClassType'] = dataClass
            
            #need to find spectral class location in the data table
            sub1 = data.Type.str.contains( classValue )
            sub2 = data.subtype[sub1].values > subTypeValue
            
            if self.np.any(sub2):
                #there is a value that exceeds the spectral subtype
                indarg =self.np.arange( data.Type.count() )[sub1]
                ind2 = indarg[sub2][0]
                ind1 = ind2 - 1
                
                subType1 = data.subtype.values[ind1]
                subType2 = data.subtype.values[ind2]
                #print('First Case')
                #if ind1 == self.np.arange( data.Type.count() )[sub1][0]:
                #    subType2 = subType2 + 10
                if dataClass[ind1] == dataClass[ind2]:
                    pass
                else:
                    subType2 = subType2 + 10
                
                
            else:
                #there isn't a value, and the next spectral class should be used
                indarg = self.np.arange( data.Type.count() )[sub1]
                ind2 = indarg[-1] + 1
                ind1 = ind2 - 1
                
                if ind2 == data.Type.count():
                    ind2 = data.Type.count()-1
                    ind1 = ind2 - 1
                else:
                    pass
                    
                subType1 = data.subtype.values[ind1]
                subType2 = data.subtype.values[ind2]+10
                    #print('Second Case')
                
            #print('ind2 = ', ind2)
            #print('ind1 = ', ind1)
            #print('subtype1 = ', subType1)
            #print('subtype2 = ', subType2)
            
            #print('ind1 star type found: ', data.Type[ind1])
            #print('ind2 star type found: ', data.Type[ind2])
            
            #interpolate values
            Temperature = (data.Temperature.values[ind2]-data.Temperature.values[ind1])/(subType2-subType1) * \
                            (subTypeValue - subType1) + data.Temperature.values[ind1]
                            
            Mass = (data.Mass.values[ind2]-data.Mass.values[ind1])/(subType2 - subType1) * \
                            (subTypeValue - subType1) + data.Mass.values[ind1]
            
            Radius = (data.Radius.values[ind2]-data.Radius.values[ind1])/(subType2-subType1) * \
                            (subTypeValue - subType1) + data.Radius.values[ind1]
            
            LumInterp = (data.Luminosity.values[ind2] - data.Luminosity.values[ind1]) / (subType2-subType1) * \
                            (subTypeValue - subType1) + data.Luminosity.values[ind1]
            Lum = LumInterp * (0.79 + 0.02*self.Nd6Gen(numDie=3, size=1, dieSize=6)[0])
            
            Lifespan = (data.Lifespan.values[ind2] - data.Lifespan.values[ind1]) / (subType2 - subType1) * \
                            (subTypeValue - subType1) + data.Lifespan.values[ind1]
            
            Lifespan = 6
            deltaRange = 0.95
            rollMax = 36
            rollMin = 6
            deltaConst = deltaRange/(rollMax-rollMin)
            startDelta = 0.5 - (deltaRange/2 + deltaConst*rollMin)
            variance = startDelta + deltaConst*self.Nd6Gen(numDie=6, size=1, dieSize=6)[0]
            
            Age = Lifespan*variance
                            
        #print('Found temperature: ', Temperature)
        #print('Found Mass: ', Mass)
        #print('Found Luminosity: ', LumInterp)
        #print('Luminosity Out: ', Lum)
        
        maxAge = 12
        Age = self.np.min([Age, maxAge])
        
        return [Temperature, Lum, Mass, Radius, Age]

    def populateCharacteristics(self):
        '''
        processes the list of stars and populates the array of values
        '''
        
        self.stellarChars = []
        
        for i in range( len(self.starType) ):
            self.stellarChars.append( self.generateStellarChars( self.starType[i] ) )

        #correct the age of the system to the youngest star
        minAge = 50
        for i in range( len(self.stellarChars)):
            if self.stellarChars[i][4] < minAge:
                minAge = self.stellarChars[i][4]
        
        for i in range( len(self.stellarChars)):
            self.stellarChars[i][4] = minAge
    
    def calculateOrbitZones(self, Temp, Lum, Mass, Rad):
        '''
        determines the orbital zones (inner, life, snow) for a star with
        given characteristics in the indicated tuple (Temp, Lum, Mass, Radius, Age)
        '''
        
        #self.orbitZones = []
    
        inner = max( 0.2*Mass, 0.008*self.np.sqrt(Lum))
    
        LifeInner = 0.95*self.np.sqrt(Lum)
        LifeOuter = 1.30*self.np.sqrt(Lum)
        
        SnowLine = 5.0*self.np.sqrt(Lum)
        
        OuterLimit = 40*Mass
        
        
        return (inner, LifeInner, LifeOuter, SnowLine, OuterLimit)
    
    def populateOrbitZones(self):
        '''  
        processes the lsit of stars and populates the array of values
        '''
        
        self.orbitZones = []
        self.forbiddenZones = []
        
        for i in range( len(self.starType)):
            self.orbitZones.append( self.calculateOrbitZones(self.stellarChars[i][0], \
                                                             self.stellarChars[i][1], \
                                                             self.stellarChars[i][2], \
                                                             self.stellarChars[i][3]))
        
        if len(self.starType) > 1:
            #there is more than one star in this system and we need forbidden zones
            for i in range(1, len(self.starType)):
                minSep, maxSep = self.minMaxOrbits(self.companionOrbits[i-1][0], \
                                                     self.companionOrbits[i-1][1])
                self.forbiddenZones.append([1./3*minSep, 3.*maxSep])
    
    def getBodeConst(self):
        '''
        returns the bode constant
        '''
        BodeRoll = self.Nd6Gen(dieSize=3)[0]
        BodeConst = 0.3
        if BodeRoll == 1:
            BodeConst = 0.3
        elif BodeRoll == 2:
            BodeConst = 0.35
        else:
            BodeConst = 0.4
        
        return BodeConst
    
    def genNumOrbits(self, innerLimit, outerLimit):
        '''
        determines 
        '''
        multValue = (self.Nd6Gen(numDie=1, size=1, dieSize=6)[0]+1)/2
        BaseRadius = innerLimit*multValue
        
        BodeConst = self.getBodeConst()
        numBodeMax = (outerLimit-BaseRadius)/BodeConst
        
        numOrbits = int(self.np.round( self.np.log(numBodeMax)/self.np.log(2) ) + 1)
        
        return BaseRadius, BodeConst, numOrbits
        
    
    def processPlanetZones(self):
        '''
        This function should be run after self.populateOrbitZones() so as to create the 
        regions for planet formation around single stars, or multiple stars in a given
        system.
        The orbit designations will indicate <star>-<orbit #s>, e.g. 1-1 indicates 
        the first planet orbit around the primary star.
        When orbits are to be located beyond one or multiple stars, then the orbit
        will be indicated 12 or 123
        '''
        
        self.planetOrbitList = []
        
        #process: generate orbits for all stars individually,
        #then if multi-star, determine which orbits are eliminated by forbidden
        #zones
        #then create multi-star orbits
        
        for i in range( len(self.starType) ):
            
            BaseRadius, BodeConst, numOrbits = self.genNumOrbits(self.orbitZones[i][0], \
                                                                 self.orbitZones[i][4])
            
            #generate list of planet orbits            
            
            for j in range(numOrbits):
                    
                if j<=1:
                    planetBaseRad = BaseRadius + j*BodeConst
                else:
                    planetBaseRad = BaseRadius + 2**(j-1)*BodeConst
                
                deltaRange = 0.10
                deltaConst = deltaRange/(18-3)
                startDelta = 1 - (deltaRange/2+deltaConst*3)
                variance = startDelta + deltaConst*self.Nd6Gen(numDie=3, size=1, dieSize=6)[0]                    
                
                #need to generate a designator
                starNum = str(i+1)
                orbitNum = str(j+1).zfill(2)
                
                orbitDesignation = starNum + '-' + orbitNum
                
                self.planetOrbitList.append([i+1, j+1, orbitDesignation, planetBaseRad*variance])
        
        #now, we've processed each star individually, can proceed to remove 
        #forbidden zone orbits
        if len(self.starType) > 1:
            for i in range(1, len(self.starType) ):
                innerLimit = self.forbiddenZones[i-1][0]
                #print('innerLimit line 510:', innerLimit, i)
                sub = [x[3] < innerLimit for x in self.planetOrbitList]
                #print('planetOrbitList before cull:', self.planetOrbitList)
                #print('sub list: ', sub)
                tempOrbitList = []
                for j in range( len(self.planetOrbitList) ):
                    if sub[j]:
                        tempOrbitList.append(self.planetOrbitList[j])
                self.planetOrbitList = tempOrbitList
        
                #have eliminated the orbits beyond the inner forbidden zone of companions
                #generate dual star orbit zones
                
                massSet = [x[2] for x in self.stellarChars[:i+1]]
                lumSet = [x[1] for x in self.stellarChars[:i+1]]
                
                effectiveMass = sum(massSet)
                effectiveLum = sum(lumSet)
                
                self.orbitZones.append(self.calculateOrbitZones(self.stellarChars[0][0], \
                                                                effectiveLum, \
                                                                effectiveMass, 1))
                
                BaseR, BodeC, numOrb = self.genNumOrbits(self.orbitZones[-1][0], \
                                                         self.orbitZones[-1][4])
                
                starNum = ''
                #print(i)
                #for j in range(i+1):
                #    starNum += str(j+1)
                starNum = str(len(self.starType) + i)
                
                
                #recall, forbidden zone ends at self.forbiddenZones[i-1][1]
                for j in range(numOrb):
                    if j<=1:
                        planetBaseRad = BaseR + j*BodeC
                    else:
                        planetBaseRad = BaseR + 2**(j-1)*BodeC
                    
                    deltaRange = 0.1
                    deltaConst = deltaRange/(18-3)
                    startDelta = 1 - (deltaRange/2 + deltaConst*3)
                    variance = startDelta + deltaConst*self.Nd6Gen(numDie=3, size=1, dieSize=6)[0]
                    
                    orbitNum = str(j+1).zfill(2)
                    orbitDesignation = starNum + '-' + orbitNum
                    
                    if planetBaseRad*variance < self.forbiddenZones[i-1][1]:
                        pass
                    else:
                        self.planetOrbitList.append([int(starNum), int(orbitNum), orbitDesignation, planetBaseRad*variance])
    
    def calcPlanetDensity(self, \
                          age, \
                          diameter, \
                          zoneType, \
                          gasGiant = False):
        '''
        this function obtains the planet density, gravity, and mass
        '''
        densityBase = 0
        minDensity = 0
        
        if gasGiant:
            if diameter < 40000:
                densityBase = 1.4
            elif diameter < 59999:
                densityBase = 1.0        
            elif diameter < 79999:
                densityBase = 0.7
            elif diameter < 84999:
                densityBase = 1.0                        
            else:
                densityBase = 1.4
        else:
            minDensity = 1.3
            if diameter < 3000:
                if zoneType < 3:
                    densityBase = 3.2
                else:
                    densityBase = 2.3
                        
            elif diameter < 5999:
                if zoneType < 3:
                    densityBase = 4.4
                else:
                    densityBase = 1.6
                        
            elif diameter < 8999:
                if zoneType < 3:
                    densityBase = 5.3
                else:
                    densityBase = 1.8
                        
            else:
                if zoneType < 3:
                    densityBase = 5.9
                else:
                    densityBase = 1.9
            
        #calculate densities
        densDM = 0
        
        densDM = -1* self.np.trunc(age/0.5)
            
        densRoll = self.Nd6Gen(numDie=3, size=1, dieSize=6)[0]+densDM
        densMod = densRoll/10
            
        density = max([densityBase + densMod, minDensity])
            
        #calculate mass and surface gravity
        Mass = 0
        Gravity = 0
        
        Mass = (density * (diameter/1000)**3)/2750
        Gravity = (62.9*Mass)/ (diameter/1000)**2
            
        return density, Gravity, Mass
            
    def fillPlanetOrbits(self):
        '''
        This function accomplishes planetary orbits
        '''
        
        #check for gas giants
        self.planetTypes = np.ones( len(self.planetOrbitList) )*2
        #0 = gas giant, 1=belt, 2=terrestrial
        
        for i in range( len(self.planetOrbitList) ):
            
            #need to check the zone type
            whichStar = self.planetOrbitList[i][0]
            #if whichStar == 12:
            thisOrbitZone = self.orbitZones[whichStar-1]            
            
            #determine zone
            #0=inner
            #1=life
            #2=middle
            #3=snow
            thisPlanetRadius = self.planetOrbitList[i][3]
            
            zoneType = 0
            if thisPlanetRadius < thisOrbitZone[1]:
                zoneType = 0
                gasLimit = 3
            elif thisPlanetRadius < thisOrbitZone[2]:
                zoneType = 1
                gasLimit = 4
            elif thisPlanetRadius < thisOrbitZone[3]:
                zoneType = 2
                gasLimit = 7
            else:
                zoneType = 3
                gasLimit = 14
                
            checkGiant = self.Nd6Gen(numDie=3, size=1, dieSize=6)[0]
            if checkGiant < gasLimit:
                #0 = gas giant, 1=belt, 2=terrestrial
                self.planetTypes[i] = 0
                
                
        #pass through to determine asteroids
        for i in range( len(self.planetOrbitList) ):
            if i < len(self.planetOrbitList)-1:
                if self.planetTypes[i+1] == 0:
                    roidLimit = 15
                    #print('setting roid limit case 0: ', roidLimit)
                elif not (self.planetOrbitList[i][0] == self.planetOrbitList[i+1][0]):
                    roidLimit = 12
                    #print('setting roid limit case 1: ', roidLimit)
                else:
                    roidLimit = 6
                    #print('setting roid limit case 2: ', roidLimit)
            else:
                roidLimit = 6
            
            checkRoids = self.Nd6Gen(numDie=3, size=1, dieSize=6)[0]
            if self.planetTypes[i] == 0:
                pass
            elif checkRoids < roidLimit:
                self.planetTypes[i] = 1

    def genWorldType(self, Mass, \
                     Diameter, \
                     zoneType, \
                     starType, \
                     moon = False, \
                     moonOrbit = 10):
        '''
        This function implements planet creation
        Takes as input:
            Mass in Earth Masses,
            Diameter in miles
            Zone as number 0=inner, 1=life, 2=middle, 3=snow
        
        and returns the 
        size parameter value
        size class ID
        size class string
        world type ID
        world type string
        '''
        sizeParameter = (7.93*float(Mass))/(float(Diameter)/1000.)
        sizeID = 0
        sizeString = 'Empty'
        
        worldID = 0
        worldString = 'Empty'
        
        if sizeParameter <= 0.13:
            sizeID = 0
            sizeString = 'Tiny'
        elif sizeParameter <= 0.24:
            sizeID = 1
            sizeString = 'Very Small'
        elif sizeParameter <= 0.38:
            sizeID = 2
            sizeString = 'Small'
        elif sizeParameter <= 1.73:
            sizeID = 3
            sizeString = 'Standard'
        else:
            sizeID = 4
            sizeString = 'Large'
        
           
        innerList = ['Rockball', \
                     'Rockball', \
                     'Desert', \
                     'Greenhouse', \
                     'Hostile(SG)', \
                     'Desert']
        LifeList = ['Rockball', \
                    'Rockball', \
                    'Desert', \
                    'Ocean', \
                    'Hostile(SG)']
        MiddleList = ['Rockball', \
                      'Rockball', \
                      'Desert', \
                      'Hostile(N)', \
                      'Hostile(SG)']
        OuterList = ['Icy Rockball', \
                     'Icy Rockball', \
                     'Hostile(A)', \
                     'Hostile(A)', \
                     'Hostile(SG)']
              
        #whichList = innerList
        if zoneType == 0:
            whichList = innerList
        elif zoneType == 1:
            whichList = LifeList
        elif zoneType == 2:
            whichList = MiddleList
        else:
            whichList = OuterList
        
        #check if a greenhouse can be converted to desert
        if (zoneType == 0) and (whichList[sizeID] == 'Greenhouse'):
            DM = 0
            if starType[0] == 'A':
                DM = 3
            elif starType[0] == 'F':
                DM = 1
            elif starType[0] == 'K':
                DM = -1
            elif starType[0] == 'M':
                DM = -3
            else:
                DM = 0
            convertRoll = self.Nd6Gen(numDie=3, \
                                      size=1, \
                                      dieSize=6)[0] + DM
            if convertRoll >= 15:
                sizeID = 5
        
        
        
        worldID = 10*zoneType + sizeID
        worldString = whichList[sizeID]
        
        moonParameter = 'Not Moon'
        
        if moon:
            moonParameter = 'Moon'
            if ((worldString == 'Rockball') or \
                (worldString == 'Icy Rockball')):
                if moonOrbit <= 10:
                    tideRoll = self.Nd6Gen(numDie=3, \
                                           size=1, \
                                           dieSize=6)[0]
                    if tideRoll < 10:
                        moonParameter = 'Frozen'
                    elif tideRoll <= 13:
                        moonParameter = 'Sub-surface Ocean'
                    else:
                        moonParameter = 'Sulfur Volcanoes'
            else:
                pass
            
        else:
            pass
                
               
        return [sizeParameter, \
                sizeID, \
                sizeString, \
                worldID, \
                worldString, \
                moonParameter]

    def genAtmosphere(self, \
                      worldType, \
                      Gravity):
        '''
        This Function takes the world-type and surface gravity
        and then calcualtes the atmospheric pressure
        '''
        Pressure = 0
        atmos_UWP = ''
        atmos_string = ''
        if worldType[0] == 'Tiny':
            Pressue = 0
            atmos_UWP = 0
            atmos_string = 'None'
        elif worldType[0] == 'Very Small':
            Pressure = 0.05
            
        else:
            PressRoll = self.Nd6Gen(numDie=3, size=1, dieSize=6)[0]
            Pressure = PressRoll*0.1*Gravity
            
        if Pressure <= 0.09:
            atmos_UWP = 1
            atmos_string = 'Trace'
        elif Pressure <= 0.42:
            atmos_UWP = 3
            atmos_string = 'Very Thin'
        elif Pressure <= 0.7:
            atmos_UWP = 5
            atmos_string = 'Thin'
        elif Pressure <= 1.49:
            atmos_UWP = 6
            atmos_string = 'Standard'
        elif Pressure <= 2.49:
            atmos_UWP = 8
            atmos_string = 'Dense'
        else:
            atmos_UWP = 13
            atmos_string = 'Very Dense'
        
        return [Pressure, atmos_UWP, atmos_string]
    
    def getHydrographics(self, starType, zoneType, worldTypeList, atmosphereList):
        '''
        generates the hydrographics of a planet
        
        '''
        #zonetype, 0=inner, 1=life, 2=middle, 3=snow
        #worldTypeList = [sizeParameter, \
        #                 sizeID, \
        #                 sizeString, \
        #                 worldID, \
        #                 worldString, \
        #                 moonParameter]
        #atmosphereList = [Pressure, \
        #                   atmos_UWP, \
        #                   atmos_string]
        check1 = (worldTypeList[1] >= 2)
        check2 = (zoneType >= 1)
        check3 = (atmosphereList[0] >= 0.1)
        
        hydrographics = 0
        hydro_UWP = 0
        anyHydro = False
        #hydro_string = 'No Hydrographics'
        
        if ((check1 and check2) and check3):
            #passes all checks and will have an ocean
            DM = 0
            if starType[0] == 'M':
                DM = 2
            elif starType[0] == 'K':
                DM = 1
            elif starType[0] == 'F':
                DM = -1
            elif starType[0] == 'A':
                DM = -2
            else:
                DM = 0
                
            if (worldTypeList[4] == 'Desert'):
                if zoneType == 1:
                    DM = DM-8
                elif zoneType == 2:
                    DM = DM-6
                else:
                    pass
            if worldTypeList[4][0:7] == 'Hostile':
                DM = DM - 2
            
            hydroRoll = self.Nd6Gen(numDie=2, size=1, dieSize=6)[0]-2 + DM
            
            variance = self.Nd6Gen(numDie=2, size=1, dieSize=6)[0]-2
            
            hydrographics = min([max([hydroRoll*10 + variance, 0]), 100])
            
            #determine UWP
            
            if hydrographics <= 5:
                hydro_UWP = 0
                #hydro_string = 'Desert World'
            elif hydrographics <= 15:
                hydro_UWP = 1
                #hydro_string = 'Dry World'
            elif hydrographics <= 25:
                hydro_UWP = 2
                #hydro_string = 'Few small seas'
            elif hydrographics <= 35:
                hydro_UWP = 3
            elif hydrographics <= 45:
                hydro_UWP = 4
            elif hydrographics <= 55:
                hydro_UWP = 5
            elif hydrographics <= 65:
                hydro_UWP = 6
            elif hydrographics <= 75:
                hydro_UWP = 7
            elif hydrographics <= 85:
                hydro_UWP = 8
            elif hydrographics <= 95:
                hydro_UWP = 9
            else:
                hydro_UWP = 10
                
            anyHydro = True
            
            #check if its a moon and Frozen or Sulfur
            #'Not Moon'
            #'Frozen'
            #'Sub-surface Ocean'
            #'Sulfur Volcanoes'
        if worldTypeList[5] == 'Not Moon':
            pass
        elif worldTypeList[5] == 'Moon':
            hydrographics = 0
            hydro_UWP = 0
            anyHydro = False
        elif worldTypeList[5] == 'Frozen':
            hydrographics = 0
            hydro_UWP = 0
            anyHydro = False
        elif worldTypeList[5] == 'Sub-surface Ocean':
            hydrographics = 100
            hydro_UWP = 10
            anyHydro = True
        else:
            hydrographics = 0
            hydro_UWP = 0
            anyHydro = False
        
        #print('worldType[5]: ', worldTypeList[5])
        #print('anyHydro value: ', anyHydro)
        return [hydrographics, hydro_UWP, anyHydro]

    def getEcosphere(self, Hydrographics, \
                     starAge, \
                     worldTypeList):
        '''
        Calculates the native ecosphere level
        '''
        DM = 0
        Ecosphere = 'No Life'
        newType = worldTypeList[4]
        Intelligence = 'Non-intelligent'
        
        lifeRoll = 0
        
        if ((worldTypeList[5] == 'Sub-surface Ocean') or \
            Hydrographics[2]):
            #it might have native life
            if worldTypeList[4] == 'Ocean':
                DM = DM + 2
            
            DM = DM + self.np.trunc(starAge/0.5)
            lifeRoll = self.Nd6Gen(numDie=2, \
                                   size=1, \
                                   dieSize=6)[0] + DM
            if lifeRoll <= 13:
                Ecosphere = 'No Life'
            elif lifeRoll <= 16:
                Ecosphere = 'Protozoa'
            elif lifeRoll <= 17:
                Ecosphere = 'Metazoa'
            elif lifeRoll <= 18:
                Ecosphere = 'Simple animals'
            else:
                Ecosphere = 'Complex animals'
                #add rule about intelligent life
                intRoll = self.Nd6Gen(numDie=3, size=1, dieSize=6)[0]
                if intRoll <= 15:
                    Intelligence = '<IQ 5 (Wolves)'
                elif intRoll <= 17:
                    Intelligence = 'Near-sentient (Chimps, hominids)'
                else:
                    Intelligence = 'Sentient!'
        
        
        
        if lifeRoll > 16:
            newType = 'Earthlike'
        else:
            newType = 'Hostile(N)'
        
        return [Ecosphere, newType, Intelligence]
    
    def getAtmosphereComposition(self, \
                                 EcosphereList, \
                                 worldTypeList):
        '''
        Determines the composition of the atmosphere by
        
        '''
        #EcosphereList = [Ecosphere, newType]
        #worldTypeList = [sizeParameter, \
        #                 sizeID, \
        #                 sizeString, \
        #                 worldID, \
        #                 worldString]
        
        atmosComp = 'Normal'
        atmosSubType = 'None'
        atmosContaminant = 'None'
        
        if ((worldTypeList[4] == 'Hostile(SG)') or \
            (worldTypeList[4] == 'Hostile(A)') ):
            atmosComp = 'Corrosive'
            atmosContaminant = 'None'
        
        if ((worldTypeList[4] == 'Hostile(N)') or \
            (worldTypeList[4] == 'Desert')):
            atmosRoll = self.Nd6Gen(numDie=3, size=1, dieSize=6)[0]
            if atmosRoll >= 13:
                atmosComp = 'Corrosive'
                atmosContaminant = 'None'
            else:
                atmosComp = 'Exotic'
                atmosContaminant = 'Nitrogen Compounds'
        
        if (worldTypeList[4] == 'Greenhouse'):
            atmosComp = 'Corrosive'
            atmosContaminant = 'Sulfur Compounds'
        
        if EcosphereList[1] == 'Earthlike':
            atmosComp = 'Oxygen-Nitrogen'
            atmosRoll = self.Nd6Gen(numDie=3, size=2, dieSize=6)
            if atmosRoll[0] >= 12:
                #tainted 
                atmosSubType = 'Tainted'
                if atmosRoll[1] <= 4:
                    atmosContaminant = 'Chlorine/Fluorine'
                elif atmosRoll[1] <= 6:
                    atmosContaminant = 'Sulfur Compounds'
                elif atmosRoll[1] <= 8:
                    atmosContaminant = 'Nitrogen Compounds'
                elif atmosRoll[1] <= 10:
                    atmosContaminant = 'Low Oxygen'
                elif atmosRoll[1] <= 12:
                    atmosContaminant = 'Pollutants'
                elif atmosRoll[1] <= 14:
                    atmosContaminant = 'High Carbon Dioxide'
                elif atmosRoll[1] <= 16:
                    atmosContaminant = 'High Oxygen'
                else:
                    atmosContaminant = 'Inert Gases'
            
        
        
        return [atmosComp, atmosSubType, atmosContaminant]
    
    def genPlanetClimate(self, \
                         whichStar, \
                         stellarCharList, \
                         atmosphereList, \
                         surfaceGravity, \
                         orbitRadius, \
                         hydrographicsList, \
                         worldType, \
                         EcosphereList):
        '''
        This function implements Overall Climate
        '''
        if whichStar >= len(stellarCharList):
            massSet = [x[2] for x in stellarCharList[:whichStar]]
            lumSet = [x[1] for x in stellarCharList[:whichStar]]
                
            effectiveMass = sum(massSet)
            effectiveLum = sum(lumSet)
        else:
            effectiveMass = stellarCharList[whichStar-1][2]
            effectiveLum = stellarCharList[whichStar-1][1]
        
        #determine albedo and greenhouse
        baseAlbedo = 1.
        baseGreenhouse = 0.0
        
        if worldType[4] == 'Hostile(SG)':
            baseAlbedo = 0.5
            baseGreenhouse = 0.2
        elif worldType[4] == 'Hostile(N)':
            baseAlbedo = 0.2
            baseGreenhouse = 0.2
        elif worldType[4] == 'Hostile(A)':
            baseAlbedo = 0.5
            baseGreenhouse = 0.2
        elif worldType[4] == 'Desert':
            baseAlbedo = 0.02
            baseGreenhouse = 0.15
        elif worldType[4] == 'Rockball':
            baseAlbedo = 0.02
            baseGreenhouse = 0
        elif worldType[4] == 'Icy Rockball':
            baseAlbedo = 0.45
            baseGreenhouse = 0
        else:
            baseAlbedo = 0.25
            baseGreenhouse = 0
            
        if ((EcosphereList[1] == 'Earthlike') or \
            (worldType[4] == 'Ocean')):
            baseGreenhouse = 0.15
            if hydrographicsList[0] < 30:
                baseAlbedo = 0.02
            elif hydrographicsList[0] <= 59:
                baseAlbedo = 0.10
            elif hydrographicsList[0] <= 89:
                baseAlbedo = 0.20
            else:
                baseAlbedo = 0.28
        
        albedoRoll = self.Nd6Gen(numDie=3, size=1, dieSize=6)[0]
        planetAlbedo = baseAlbedo + 0.01*albedoRoll
        
        #determine greenhouse factor
        greenhouseEffect = baseGreenhouse * \
                            (atmosphereList[0]/surfaceGravity)
        
        #determine surface temperature
        blackbodyTemp = 278 * effectiveLum**0.25 / self.np.sqrt(orbitRadius)
        
        surfaceTemp = blackbodyTemp * (1-planetAlbedo)**0.25 * \
                                    (1+greenhouseEffect)
        
        climate = 'Empty'
        if surfaceTemp <= 238:
            climate = 'Uninhabitable (Frigid)'
        elif surfaceTemp <= 249:
            climate = 'Frozen'
        elif surfaceTemp <= 260:
            climate = 'Very Cold'
        elif surfaceTemp <= 272:
            climate = 'Cold'
        elif surfaceTemp <= 283:
            climate = 'Chilly'
        elif surfaceTemp <= 294:
            climate = 'Cool'
        elif surfaceTemp <= 302:
            climate = 'Earth-normal'
        elif surfaceTemp <= 308:
            climate = 'Warm'
        elif surfaceTemp <= 313:
            climate = 'Tropical'    
        elif surfaceTemp <= 319:
            climate = 'Hot'
        elif surfaceTemp <= 324:
            climate = 'Very Hot'
        else:
            climate = 'Uninhabitable (Torrid)'
        
        return [ planetAlbedo, \
                 greenhouseEffect, \
                 surfaceTemp, \
                 climate]
    
    def getResourceValue(self, \
                         planetType):
        '''
        conducts resource value
        planetType = 0 for GasGiant, =1 for asteroids, 2 for terrestrial
        '''
        resourceRoll = self.Nd6Gen(numDie=3, size=1, dieSize=6)[0]
        
        Modifier = 0
        OverallValue = ''
        if planetType == 1:
            if resourceRoll == 3:
                OverallValue = 'Worthless'
                Modifier = -5
            elif resourceRoll == 4:
                OverallValue = 'Very Poor'
                Modifier = -4
            elif resourceRoll == 5:
                OverallValue = 'Poor'
                Modifier = -3
            elif resourceRoll <= 7:
                OverallValue = 'Very Scant'
                Modifier = -2
            elif resourceRoll <= 9:
                OverallValue = 'Scant'
                Modifier = -1
            elif resourceRoll <= 11:
                OverallValue = 'Average'
                Modifier = 0
            elif resourceRoll <= 13:
                OverallValue = 'Abundant'
                Modifier = 1
            elif resourceRoll <= 15:
                OverallValue = 'Very Abundant'
                Modifier = 2
            elif resourceRoll == 16:
                OverallValue = 'Rich'
                Modifier = 3
            elif resourceRoll == 17:
                OverallValue = 'Very Rich'
                Modifier = 4
            else:
                OverallValue = 'Motherlode'
                Modifier = 5
        else:
            if resourceRoll <= 4:
                OverallValue = 'Very Poor'
                Modifier = -2
            elif resourceRoll <= 6:
                OverallValue = 'Poor'
                Modifier = -1
            elif resourceRoll <= 14:
                OverallValue = 'Average'
                Modifier = 0
            elif resourceRoll <= 16:
                OverallValue = 'Rich'
                Modifier = 1
            else:
                OverallValue = 'Very Rich'
                Modifier = 2
            
        return [OverallValue, Modifier]
    
    def determineMSPR(self, \
                      worldSize, \
                      hydrographics, \
                      atmosphere, \
                      atmosComp, \
                      climate):
        '''
        The maximum sustainable population rating (MSPR) is calculated
        in step 19
        '''
        #atmosComp = [atmosComp, atmosSubType, atmosContaminant]
        MSPR = 9
        
        if ((atmosComp[0] == 'Corrosive') or \
            (atmosComp[0] == 'Exotic') or \
            (atmosphere[0] <= 0.42)):
            MSPR = 0
        else:
            DM = 0
            if worldSize <= 2000:
                DM = DM - 2
            elif worldSize <= 4000:
                DM = DM - 1
            else:
                DM = DM + 0
        
            if hydrographics[0] < 1:
                DM = DM - 2
            elif ((hydrographics[0] <= 30) or \
                  (hydrographics[0] >= 90)):
                DM = DM - 1
            else:
                DM = DM + 0
            
            if ((atmosphere[0] <= 0.7) or \
                (atmosphere[0] > 2.49)):
                DM = DM - 1
            else:
                DM = DM + 0
            
            if atmosComp[1] == 'Tainted':
                DM = DM - 1
            else:
                DM = DM + 0
                
            if ((climate[3] == 'Very Hot') or \
                (climate[3] == 'Very Cold') or \
                (climate[3] == 'Frozen')):
                DM = DM - 1
            elif ((climate[3] == 'Uninhabitable (Frigid)') or \
                  (climate[3] == 'Uninhabitable (Torrid)')):
                DM = DM - 2
                
            MSPR = 9 + DM
            
        return ['MSPR=', MSPR]
            
    
    def genPlanetCharacteristics(self):
        '''
        This function passes through the planets and determines the characteristics
        of the world
        '''
        #size, density, moons, eccentricity, period, tidal effects, 
        #rotation period, axial tilt table, world type, atmosphere press, 
        #hydrographics, native ecosphere, atmosphere composition, overall climate
        #resource value
        
        self.planetChars = []
        
        for i in range( len(self.planetOrbitList) ):
            #find the orbit zone
            whichStar = self.planetOrbitList[i][0]
        
            thisOrbitZone = self.orbitZones[whichStar-1]            
            #determine zone
            #0=inner
            #1=life
            #2=middle
            #3=snow
            thisPlanetRadius = self.planetOrbitList[i][3]
            
            zoneType = 0
            zoneString = ''
            if thisPlanetRadius < thisOrbitZone[1]:
                zoneType = 0
                zoneString = 'Inner'
            elif thisPlanetRadius < thisOrbitZone[2]:
                zoneType = 1
                zoneString = 'Life'
            elif thisPlanetRadius < thisOrbitZone[3]:
                zoneType = 2
                zoneString = 'Middle'
            else:
                zoneType = 3
                zoneString = 'Snow'
                
            #generate some modifiers
            DM = 0
            #innermost?
            if (self.planetOrbitList[i][1] == 1) and \
                (self.planetOrbitList[i][0] <= len(self.starType) ):
                DM = -4
            #inside snow line?
            if (zoneType < 3) and (self.planetOrbitList[i][1] > 1):
                DM = -2
            
            if thisPlanetRadius - thisOrbitZone[3] > 0:
                #the planet is past the snowline
                if thisPlanetRadius-thisOrbitZone[3] < 1:
                    DM = DM+6
                elif thisPlanetRadius - thisOrbitZone[3] < 5:
                    DM = DM+4
                
            
            if whichStar <= len(self.starType):
                
                starClass = self.starType[whichStar-1]
                if starClass[0] == 'M':
                    DM = DM-1
            
            sizeRoll = self.Nd6Gen(numDie=2, size=1, dieSize=6)[0] + DM
            
            planetMult = 0
            minSize = 0
            varMult = 0
            if self.planetTypes[i] == 0:
                planetMult = 5000 
                minSize = 25000
                varMult = 100
            elif self.planetTypes[i] == 2:
                planetMult = 1000
                minSize = 1000
                varMult = 500
            else:
                planetMult = 0
                minSize = 0
                varMult = 0
            
            varSize = varMult * (self.Nd6Gen(numDie=2, size=1, dieSize=6)[0]-7)
            
            planetSize = max([planetMult * sizeRoll + varSize, \
                              minSize])
            
            planetString = ''
            
            densityBase = 0
            if self.planetTypes[i] == 0:
                planetString = 'Gas Giant'
                
            elif self.planetTypes[i] == 1:
                planetString = 'Asteroid Belt'
                
            else:
                planetString = 'Terrestrial'
            
            if self.planetTypes[i] == 1:
                density = 0
                Gravity = 0
                Mass = 0
            elif self.planetTypes[i] == 2:
                density, Gravity, Mass = self.calcPlanetDensity(self.stellarChars[0][4], \
                                                                planetSize, \
                                                                zoneType, \
                                                                gasGiant = False)
            else:
                density, Gravity, Mass = self.calcPlanetDensity(self.stellarChars[0][4], \
                                                                planetSize, \
                                                                zoneType, \
                                                                gasGiant = True)
                    
            
            
            #Step 10, place moons
            if self.planetTypes[i] == 2:
                #terrestrial world
                #determine DM first
                moonDM = 0
                if thisPlanetRadius < 0.75:
                    moonDM = moonDM - 3
                elif thisPlanetRadius < 1.5:
                    moonDM = moonDM - 1
                else:
                    moonDM = moonDM + 0
                
                if planetSize < 2000:
                    moonDM = moonDM - 2
                elif planetSize < 3999:
                    moonDM = moonDM - 1
                elif planetSize > 9000:
                    moonDM = moonDM + 1
                else:
                    moonDM = moonDM + 0
                    
                largeMoonRoll = self.Nd6Gen(numDie=1, \
                                            size=1, \
                                            dieSize=6)[0] - \
                                            4 + moonDM
                largeMoons = False
                smallMoons = False
                numMoons = 0
                moonList = []
                
                if largeMoonRoll <= 0:
                    largeMoons = False
                else:
                    largeMoons = True
                    numMoons = int(largeMoonRoll)
                
                if not largeMoons:
                    smallMoonRoll = self.Nd6Gen(numDie=1, \
                                                size=1, \
                                                dieSize=6)[0] - \
                                                2 + moonDM
                    if smallMoonRoll <= 0:
                        smallMoons = False
                    else:
                        smallMoons = True
                        numMoons = int(smallMoonRoll)
                
                #append all the moon parameters
                #[moon number, moon orbit radius, moon size]
                #print('Planet Number: ', i+1)
                #print('NumMoons value: ', numMoons)
                if numMoons > 0:
                    for j in range(numMoons):
                
                        moonDM = 0
                        if planetSize < 2000:
                            moonDM = -5
                        elif planetSize < 3999:
                            moonDM = -4
                        elif planetSize < 5999:
                            moonDM = -3
                        elif planetSize < 7999:
                            moonDM = -2
                        elif planetSize < 9999:
                            moonDM = -1
                        else:
                            moonDM = 0
                
                        #for j in range(numMoons):
                        if largeMoons:
                            moonSizeRoll = self.Nd6Gen(numDie=2, \
                                                   size = 1, \
                                                   dieSize=6)[0] - \
                                                   7 + moonDM
                            if moonSizeRoll > 0:
                                moonDiameter = moonSizeRoll * 1000
                            else:
                                moonDiameter = self.Nd6Gen(numDie=1, \
                                                       size=1, \
                                                       dieSize=6)[0]*100
                        else:
                            moonSizeRoll = self.Nd6Gen(numDie=2, \
                                                   size=1, \
                                                   dieSize=6)[0] - \
                                                   5
                            if moonSizeRoll > 0:
                                moonDiameter = moonSizeRoll*10
                            else:
                                moonDiameter = self.Nd6Gen(numDie=2, \
                                                       size=1, \
                                                       dieSize=4)[0] + 1
                                    
                        moonDens, moonGrav, moonMass = \
                            self.calcPlanetDensity(self.stellarChars[0][4], \
                                                   moonDiameter, \
                                                   zoneType, \
                                                   gasGiant = False)
                        orbitDM = 0
                        if largeMoons:
                            if moonDiameter <= 499:
                                orbitDM = 0
                            elif moonDiameter <= 999:
                                orbitDM = 1
                            elif moonDiameter <= 1999:
                                orbitDM = 2
                            elif moonDiameter <= 3999:
                                orbitDM = 3
                            else:
                                orbitDM = 4
                            orbitRoll = (self.Nd6Gen(numDie=2, \
                                                    size=1, \
                                                    dieSize=6)[0] + \
                                                    orbitDM)*5
                            moonType = 'Large'
                        else:
                            orbitRoll = self.Nd6Gen(numDie=1, \
                                                    size=1, \
                                                    dieSize=6)[0]
                            moonType = 'Small'
                        
                        moonWorldType = ['Empty']
                        
                        moonList.append([j+1, \
                                         moonType, \
                                         orbitRoll, \
                                         moonDiameter*1.60934, \
                                         moonDens, \
                                         moonMass, \
                                         moonGrav, \
                                         moonWorldType])
                else:
                    moonList.append(['No Moons'])
            
            elif self.planetTypes[i] == 0:
                #gas giant
                #for now, only gen the large moons
                numMoons = int(self.Nd6Gen(numDie=1, size=1, dieSize=6)[0])
                #numMoons = 0
                moonList = []
                moonType = 'Large'
                for j in range(numMoons):
                    moonDM = 0
                    if planetSize < 2000:
                        moonDM = -5
                    elif planetSize < 3999:
                        moonDM = -4
                    elif planetSize < 5999:
                        moonDM = -3
                    elif planetSize < 7999:
                        moonDM = -2
                    elif planetSize < 9999:
                        moonDM = -1
                    elif planetSize < 40000:
                        moonDM = 0
                    elif planetSize < 59999:
                        moonDM = 1
                    elif planetSize < 70000:
                        moonDM = 2
                    else:
                        moonDM = 3
                    
                    if zoneType == 1:
                        #this is not in GURPS Traveller, I added it
                        moonDM = moonDM + 2
                
                    moonSizeRoll = self.Nd6Gen(numDie=2, \
                                               size = 1, \
                                               dieSize=6)[0] - \
                                               7 + moonDM
                    if moonSizeRoll > 0:
                        moonDiameter = moonSizeRoll * 1000
                    else:
                        moonDiameter = self.Nd6Gen(numDie=1, \
                                                   size=1, \
                                                   dieSize=6)[0]*100
                    moonDens, moonGrav, moonMass = \
                            self.calcPlanetDensity(self.stellarChars[0][4], \
                                                   moonDiameter, \
                                                   zoneType, \
                                                   gasGiant = False)
                    
                    orbitRoll = self.Nd6Gen(numDie=3, size=1, dieSize=6)[0]+3
                    if orbitRoll >= 15:
                        orbitRoll = orbitRoll + self.Nd6Gen(numDie=2, \
                                                            size=1, \
                                                            dieSize=6)[0]
                    
                    if whichStar <= len(self.starType):
                        starType = self.starType[whichStar-1]
                    else:
                        starType = self.starType[0]
            
                    #print('planetSize value: ', planetSize)
                    moonWorldType = self.genWorldType(moonMass, \
                                                      moonDiameter, \
                                                      zoneType, \
                                                      starType, \
                                                      moon = True, \
                                                      moonOrbit = orbitRoll)
                    Atmosphere = self.genAtmosphere(moonWorldType, \
                                                    moonGrav)
                    
                    Hydrographics = self.getHydrographics(starType, \
                                                          zoneType, \
                                                          moonWorldType, \
                                                          Atmosphere)
                    Ecosphere = self.getEcosphere(Hydrographics, \
                                                  self.stellarChars[0][4], \
                                                  moonWorldType)
                    atmosComposition = self.getAtmosphereComposition(Ecosphere,\
                                                                     moonWorldType)
                    moonClimate = self.genPlanetClimate(whichStar, \
                                                          self.stellarChars, \
                                                          Atmosphere, \
                                                          moonGrav, \
                                                          thisPlanetRadius, \
                                                          Hydrographics, \
                                                          moonWorldType, \
                                                          Ecosphere)
                    moonMSPR = self.determineMSPR(moonDiameter, \
                                                  Hydrographics, \
                                                  Atmosphere, \
                                                  atmosComposition, \
                                                  moonClimate)
                    
                    moonResource = self.getResourceValue(2)
                    
                    
                    moonList.append([j+1, \
                                     moonType, \
                                     orbitRoll, \
                                     moonDiameter*1.60934, \
                                     moonDens, \
                                     moonMass, \
                                     moonGrav, \
                                     moonWorldType, \
                                     Atmosphere, \
                                     Hydrographics, \
                                     Ecosphere, \
                                     atmosComposition, \
                                     moonClimate, \
                                     moonResource, \
                                     moonMSPR])
            else:
                #asteroid belt, no moons
                numMoons = 0
                moonList = []
            
            #now, generate dynamic effects step 11 
            #dynamic parameters
            
            #skip to step 12 for now
            #size parameter
            
            
            if whichStar <= len(self.starType):
                starType = self.starType[whichStar-1]
            else:
                starType = self.starType[0]
            
            #print('planetSize value: ', planetSize)
            if self.planetTypes[i] == 2:
                #then planet is a terrestrial
                worldType = self.genWorldType(Mass, \
                                              planetSize, \
                                              zoneType, \
                                              starType)
            else:
                worldType = ['Empty']
                
            #return [sizeParameter, \
            #        sizeID, \
            #        sizeString, \
            #        worldID, \
            #        worldString]
            
            #set atmospheric pressure, step 13
            if not worldType[0] == 'Empty':
                Atmosphere = self.genAtmosphere(worldType, \
                                                Gravity)
                Hydrographics = self.getHydrographics(starType, \
                                                      zoneType, \
                                                      worldType, \
                                                      Atmosphere)
                Ecosphere = self.getEcosphere(Hydrographics, \
                                              self.stellarChars[0][4], \
                                              worldType)
                atmosComposition = self.getAtmosphereComposition(Ecosphere, \
                                                                 worldType)
                
                planetClimate = self.genPlanetClimate(whichStar, \
                                                      self.stellarChars, \
                                                      Atmosphere, \
                                                      Gravity, \
                                                      thisPlanetRadius, \
                                                      Hydrographics, \
                                                      worldType, \
                                                      Ecosphere)
                
                planetMSPR = self.determineMSPR(planetSize, \
                                                Hydrographics, \
                                                Atmosphere, \
                                                atmosComposition, \
                                                planetClimate)
                
                
                
            else:
                Atmosphere = 'No Atmos'
                Hydrographics = 'No Hydro'
                Ecosphere = 'No Life'
                atmosComposition = 'No Atmosphere'
                planetClimate = 'No Climate'
                planetMSPR = ['MSPR=', 0]
            
            resources = self.getResourceValue(self.planetTypes[i])
            
            
            self.planetChars.append([self.planetOrbitList[i][2], \
                                     planetString, \
                                     zoneString, \
                                     planetSize*1.60934, \
                                     density, \
                                     Mass, \
                                     Gravity, \
                                     numMoons, \
                                     moonList, \
                                     worldType, \
                                     Atmosphere, \
                                     Hydrographics, \
                                     Ecosphere, \
                                     atmosComposition, \
                                     planetClimate, \
                                     resources, \
                                     planetMSPR])
    
    def systemHabitabilityScore(self, \
                                gravWant = 1.0, \
                                gravExponent = 3.0, \
                                atmosWant = 1.0, \
                                atmosExponent = 2.0):
        '''
        This generates a habitability score based on planet properties
        It also creates a "terraformability" score
        '''
        #most important benefit is MSPR
        #next is resources
        #next is gravity
        #next is 
        
        #MSPR
        #terraformability needs gravity close to 1
        #atmosphere near 1
        #lifezone, middle, or inner
        
        habitability = []
        planetList = []
        MSPRset = []
        planetNumber = []
        moonNumber = []
        resourceAvail = []
        terraformability = []
        zoneList = []
        
        
        habWeight = [10, 3, 6, 1, 1, 3]
        #MSPR, resources, gravity, atmos, composition, zone
        terraWeight = [0, 3, 10, 3, 0, 3]
        #moonData = []
        #planetData = []
        moonValues = [0, 0, 0, 0, 0, 0]
        planetValues = [0, 0, 0, 0, 0, 0]
        
        thisPlanetNumber = 0
        thisMoonNumber = 0
        
        for i in range(len(self.planetTypes)):
            thisPlanetNumber = i
            zone = self.planetChars[i][2]
            
            if zone == 'Life':
                zoneValue = 10
            elif zone == 'Inner':
                zoneValue = 1
            elif zone == 'Snow':
                zoneValue = 1
            else:
                zoneValue = 3
            
            if self.planetTypes[i] == 0:
                #gas giant, so loop through moons
                
                for j in range( self.planetChars[i][7]):
                    #moonData = self.planetChars[i][8][j]
                    thisMoonNumber = j
                    MSPR = self.planetChars[i][8][j][14][1]
                    gravity = self.planetChars[i][8][j][6]
                    resources = self.planetChars[i][8][j][13][1]
                    atmosPress = self.planetChars[i][8][j][8][0]
                    atmosComp = self.planetChars[i][8][j][10][1]
            
                    gravValue = (1 - np.abs(gravity - gravWant))**gravExponent
                    atmosValue = (1 - np.abs(atmosPress - atmosWant))**atmosExponent
                    compValue = 0
                    #zoneValue = 0
                    #if zone == 1:
                    #    zoneValue = 10
                    #elif zone == 2:
                    #    zoneValue = 3
                    #elif zone == 0:
                    #    zoneValue = 2
                    #else:
                    #    zoneValue = 1
                    
                    if atmosComp == 'Hostile(N)':
                        compValue = 3
                    elif atmosComp == 'Oxygen-Nitrogen':
                        compValue = 4
                    else:
                        compValue = 1
                    
                    moonValues = [MSPR, \
                                  resources, \
                                  gravValue, \
                                  atmosValue, \
                                  compValue, \
                                  zoneValue]
                    
                    habSet = [moonValues[i]*habWeight[i] for i in range(len(moonValues))]
                    habitability.append(sum(habSet))
                    
                    terraSet = [moonValues[i]*terraWeight[i] for i in range(len(moonValues))]
                    terraformability.append(sum(terraSet))
                    
                    planetList.append('Moon')
                    MSPRset.append(MSPR)
                    planetNumber.append(thisPlanetNumber)
                    moonNumber.append(thisMoonNumber)
                    resourceAvail.append(resources)
                    zoneList.append(zoneValue)
            
            elif self.planetTypes[i] == 1:
                #asteroid, mostly just resources
                resources = self.planetChars[i][15][1]
                habitability.append(resources * habWeight[1])
                terraformability.append(0)
                planetList.append('Asteroid')
                MSPRset.append(0)
                thisMoonNumber = -1
                planetNumber.append(thisPlanetNumber)
                moonNumber.append(thisMoonNumber)
                resourceAvail.append(resources)
                zoneList.append(zoneValue)
            else:
                
                #terrestrial
                #planetData = self.planetChars[i]
                thisMoonNumber = -1
                MSPR = self.planetChars[i][16][1]
                gravity = self.planetChars[i][6]
                resources = self.planetChars[i][15][1]
                atmosPress = self.planetChars[i][10][0]
                atmosComp = self.planetChars[i][12][1]
            
                gravValue = (1 - np.abs(gravity - gravWant))**gravExponent
                atmosValue = (1 - np.abs(atmosPress - atmosWant))**atmosExponent
                compValue = 0
                #zoneValue = 0
                #if zone == 1:
                #    zoneValue = 10
                #elif zone == 2:
                #    zoneValue = 3
                #elif zone == 0:
                #    zoneValue = 2
                #else:
                #    zoneValue = 1
                    
                if atmosComp == 'Hostile(N)':
                    compValue = 3
                elif atmosComp == 'Earthlike':
                    compValue = 4
                else:
                    compValue = 1
                
                planetValues = [MSPR, \
                                resources, \
                                gravValue, \
                                atmosValue, \
                                compValue, \
                                zoneValue]
                    
                habSet = [planetValues[i]*habWeight[i] for i in range(len(planetValues))]
                habitability.append(sum(habSet))
                
                terraSet = [planetValues[i]*terraWeight[i] for i in range(len(planetValues))]
                terraformability.append(sum(terraSet))
                
                planetList.append('Terrestrial')
                MSPRset.append(MSPR)
                planetNumber.append(thisPlanetNumber)
                moonNumber.append(thisMoonNumber)
                resourceAvail.append(resources)
                zoneList.append(zoneValue)
        return[habitability, terraformability, planetList, \
               MSPRset, planetNumber, moonNumber, resourceAvail] #, zoneList]
    
    def UWP_size(self, worldSize):
        '''
        converts km of a world size into a UWP number, returns string
        '''
        sizeSTR = ''
        if worldSize < 1000:
            sizeSTR = '0'
        elif worldSize < 1600:
            sizeSTR = '1'
        elif worldSize < 3200:
            sizeSTR = '2'
        elif worldSize < 4800:
            sizeSTR = '3'
        elif worldSize < 6400:
            sizeSTR = '4'
        elif worldSize < 8000:
            sizeSTR = '5'
        elif worldSize < 9600:
            sizeSTR = '6'
        elif worldSize < 11200:
            sizeSTR = '7'
        elif worldSize < 12800:
            sizeSTR = '8'
        elif worldSize < 14400:
            sizeSTR = '9'
        elif worldSize < 16000:
            sizeSTR = 'A'
        else:
            sizeSTR = 'B'
    
        return sizeSTR
    
    def UWP_atmos(self, atmosPress, atmosType1, atmosType2):
        '''
        converts the input data into a UWP data
        '''
        #atmosType1 example = ['No Life', 'Hostile(N)', 'Non-intelligent']
        #atmosType2 exampel = ['Normal', 'None', 'None']
        
        atmosSTR = ''
        
        exotic = False
        tainted = False
        corrosive = False
                
        if ((atmosType1[1] == 'Hostile(N)') or \
            (atmosType2[0] == 'Exotic')):
            exotic = True
            corrosive = False
            
        if atmosType2[0] == 'Oxygen-Nitrogen':
            exotic = False
            corrosive = False
        elif atmosType2[0] == 'Corrosive':
            exotic = False
            corrosive = True
            
        if atmosType2[1] == 'Tainted':
            tainted = True
            
        if exotic:
            atmosSTR = 'A'
        elif corrosive:
            atmosSTR = 'B'
        else:
            if atmosPress <= 0.001:
                atmosSTR = '0'
            elif atmosPress <= 0.09:
                atmosSTR = '1'
            elif atmosPress <= 0.42:
                if tainted:
                    atmosSTR = '2'
                else:
                    atmosSTR = '3'
            elif atmosPress <= 0.7:
                if tainted:
                    atmosSTR = '4'
                else:
                    atmosSTR = '5'
            elif atmosPress <= 1.49:
                if tainted:
                    atmosSTR = '7'
                else:
                    atmosSTR = '6'
            elif atmosPress <= 2.49:
                if tainted:
                    atmosSTR = '9'
                else:
                    atmosSTR = '8'
            else:
                atmosSTR = 'D'
        
        return atmosSTR
    
    def UWP_hydro(self, hydroList):
        '''
        Converts the hydrographics list to a UWP
        '''
        hydroSTR = ''
        if hydroList[1] <= 9:
            hydroSTR = str( int(hydroList[1]) )
        else:
            hydroSTR = 'A'
            
        return hydroSTR
    
    def convertEHEX(self, intValue):
        '''
        converts an integer value into extended hexadecimal
        returns a string
        '''
        eSTR = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        strValue = ''
        intValue = max([0, intValue])
        if intValue <= 9:
            strValue = str( int(intValue) )
        else:
            strValue = eSTR[ int(intValue) - 10]
            
        return strValue
    
    def convertFromEHEX(self, ehex):
        '''
        converts from ehex back to integer.  ehex is a string
        '''
        convString = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        sub = [ (x == ehex) for x in convString ]
        result = 0
        for i in range(len(convString)):
            if sub[i]:
                result = i
        
        return result
    
    
    def populateSystem(self, \
                       habLimit = 10, \
                       minPopulation = 0, \
                       minGovernment = 0, \
                       minLaw = 0):
        '''
        This function creates a population, law, government, and tech for a world
        the function find the best planet/moon and works on that based on the hab scores and MSPR
        returns the UWP
        '''
        #systemPopulation = [] #planet, moon, population, government, law, starport, and tech
        self.systemPopulation = []
        
        if len(self.planetTypes) == 0:
            self.systemPopulation = [-1, \
                                     -1, \
                                     0, \
                                     0, \
                                     0, \
                                     0, \
                                     0, \
                                     'X000000-0']
            return
        
        
        sub = self.np.asarray( [self.habScores[0][i] == max(self.habScores[0]) \
                                for i in range( len(self.habScores[0])) ])
        if not self.np.any(sub):
            print('sub has come up empty??')
            print('number planets: ', len(self.planetTypes))
            print('sub values: ', sub)
            print('max habScore: ', max(self.habScores[0]))
            print('hab scores: ', self.habScores)
        
        bestInd = self.np.arange(len(self.habScores[0]))[sub][0]
        
        bestMSPR = self.habScores[3][bestInd]
        population = 0
        government = 0
        law = 0
        tech = 0
        starport = 'X'
        
        UWP = ''
        if self.habScores[0][bestInd] < habLimit:
            population = 0
        else:
            #populate the system
            popRoll = self.Nd6Gen(numDie=2, size=1, dieSize=6)[0]-2
            func = max([0, 5-bestMSPR])
            population = max([minPopulation, popRoll - func + self.habScores[6][bestInd]])
            
        if population == 0:
            government = 0
            law = 0
        else:
            government = max([minGovernment, self.Nd6Gen(numDie=2, size=1, dieSize=6)[0]-7 + population])
            law = max([minLaw, self.Nd6Gen(numDie=2, size=1, dieSize=6)[0]-7 + government])
        
        
        
        #determine starport
        DM = 0
        if population == 0:
            DM = -10
        elif population <= 2:
            DM = -2
        elif population <= 4:
            DM = -1
        elif population <= 8:
            DM = 0
        elif population <= 10:
            DM = 1
        else: 
            DM = 2
        
        starportRoll = self.Nd6Gen(numDie=2, size=1, dieSize=6)[0] + DM
        if starportRoll <= 2:
            starPort = 'X'
        elif starportRoll <= 4:
            starport = 'E'
        elif starportRoll <= 6:
            starport = 'D'
        elif starportRoll <= 8:
            starport = 'C'
        elif starportRoll <= 10:
            starport = 'B'
        else:
            starport = 'A'
        
        #determine tech level
        
        #find the planet so I can query the moon or planet characteristics
        DM_starport = 0
        DM_size = 0
        DM_atmos = 0
        DM_hydro = 0
        DM_population = 0
        DM_government = 0
        
        if starport == 'A':
            DM_starport = 6
        elif starport == 'B':
            DM_starport = 4
        elif starport == 'C':
            DM_starport = 2
        elif starport == 'X':
            DM_starport = -4
        else:
            DM_starport = 0
        
        if ((population >= 1) and (population <= 5)):
            DM_population = 1
        elif population == 8:
            DM_population = 1
        elif population == 9:
            DM_population = 2
        elif population == 10:
            DM_population = 4
        else:
            DM_population = 0
        
        if ((government == 0) or (government == 5)):
            DM_government = 1
        elif government == 7:
            DM_government = 2
        elif ((government == 13) or (government == 14)):
            DM_govenment = -2
        else:
            DM_government = 0
        
        #find size, atmos, and hydro characteristics
        if self.habScores[2][bestInd] == 'Moon':
            #it's a moon!
            planetNum = self.habScores[4][bestInd]
            moonNum = self.habScores[5][bestInd]
            
            worldChars = self.planetChars[planetNum][8][moonNum]
            sizeNum = self.UWP_size(worldChars[3])
            
            atmosPress = worldChars[8][0]
            atmosType1 = worldChars[10]
            atmosType2 = worldChars[11]
            
            atmosNum = self.UWP_atmos(atmosPress, atmosType1, atmosType2)
            
            hydroNum = self.UWP_hydro(worldChars[9])
            
            
        elif self.habScores[2][bestInd] == 'Asteroid':
            #it's an asteroid!
            planetNum = self.habScores[4][bestInd]
            moonNum = -1
            worldChars = self.planetChars[planetNum]
            sizeNum = '0'
            atmosNum = '0'
            hydroNum = '0'
        
        else: 
            #it's a terrestrial
            planetNum = self.habScores[4][bestInd]
            moonNum = -1
            worldChars = self.planetChars[planetNum]
            sizeNum = self.UWP_size(worldChars[3])
            
            atmosPress = worldChars[10][0]
            atmosType1 = worldChars[12]
            atmosType2 = worldChars[13]
            
            atmosNum = self.UWP_atmos(atmosPress, atmosType1, atmosType2)
            
            hydroNum = self.UWP_hydro(worldChars[11])
        
        #determine the dice modifiers for each characteristics
        if ((sizeNum == '0') or (sizeNum == '1')):
            DM_size = 2
        elif ((sizeNum == '2') or (sizeNum == '3') or (sizeNum == '4')):
            DM_size = 1
        
        if atmosNum in '0123ABCDEF':
            DM_atmos = 1
        else:
            DM_atmos = 0
            
        if hydroNum in '09':
            DM_hydro = 1
        elif hydroNum == 'A':
            DM_hydro = 2
        else:
            DM_hydro = 0
        
        DM_sum = DM_starport + DM_size + DM_atmos + \
                DM_hydro + DM_population + DM_government
        
        #now, FINALLY, determine tech level
        if population > 0:
            tech = self.Nd6Gen(numDie=1, size=1, dieSize=6)[0] + DM_sum
        else:
            tech = 0
            
        techSTR = self.convertEHEX(tech)
        
        UWP = starport + \
                sizeNum + \
                atmosNum + \
                hydroNum + \
                self.convertEHEX(population) + \
                self.convertEHEX(government) + \
                self.convertEHEX(law) + \
                '-' + techSTR
        
        self.systemPopulation = [self.habScores[4][bestInd], \
                                 self.habScores[5][bestInd], \
                                 population , \
                                 government, \
                                 law, \
                                 starport, \
                                 tech, \
                                 UWP] 
        #planet, moon, population, government, law, starport, and tech
    
    def terraformWorld(self, \
                       levels=3, \
                       empireCode = 'None', \
                       terraformLimit = 20, \
                       habLimit=10, \
                       noHydro = False, \
                       minPop = 4, \
                       noNameChange = False, \
                       lastUWP = 'X000000-0'):
        '''
        applies some set of levels of improvement to atmosphere and hydrographics
        so as to improve the quality/habitability of the world
        can only be run AFTER populateSystem()
        needs to recalculate MSPR as well!!!
        '''
        #any level of terraforming will change the atmosphere to breathable
        #the number of levels will increase the hydrographics
        
        #create a test for the best terraformability and avoid trying to terraform
        #asteroids
        #steps: 1) reprocess habScores to find best terraformability planet
        #steps: 2) check that it is a terrestrial or moon
        #steps: 3) if either of those, check terraformLimit
        #steps: 4) if it meets the requirements, process and update
        
        #check terraform scores
        #terrIndex = self.np.arange(len(self.habScores[1]))[self.habScores[1]==max(self.habScores[1])][0]
        terrIndex = self.habScores[1].index( max(self.habScores[1]))
        planetInd = self.habScores[4][terrIndex]
        moonInd = self.habScores[5][terrIndex]
        worldType = self.habScores[2][terrIndex]
        terrScore = max(self.habScores[1])
        
        lastHydro = 10*self.convertFromEHEX(lastUWP[3])
        
        shouldTerraform = False
        
        if ( (not worldType == 'Asteroid') and \
            (terrScore >= terraformLimit )):
            shouldTerraform = True
        else:
            shouldTerraform = False
            #print('Cannot terraform world.')
            return
        
        #easist is to just change the UWP
        
        #find atmosPress
        atmosPress = 0
        
        taintSTR = ''
        if levels <= 2:
            taintSTR = 'Tainted'
        else:
            taintSTR = 'None'
        
        atmosMin = min([levels*0.2, 0.7])
        atmosList = []
        atmosType1 = ['Metazoa', 'Earthlike', 'Non-intelligent'] #eco list
        atmosType2 = ['Oxygen-Nitrogen', taintSTR, 'None']
        
        worldSize = 0
        worldClimate = []
        newClimateTag = 'Uninhabitable (Frigid)'
        MSPRChange = False
        
        if noHydro:
            hydroMax = -1 #this is a code to set the max hydro to the existing hydro
        else:
            hydroMax = 72
        if shouldTerraform:
            
            if worldType == 'Moon':
                #it's a moon
                atmosPress = self.planetChars[planetInd][8][moonInd][8][0]
            
                newPressure = max([atmosMin, atmosPress])
                self.planetChars[planetInd][8][moonInd][8][0] = newPressure
                self.planetChars[planetInd][8][moonInd][10] = atmosType1
                self.planetChars[planetInd][8][moonInd][11] = atmosType2
            
                hydroStart = max([self.planetChars[planetInd][8][moonInd][9][0], lastHydro])
                if noHydro:
                    hydroMax = min([hydroStart, 72])
                else:
                    hydroMax = max([72, lastHydro])
                maxHydro = min([hydroStart + 10*levels, hydroMax])
                hydroValue = int(self.np.trunc(maxHydro/10))
                hydroList = [maxHydro, hydroValue, True]
                self.planetChars[planetInd][8][moonInd][9] = hydroList
            
                atmosList = self.planetChars[planetInd][8][moonInd][8]
            
                worldClimate = self.planetChars[planetInd][8][moonInd][12]
                newClimateTag = 'Cold'
                if worldClimate[3] == 'Uninhabitable (Frigid)':
                    newClimateTag = 'Very Cold'
                elif worldClimate[3] == 'Uninhabitable (Torrid)':
                    newClimateTag = 'Very Hot'
                else:
                    newClimateTag = worldClimate[3]
            
                self.planetChars[planetInd][8][moonInd][12][3] = newClimateTag
                
                worldSize = self.planetChars[planetInd][8][moonInd][3]
                MSPRChange = True
            
            else:
                #it's a planet or asteroid
                if worldType == 'Asteroid Belt':
                    #I don't think any terraforming is possible
                    atmosPress = 0
                elif worldType == 'Gas Giant':
                    #you got here by mistake!
                    atmosPress = 0
                else:
                    #it better be terrestrial
                    atmosPress = self.planetChars[planetInd][10][0]
                    #if type(atmosPress):
                    #    print('worldType: ', worldType, ' atmosMin: ', atmosMin, ' atmosPress: ', atmosPress)
                    
                    newPressure = max([atmosMin, atmosPress])
                    self.planetChars[planetInd][10][0] = newPressure
                    self.planetChars[planetInd][12] = atmosType1
                    self.planetChars[planetInd][13] = atmosType2
        
                    hydroStart = max([self.planetChars[planetInd][11][0], lastHydro])
                    if noHydro:
                        hydroMax = min([hydroStart, 72])
                    else:
                        hydroMax = max([72, lastHydro])
                
                    maxHydro = min([hydroStart + 10*levels, hydroMax])
                    hydroValue = int(self.np.trunc(maxHydro/10))
                    hydroList = [maxHydro, hydroValue, True]
            
                    self.planetChars[planetInd][11] = hydroList
                
                    atmosList = self.planetChars[planetInd][10]
                
                    worldClimate = self.planetChars[planetInd][14]
                    worldSize = self.planetChars[planetInd][3]
                
                    newClimateTag = 'Cold'
                    if worldClimate[3] == 'Uninhabitable (Frigid)':
                        newClimateTag = 'Very Cold'
                    elif worldClimate[3] == 'Uninhabitable (Torrid)':
                        newClimateTag = 'Very Hot'
                    else:
                        newClimateTag = worldClimate[3]
            
                    self.planetChars[planetInd][14][3] = newClimateTag
            
                
                    MSPRChange = True
            #terraforming changes UWP to appropriate "regular" value for oxygen-nitrogen
        
            worldClimate[3] = newClimateTag
        
            if MSPRChange:
                newMSPR = self.determineMSPR(worldSize, \
                                             hydroList, \
                                             atmosList, \
                                             atmosType2, \
                                             worldClimate)
                if worldType == 'Moon':
                    self.planetChars[planetInd][8][moonInd][14] = newMSPR
                else:
                    self.planetChars[planetInd][16] = newMSPR

            existingPopCode = self.systemDataList[2][4]
            if existingPopCode in '0123456789':
                existingPop = int(existingPopCode)
            else:
                existingPop = 10

            self.habScores = self.systemHabitabilityScore()
            self.populateSystem(minPopulation=max([levels, existingPop, minPop]), \
                                habLimit = habLimit)
            self.createSystemSectorData(empireCode = empireCode, \
                                        noNameChange = noNameChange)

    def genStarString(self):
        '''
        This returns a string of the stars in the system
        '''
        starString = ''
        for i in range(len(self.starType)):
            starString = starString + self.starType[i] + ' '
        return starString
    
    def genPBGstring(self):
        '''
        This function returns a string for the PBG (population-belt-gas giants)
        string of the system
        '''
        PBGstring = ' '
        popRoll = int(self.Nd6Gen(numDie=1, size=1, dieSize=9)[0])
        numAst = len(self.planetTypes[self.planetTypes==1])
        numGas = len(self.planetTypes[self.planetTypes==0])
        PBGstring = str(popRoll)+str(numAst)+str(numGas)
        return PBGstring
    
    def genWorldString(self):
        '''
        This function takes the world data and creates an entry list for the number of 
        worlds in the system
        '''
        worldSTR = ' '
        numPlanets = len(self.planetTypes)
        if self.systemPopulation[1] >= 0:
            #then it's a moon!
            numPlanets = numPlanets + 1
        else:
            pass
        
        worldStr = str(numPlanets)
        return worldStr
    
    def genBaseString(self):
        '''
        generates random bases
        '''
        baseStr = ''
        starportType = self.systemPopulation[5]
        naval = False
        scout = False
        research = False
        TAS = False
        
        baseRoll = self.Nd6Gen(numDie=2, size=1, dieSize=6)[0]
        if starportType == 'A':
            if baseRoll >= 8:
                naval = True
                research = True
            
            if baseRoll >= 10:
                scout = True
            TAS = True
        elif starportType == 'B':
            if baseRoll >= 8:
                naval = True
                scout = True
            if baseRoll >= 10:
                research = True
            TAS = True
            
        elif starportType == 'C':
            if baseRoll >= 8:
                scout = True
            if baseRoll >= 10:
                research = True
                TAS = True
        elif starportType == 'D':
            if baseRoll >= 7:
                scout = True
        else:
            pass
        
        if naval:
            baseStr = baseStr + 'N'
        if scout:
            baseStr = baseStr + 'S'
        if research:
            baseStr = baseStr + 'V'
    
        return baseStr
    
    def travelString(self):
        '''
        generates the travel string (amber or red)
        '''
        travelStr = ' '
        popStr = self.systemPopulation[7][4]
        govStr = self.systemPopulation[7][5]
        lawStr = self.systemPopulation[7][6]
        
        if popStr == '0':
            pass
        elif ((govStr in '07A') and (lawStr in '09ABCDEFGHIJKL')):
            travelStr = 'A'
        else:
            pass
        
        return travelStr
    
    def genRemarkString(self):
        '''
        generates the remarks for the system
        '''
        remarkStr = ''
        
        UWP = self.systemPopulation[7]
        
        if self.systemPopulation[1] >= 0:
            #this is a moon!
            remarkStr = 'Sa '
        else:
            #that's no moon
            pass
        
        if ((UWP[2] in '456789') and \
            (UWP[3] in '45678') and \
            (UWP[4] in '567')):
            #ag system
            remarkStr = remarkStr + 'Ag '
            
        if ((UWP[1] == '0') and \
            (UWP[2] == '0') and \
            (UWP[3] == '0')):
            #asteroid world
            remarkStr = remarkStr + 'As '
        
        if ((UWP[4] == '0') and \
            (UWP[5] == '0') and \
            (UWP[6] == '0')):
            #barren world
            remarkStr = remarkStr + 'Ba '
        
        if ((UWP[2] in '23456789ABCDEFG') and \
            (UWP[3] == '0')):
            #desert world
            remarkStr = remarkStr + 'De '
            
        if ((UWP[2] in 'ABCDEFGH') and \
            (UWP[3] in '123456789A')):
            #Fluid oceans
            remarkStr = remarkStr + 'Fl '
        
        if ((UWP[1] in '678') and \
            (UWP[2] in '568') and \
            (UWP[3] in '567')):
            #ag system
            remarkStr = remarkStr + 'Ga '
        
        if UWP[4] in '9ABCDEFGHIJKL':
            #hi population system
            remarkStr = remarkStr + 'Hi '
        
        if UWP[8] in 'CDEFGHIJKLMNO':
            #high tech
            remarkStr = remarkStr + 'Ht '
            
        if ((UWP[2] in '01') and \
            (UWP[3] in '123456789A')):
            #ice-capped
            remarkStr = remarkStr + 'Ic '
        
        if ((UWP[2] in '012479') and \
            (UWP[4] in '9ABCDEFGHIJKL')):
            #Industrial
            remarkStr = remarkStr + 'In '
            
        if (UWP[4] in '0123'):
            #Lo population
            remarkStr = remarkStr + 'Lo '
        
        if (UWP[8] in '012345'):
            #Low tech
            remarkStr = remarkStr + 'Lt '
        
        if ((UWP[2] in '0123') and \
            (UWP[3] in '0123') and \
            (UWP[4] in '6789ABCDEFGHIJKL')):
            #non agricultural
            remarkStr = remarkStr + 'Na '
        
        if (UWP[4] in '0123456'):
            #non-industrial
            remarkStr = remarkStr + 'NI '
        
        if ((UWP[2] in '23456') and \
            (UWP[3] in '0123')):
            #poor
            remarkStr = remarkStr + 'Po '
        
        if ((UWP[2] in '68') and \
            (UWP[4] in '678') and \
            (UWP[5] in '456789')):
            #rich world
            remarkStr = remarkStr + 'Ri '
        
        if (UWP[2] == '0'):
            #vacuum world
            remarkStr = remarkStr + 'Va '
        
        if (UWP[3] == 'A'):
            #water world
            remarkStr = remarkStr + 'Wa '
        
        return remarkStr
    
    def genSystemName(self, empireCode = 'None'):
        '''
        creates a random name based on the mpire code (if available)
        if empire is "None", then a random name is made based on some pseudo catalog
        '''
        catalog = ['HiParc', 'ExCen', 'Kep', 'CassIR']
        if empireCode == 'None':
            catNum = int(self.Nd6Gen(numDie=1, size=1, dieSize = len(catalog))[0]-1)
            startStr = str(len(self.starType))+'-'
            midStr = self.starType[0][0:2]
            PBGStr = '.'+self.genPBGstring()[2]
            nameStr = startStr + catalog[catNum]+'-'+midStr+PBGStr
        else:
            nameStr = empireCode+'_name'
            
        return nameStr

    def genImportanceExtension(self, \
                               baseString, \
                               UWP = 'default', \
                               remarkString = 'default'):
        '''
        This function generates an importance string
        '''
        if UWP == 'default':
            UWP = self.systemPopulation[7]
        else:
            pass
        
        IxValue = 0
        starport = UWP[0]
        if starport in 'AB':
            IxValue = IxValue + 1
        elif starport in 'DEX':
            IxValue = IxValue - 1
        else:
            pass
        
        techValue = self.convertFromEHEX(UWP[-1])
        techG = self.convertFromEHEX('G')
        techA = self.convertFromEHEX('A')
        if techValue >= techG:
            IxValue = IxValue + 2
        elif techValue >= techA:
            IxValue = IxValue + 1
        elif techValue <= 8:
            IxValue = IxValue - 1
        else:
            pass
        
        if remarkString == 'default':
            remarkString = self.genRemarkString()
        else:
            pass
        
        checkStrings = ['Ag', 'Hi', 'In', 'Ri']
        for i in range(len(checkStrings)):
            if checkStrings[i] in remarkString:
                IxValue = IxValue + 1
        
        popValue = self.convertFromEHEX(UWP[4])
        if popValue <= 6:
            IxValue = IxValue - 1
        
        if (('N' in baseString) and ('S' in baseString)):
            IxValue = IxValue + 1
        
        signStr = ''
        if IxValue >= 0:
            signStr = '+'
        else:
            pass
        
        return '{ '+signStr+str(IxValue)+' }'
        
        
    def createSystemSectorData(self, \
                               empireCode = 'None', \
                               noNameChange = False):
        '''
        This function takes the world data and creates an entry list appropriate for 
        inclusion in the traveller map data
        '''
        
        if noNameChange:
            empireString = 'None'
            allegianceString = empireCode
        else:
            empireString = empireCode
            allegianceString = empireCode
        self.systemDataList = []
        dumHex = '0101'
        name = self.genSystemName(empireCode = empireString)
        baseString = self.genBaseString()
        travelZoneString = self.travelString()
        remarkString = self.genRemarkString()
        PBGstring = self.genPBGstring()
        #allegianceString =
        starString = self.genStarString()
        importString = self.genImportanceExtension(baseString)
        EconExString = '(A46+2)'
        CultExString = '[1716]'
        NobleString = '-'
        worldString = self.genWorldString()
        
        self.systemDataList.append(dumHex)
        self.systemDataList.append(name)
        self.systemDataList.append(self.systemPopulation[7])
        self.systemDataList.append(baseString)
        self.systemDataList.append(remarkString)
        self.systemDataList.append(travelZoneString)
        self.systemDataList.append(PBGstring)
        self.systemDataList.append(allegianceString)
        self.systemDataList.append(starString)
        self.systemDataList.append(importString)
        self.systemDataList.append(EconExString)
        self.systemDataList.append(CultExString)
        self.systemDataList.append(NobleString)
        self.systemDataList.append(worldString)
        
    
    def Nd6Gen(self, numDie = 1, size = 1, dieSize = 6):
        '''
        Generates a set of die rolls.
        Inputs:
            numDie = 1, how many die to roll together for a single roll.  2D6 has numDie=2
            size = 1, how many independent rolls to conduct
            dieSize = 6, type of die rolled, 6 = 6-sided die
        outputs:
            np.array([result1, result2, result3..., result{size}])
        '''
        rolls = self.np.zeros( (size, numDie) )
        for i in range( numDie ):
            rolls[:,i] = self.np.random.randint(1, dieSize+1, size=size)
    
        return self.np.sum(rolls, axis=1)

class mapProcess():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    SS = starSystem()
    
    randSeed = 1 #random seed used for generation
    
    image = [] #base image that is processed
    density = [] #processed density map showing rift-scattered-sparse, etc.
    imgSize = [] # size of the map, should be convenient size for sectors
    
    starMap = [] # resulting star map
    
    sectorSets = [] #output file giving star positions
    
    sectorStarData = [] #names, UWPs, remarks, Zone, PBG, Allegiance, Stars, Ix, Ex, Cx, Nobility, W
    sectorSeedList = [] #keeps the randSeed value for the system if regeneration is needed later
    sectorTerraformList = [] #keeps a list of what level a system has been terraformed to
    
    def __init__(self, imageFile):
        '''
        create the system
        '''
        #imgSize = []
        self.image = self.plt.imread(imageFile)
        self.imgSize.append(self.np.shape(self.image)[0])
        self.imgSize.append(self.np.shape(self.image)[1])

    def setMapSeed(self, seed=randSeed):
        '''
        sets the randome seed to be used
        '''
        self.randSeed = seed
        self.np.random.seed(seed = self.randSeed)

    def showDensity(self, filename='testImage'):
        self.plt.imshow(self.density)
        self.plt.savefig('densityImage_'+filename+'.png')
        

    def showMap(self, filename='testImage'):
        '''
        plots the density and generated starMap
        '''
        
        sectorSizeX = 40
        sectorSizeY = 32
        
        numX = int(self.imgSize[0]/sectorSizeX)
        numY = int(self.imgSize[1]/sectorSizeY)
        
             
        self.plt.imshow(self.starMap)
        
        for i in range(numX):
            self.plt.axhline(y = sectorSizeX*(i))
        for i in range(numY):
            self.plt.axvline(x = sectorSizeY*(i))
        
        self.plt.savefig('mapImage_'+filename+'.png')
        

    def showImage(self, filename='testImage'):
        '''
        plots the image file
        '''
        print('Image size: ', self.imgSize[0], 'x', self.imgSize[1])
        self.plt.imshow(self.image, cmap=self.cm.gray)
        self.plt.savefig('originImage_'+filename+'.png')

    def createStarMap(self):
        '''
        processes the image into a starmap
        '''
        inlineImage = self.np.reshape(self.image*256, (self.imgSize[0]*self.imgSize[1]))
        numEntry = len(inlineImage)
        
        regionList = self.np.zeros(numEntry)
        limits = [0, 50, 115, 150, 205, 230, 300]
        for i in range(1, len(limits)):
            sub = [(inlineImage[j] > limits[i-1] and inlineImage[j] <= limits[i]) for j in range(numEntry)]
            regionList[sub] = i-1
            
        self.density = self.np.reshape(regionList, (self.imgSize[0], self.imgSize[1]))
        
        rollList1 = self.SS.Nd6Gen(numDie=1, size=numEntry, dieSize=6) #1D6 rolls
        rollList2 = self.SS.Nd6Gen(numDie=2, size=numEntry, dieSize=6) #2D6 rolls, for rifts
        
        numType = len(limits)-1
        starList = self.np.zeros(numEntry)
        
        for i in range(numType):
            thisType = (regionList == i)
            if i == 0:
                #rift space
                starTest = (rollList2 >= 12)
                #print('Rift region, test = ', 12)
            else:
                starTest = (rollList1 >= 6 - i + 1)
                #print('other region, test = ', 6 - i +1)
                
            for j in range(numEntry):
                if thisType[j] and starTest[j]:
                    starList[j] = 1
                else:
                    pass
        self.starMap = self.np.reshape(starList, (self.imgSize[0], self.imgSize[1]))
    
    
    def createSectorSets(self, default = True, \
                         habLimit = 40):
        '''
        This passes through the map and generates starlists by sector location (XXYY)
        that can later be output
        Sectors are listed in X and then Y
        '''
        self.sectorSets = [] #this is a list of lists that will keep the sector coordinate straight
        #and then list the star systems within by sector coordinate pairs (XXYY)
        sectorSizeDn = 40
        sectorSizeRt = 32
        
        thisSector = self.np.zeros((sectorSizeDn, sectorSizeRt))
        
        numDn = int(self.imgSize[0]/sectorSizeDn)
        numRt = int(self.imgSize[1]/sectorSizeRt)
        
        sectorCoordDn = 0
        sectorCoordRt = 0
        starList = []       
        
        lastSeed = int(self.np.trunc( self.np.random.rand()*4294967294))
        
        for i in range(numDn):
            sectorCoordDn = -1*i
            for j in range(numRt):
                sectorCoordRt = j
                starList = []
                
                starDataSet = []
                thisSector = self.starMap[i*sectorSizeDn:(i+1)*sectorSizeDn, \
                                          j*sectorSizeRt:(j+1)*sectorSizeRt]
                thisSeedList = []
                thisTerraformList = []
                for i2 in range(sectorSizeDn):
                    Dnstr = str(i2+1).zfill(2)
                    for j2 in range(sectorSizeRt):
                        Rtstr = str(j2+1).zfill(2)
                        
                        if thisSector[i2, j2] == 1:
                            starCoord = Rtstr+Dnstr
                            starList.append( starCoord )
                            
                            if default:
                                #generate dummy data
                                #entries = ['Hex', \
                                #           'Name', \
                                #           'UWP', \
                                #           'Bases', \
                                #           'Remarks', \
                                #           'Zone', \
                                #           'PBG', \
                                #           'Allegiance', \
                                #           'Stars', \
                                #           '{Ix}', \
                                #           '{Ex}', \
                                #           '{Cx}', \
                                #           'Nobility', \
                                #           'W']
                                
                                planetData = ['No Name', \
                                              'B564500-B', \
                                              'N', \
                                              'Ag Ni Pi Da', \
                                              'R', \
                                              '503', \
                                              'Im', \
                                              'M0 V', \
                                              '{ 2 }', \
                                              '{A46+2}', \
                                              '[1716]', \
                                              'BcC', \
                                              '6']
                                thisSeedList.append(1)
                                thisTerraformList.append(0)
                            else:
                                #generate real data
                                #self.sectorSeedList.append(lastSeed)
                                thisSeedList.append(lastSeed)
                                thisTerraformList.append(0)
                                
                                self.SS.setSeed(seed = lastSeed)
                                
                                self.SS.createTotalSystem(habLimit=habLimit)
                                planetData = self.SS.systemDataList[1:]
                                
                            lastSeed = int(self.np.trunc( self.np.random.rand()*4294967294))
                            
                            starDataSet.append( planetData )
                            
                setData = [sectorCoordRt, sectorCoordDn, starList]
                self.sectorSets.append(setData)
                self.sectorStarData.append(starDataSet)
                self.sectorSeedList.append(thisSeedList)
                self.sectorTerraformList.append(thisTerraformList)
                #sectorStarData = [] #names, UWPs, Bases, remarks, Zone, PBG, Allegiance, Stars, Ix, Ex, Cx, Nobility, W

    def setSystemName(self, \
                      sectorX, \
                      sectorY, \
                      starHex, \
                      name):
        '''
        changes the name of a system to something new
        '''
        sub = [ (self.sectorSets[i][0] == sectorX) and (self.sectorSets[i][1]==sectorY) for i in range(len(self.sectorSets))]
        #linestring = ''
        if not self.np.any(sub):
            print('sector not found, no output')
            return 0
        else:
            #print('sector found')
            pass
        sectInd = self.np.arange(len(self.sectorSets))[sub][0]
        #print('Sector Index: ', sectInd)
        
        #find the star
        sub = [ (x == starHex) for x in self.sectorSets[sectInd][2]]
        starIndex = -1
        for i in range( len(self.sectorSets[sectInd][2]) ):
            if sub[i]:
                starIndex = i
        if starIndex == -1:
            print('Star not found!')
            return
        
        self.sectorStarData[sectInd][starIndex][0] = name

    def setSystemAllegiance(self, \
                            sectorX, \
                            sectorY, \
                            starHex, \
                            allegiance):
        '''
        takes the sectorXY and starHex and sets the allegiance to something else
        '''
        sub = [ (self.sectorSets[i][0] == sectorX) and \
               (self.sectorSets[i][1]==sectorY) for i in range(len(self.sectorSets))]
        #linestring = ''
        if not self.np.any(sub):
            print('sector not found, no output')
            return 0
        else:
            #print('sector found')
            pass
        sectInd = self.np.arange(len(self.sectorSets))[sub][0]
        #print('Sector Index: ', sectInd)
        
        #find the star
        sub = [ (x == starHex) for x in self.sectorSets[sectInd][2]]
        starIndex = -1
        for i in range( len(self.sectorSets[sectInd][2]) ):
            if sub[i]:
                starIndex = i
        if starIndex == -1:
            print('Star not found!')
            return
        
        self.sectorStarData[sectInd][starIndex][6] = allegiance

    

    def findJumpGroup(self, \
                      sectorX, \
                      sectorY, \
                      startHex, \
                      jump=1, \
                      maxJumps=6):
        '''
        creates a list of all worlds that can be linked by a jump
        engine of type jump with as many as maxJumps jumps
        '''
        #for this algorithm, it will start at the given location, and calculate distance
        #for stars in the sector.  All the stars within a given distance are
        #added to a large list for a given jump distance
        
        sysList = [] #this will be the list of found stars
        jumpNumber = [] #this is a matched list to sysList that keeps
        #track of which jump this is
        
        sub = [ (self.sectorSets[i][0] == sectorX) and (self.sectorSets[i][1]==sectorY) for i in range(len(self.sectorSets))]
        #linestring = ''
        if not self.np.any(sub):
            print('sector not found, no output')
            return 0, 0
        else:
            #print('sector found')
            pass
        sectInd = self.np.arange(len(self.sectorSets))[sub][0]
        #print('Sector Index: ', sectInd)
        
        starList = self.sectorSets[sectInd][2]
        if startHex in starList:
            #print('Start star found in sector.')
            sysList.append(startHex)
            jumpNumber.append(0)
        else:
            print('Start star not found in sector.')
            return 1, 1
        #print('sysList check: ', sysList)
        #print('jump number check: ', jumpNumber)
        distCheck = self.np.zeros(len(starList))
        
        for i in range(maxJumps):
            #up to the max number of jumps, find all stars within the range given
            test = [jumpNumber[j] == i for j in range(len(jumpNumber))]
            #print('test list: ', test)
            checkList = []
            for j in range(len(test)):
                if test[j]:
                    checkList.append(sysList[j])
            #checkList = [sysList[test[j]] for j in range(len(test))]
            if type(checkList) == str:
                checkList = [checkList]
            else:
                pass
            
            for j in range(len(checkList)):
                
                distCheck = self.np.asarray( [self.starDistance(checkList[j],starList[k]) \
                                              for k in range(len(starList))] )
            
                reachable = distCheck <= jump
                #print('check starList[reachable]: ')
                
                for k in range( len(distCheck)):
                    if reachable[k]:
                        #can make it
                        if starList[k] in sysList:
                            #already found this star
                            pass
                        else:
                            sysList.append(starList[k])
                            jumpNumber.append(i+1)
                    else:
                        pass
        
        return sysList, jumpNumber

    def filterSystemList(self, \
                       sectorX, \
                       sectorY, \
                       systemList, \
                       outfile='defaultFilter.txt'):
        '''
        finds the requisite sector and uses the input systemlist
        generated, for example, from the findJumpGroup function, 
        and creates an outfile of just the systems in the list
        '''
        sub = [ (self.sectorSets[i][0] == sectorX) and (self.sectorSets[i][1]==sectorY) for i in range(len(self.sectorSets))]
        #linestring = ''
        if not self.np.any(sub):
            print('sector not found, no output')
            return 0
        else:
            print('sector found')
        sectInd = self.np.arange(len(self.sectorSets))[sub][0]
        print('Sector Index: ', sectInd)
        
        tempX = 10000
        tempY = 10000
        
        tempStarList = []
        tempPlanetData = []
        
        for i in range( len(self.sectorSets[sectInd][2])):
            if self.sectorSets[sectInd][2][i] in systemList:
                tempStarList.append( self.sectorSets[sectInd][2][i] )
                tempPlanetData.append( self.sectorStarData[sectInd][i] )
        
        #create a dummy sector entry and write, then delete
        self.sectorSets.append( [tempX, tempY, tempStarList])
        self.sectorStarData.append( tempPlanetData )
        
        self.writeSectorData(tempX, tempY, filename=outfile)
        
        del self.sectorSets[-1]
        del self.sectorStarData[-1]
        
        print('Operation complete...')
        print('File ', outfile, ' written.')
        
        return tempStarList, tempPlanetData

    def filterEmpire(self, \
                     sectorX, sectorY, \
                     empireCode):
        '''
        This function returns a system list for all stars with a specific empireCode
        string
        '''
        sub = [ (self.sectorSets[i][0] == sectorX) and (self.sectorSets[i][1]==sectorY) for i in range(len(self.sectorSets))]
        
        if not self.np.any(sub):
            print('sector not found, no output')
            return 0
        else:
            #print('sector found')
            pass
        sectInd = self.np.arange(len(self.sectorSets))[sub][0]
        #print('Sector Index: ', sectInd)
        
        sysList = []
        
        for i in range( len(self.sectorSets[sectInd][2]) ):
            if self.sectorStarData[sectInd][i][6] == empireCode:
                sysList.append( self.sectorSets[sectInd][2][i])
                
        return sysList
    
    def createSophontMigration(self, \
                               sectorX, sectorY, \
                               startHex, \
                               jump = 1, \
                               maxJumps = 6, \
                               SophCode = 'Varg', \
                               minUWP = 'X550000-7', \
                               maxPopFraction = 40, \
                               popDecayRate = 30):
        '''
        This function creates a spreading sophont code according to the parameters
        entered into the function.
        Spreading out of the sophont race from a starting location, it will use the 
        jump-level to determine how much the population decays.  Note: this decay
        rate is a percentage decrease per jump.  Some additional variability will be added
        minUWP gives a minimum set of parameters that a sophont race requires before taking up residence
        on the best world in the system.
        
        '''
        sub = [ (self.sectorSets[i][0] == sectorX) and (self.sectorSets[i][1]==sectorY) for i in range(len(self.sectorSets))]
        
        if not self.np.any(sub):
            print('sector not found, no output')
            return 0
        else:
            #print('sector found')
            pass
        sectInd = self.np.arange(len(self.sectorSets))[sub][0]
        #print('Sector Index: ', sectInd)
        
        sysList, jumpNumber = self.findJumpGroup(sectorX, sectorY, startHex, jump=jump, maxJumps=maxJumps)
        if not type(sysList) == list:
            print('No stars for empire!')
            return
        
        oldRemarkString = ' '
        for i in range( len(self.sectorSets[sectInd][2]) ):
            if self.sectorSets[sectInd][2][i] in sysList:
                #should be considered for colonization
                #obtain the UWP and compare with minUWP
                thisUWP = self.sectorStarData[sectInd][i][1]
                oldRemarkString = self.sectorStarData[sectInd][i][3]
                
                '''
                if len(oldRemarkString) <= 1:
                    print('problem with oldRemarkString: ', oldRemarkString)
                    print('sector number: ', sectInd)
                    print('star index number: ', i)
                    print('star hex: ', self.sectorSets[sectInd][2][i])
                    print('the UWP: ', thisUWP)
                '''
                
                if oldRemarkString == '':
                    oldRemarkString = ' '
                
                if oldRemarkString[-1] == ' ':
                    pass
                else:
                    oldRemarkString = oldRemarkString + ' '
                
                inhabit = True
                for j in range( len(thisUWP) ):
                    if j == 0:
                        pass
                    elif j == 7:
                        pass
                    else:
                        currentValue = self.SS.convertFromEHEX(thisUWP[j])
                        minValue = self.SS.convertFromEHEX(minUWP[j])
                        if inhabit and (minValue <= currentValue):
                            pass
                        else:
                            inhabit = False
                if inhabit:
                    #find popFraction and apply according to jump number value
                    listIndex = sysList.index( self.sectorSets[sectInd][2][i] )
                    thisJump = jumpNumber[listIndex]
                    
                    thisMaxPop = (maxPopFraction + (self.SS.Nd6Gen(numDie=2, size=1, dieSize=6)[0]-7))/100
                    thisFraction = max([0, thisMaxPop * (1 - popDecayRate/100)**thisJump])                    
                                        
                    fracCode = str( int(self.np.trunc(thisFraction*10)))
                    #print('fracCode: ', fracCode)
                                        
                    if fracCode == '0':
                        newRemark = oldRemarkString
                    elif SophCode in oldRemarkString:
                        newRemark = oldRemarkString
                    else:
                        newRemark = oldRemarkString + SophCode + fracCode + ' '
                    
                    self.sectorStarData[sectInd][i][3] = newRemark
    
    
    def createEmpireStage(self, sectorX, sectorY, \
                          startHex, \
                          jump = 1, \
                          maxJumps = 6, \
                          terraformLevel = 3, \
                          terraformLimit = 20, \
                          habLimit = 10, \
                          empireCode = 'CF', \
                          noHydro=False, \
                          minTech = 6, \
                          maxTech = 17, \
                          minPop = 4, \
                          noNameChange=False):
        '''
        This function takes the starting hex and uses jump groups to specify a 
        faction/allegiance and also the worlds that will be terraformed in that group
        '''
        sub = [ (self.sectorSets[i][0] == sectorX) and (self.sectorSets[i][1]==sectorY) for i in range(len(self.sectorSets))]
        
        if not self.np.any(sub):
            print('sector not found, no output')
            return 0
        else:
            #print('sector found')
            pass
        sectInd = self.np.arange(len(self.sectorSets))[sub][0]
        #print('Sector Index: ', sectInd)
        
        sysList, jumpNumber = self.findJumpGroup(sectorX, sectorY, startHex, jump=jump, maxJumps=maxJumps)
        if not type(sysList) == list:
            print('No stars for empire!')
            return
        
        oldName = ''
        for i in range( len(self.sectorSets[sectInd][2]) ):
            if self.sectorSets[sectInd][2][i] in sysList:
                #should be terraformed
                if noNameChange:
                    oldName = self.sectorStarData[sectInd][i][0]
                previousUWP = self.sectorStarData[sectInd][i][1]
                previousTech = self.SS.convertFromEHEX(previousUWP[-1])
                previousPop = previousUWP[4]
                
                self.SS.setSeed(seed = self.sectorSeedList[sectInd][i])
                self.SS.createTotalSystem(empireCode = empireCode, \
                                          noNameChange = noNameChange)
                levelSet = terraformLevel + self.sectorTerraformList[sectInd][i]
                self.SS.terraformWorld(levels=levelSet, \
                                       empireCode = empireCode, \
                                       terraformLimit = terraformLimit, \
                                       habLimit = habLimit, \
                                       noHydro = noHydro, \
                                       minPop = max([minPop, self.SS.convertFromEHEX(previousPop)]), \
                                       noNameChange = noNameChange, \
                                       lastUWP = previousUWP)
                
                self.sectorTerraformList[sectInd][i] = levelSet
                
                        
                self.SS.systemDataList[7] = empireCode
                if noNameChange:
                    self.SS.systemDataList[1] = oldName
                
                currentUWP = self.SS.systemDataList[2]
                
                if self.sectorSets[sectInd][2][i] == '1421':
                    print('Previous UWP: ', previousUWP)
                    print('Level Set: ', levelSet)
                    print('terraform Level: ', terraformLevel)
                    print('Current UWP: ', currentUWP)
                
                
                currentTech = self.SS.convertFromEHEX(self.SS.systemDataList[2][-1])
                if currentTech < max([minTech, previousTech]):
                    #self.SS.systemDataList[2][-1] = self.SS.convertEHEX(minTech)
                    self.SS.systemDataList[2] = currentUWP[0:-1]+self.SS.convertEHEX( max([minTech, previousTech]) )
                if (currentTech > maxTech) or (previousTech > maxTech):
                    self.SS.systemDataList[2] = currentUWP[0:-1]+self.SS.convertEHEX(maxTech)
                
                if currentUWP[4] == '0':
                    self.SS.systemDataList[2] = currentUWP[0:-1]+'0'
                
                '''
                if self.SS.convertFromEHEX(currentUWP[4]) < max([minPop, self.SS.convertFromEHEX(previousUWP[4])]):
                    self.SS.systemDataList[2] = currentUWP[0:4] + \
                            self.SS.convertEHEX(max([minPop, self.SS.convertFromEHEX(previousUWP[4])])) + \
                            currentUWP[5:]
                '''
                
                self.sectorStarData[sectInd][i] = self.SS.systemDataList[1:]
                
            
    def setBaseString(self, sectorX, sectorY, \
                      starHex, \
                      baseString = 'N'):
        '''
        This manually changes the base strings of the world and recalculates the importance
        string
        '''
        sub = [ (self.sectorSets[i][0] == sectorX) and (self.sectorSets[i][1]==sectorY) for i in range(len(self.sectorSets))]
        
        if not self.np.any(sub):
            print('sector not found, no output')
            return 0
        else:
            #print('sector found')
            pass
        sectInd = self.np.arange(len(self.sectorSets))[sub][0]
        #print('Sector Index: ', sectInd)
        
        if starHex in self.sectorSets[sectInd][2]:
            sysInd = self.sectorSets[sectInd][2].index(starHex)
        else:
            print('Star not found!')
            return 
        
        baseStringOrder = 'NSV'
        currentBaseString = self.sectorStarData[sectInd][sysInd][2]
        
        #work through new base string and only append bases if not present
        #for i in range(len(baseString)):
        #if not baseString[i] in currentBaseString:
        currentBaseString = baseString
        #else:
        #    pass
        
        self.sectorStarData[sectInd][sysInd][2] = currentBaseString
        
        #currentIx = self.sectorStarData[sectInd][sysInd][8]
        
        newIx = self.SS.genImportanceExtension(currentBaseString, \
                                               UWP = self.sectorStarData[sectInd][sysInd][1], \
                                               remarkString = self.sectorStarData[sectInd][sysInd][3])
        self.sectorStarData[sectInd][sysInd][8] = newIx
    
    def setStarport(self, sectorX, sectorY, \
                          starHex, \
                          starport = 'C'):
        '''
        This manually changes the base strings of the world and recalculates the importance
        string
        '''
        sub = [ (self.sectorSets[i][0] == sectorX) and (self.sectorSets[i][1]==sectorY) for i in range(len(self.sectorSets))]
        
        if not self.np.any(sub):
            print('sector not found, no output')
            return 0
        else:
            #print('sector found')
            pass
        sectInd = self.np.arange(len(self.sectorSets))[sub][0]
        #print('Sector Index: ', sectInd)
        
        if starHex in self.sectorSets[sectInd][2]:
            sysInd = self.sectorSets[sectInd][2].index(starHex)
        else:
            print('Star not found!')
            return 
        
        currentUWP = self.sectorStarData[sectInd][sysInd][1]
        
        #work through new base string and only append bases if not present
        newUWP = starport + currentUWP[1:]
        
        self.sectorStarData[sectInd][sysInd][1] = newUWP
        
        #currentIx = self.sectorStarData[sectInd][sysInd][8]
        
        newIx = self.SS.genImportanceExtension(self.sectorStarData[sectInd][sysInd][2], \
                                               UWP = self.sectorStarData[sectInd][sysInd][1], \
                                               remarkString = self.sectorStarData[sectInd][sysInd][3])
        self.sectorStarData[sectInd][sysInd][8] = newIx
    
    def setTechLevel(self, \
                     sectorX, \
                     sectorY, \
                     starHex, \
                     techLevel = 14):
        '''
        This manually changes the tech string of the world and recalculates the importance
        string
        '''
        sub = [ (self.sectorSets[i][0] == sectorX) and (self.sectorSets[i][1]==sectorY) for i in range(len(self.sectorSets))]
        
        if not self.np.any(sub):
            print('sector not found, no output')
            return 0
        else:
            #print('sector found')
            pass
        sectInd = self.np.arange(len(self.sectorSets))[sub][0]
        #print('Sector Index: ', sectInd)
        
        if starHex in self.sectorSets[sectInd][2]:
            sysInd = self.sectorSets[sectInd][2].index(starHex)
        else:
            print('Star not found!')
            return 
        
        currentUWP = self.sectorStarData[sectInd][sysInd][1]
        
        #work through new base string and only append bases if not present
        newUWP = currentUWP[:-1] + self.SS.convertEHEX(techLevel)
        
        self.sectorStarData[sectInd][sysInd][1] = newUWP
        
        #currentIx = self.sectorStarData[sectInd][sysInd][8]
        
        newIx = self.SS.genImportanceExtension(self.sectorStarData[sectInd][sysInd][2], \
                                               UWP = self.sectorStarData[sectInd][sysInd][1], \
                                               remarkString = self.sectorStarData[sectInd][sysInd][3])
        self.sectorStarData[sectInd][sysInd][8] = newIx    
    
    def setTravelZone(self, sectorX, sectorY, \
                      starHex, \
                      zoneString = 'A'):
        '''
        This manually changes the travel string of the world 
        '''
        sub = [ (self.sectorSets[i][0] == sectorX) and (self.sectorSets[i][1]==sectorY) for i in range(len(self.sectorSets))]
        
        if not self.np.any(sub):
            print('sector not found, no output')
            return 0
        else:
            #print('sector found')
            pass
        sectInd = self.np.arange(len(self.sectorSets))[sub][0]
        #print('Sector Index: ', sectInd)
        
        if starHex in self.sectorSets[sectInd][2]:
            sysInd = self.sectorSets[sectInd][2].index(starHex)
        else:
            print('Star not found!')
            return 
        
        currentZoneString = self.sectorStarData[sectInd][sysInd][4]
        
        #work through new base string and only append bases if not present
        if not zoneString in currentZoneString:
            currentZoneString = zoneString
        else:
            pass
        
        self.sectorStarData[sectInd][sysInd][4] = currentZoneString
        
        
    def implantSubsector(self, sectorX, sectorY, \
                         subsectorX, subsectorY, \
                         inputFile):
        '''
        Allows read-in of a specific sub-sector
        '''
        sectorSizeX = 40
        sectorSizeY = 32
        
        thisSector = self.np.zeros((sectorSizeX, sectorSizeY))
        
        sub = [ (self.sectorSets[i][0] == sectorX) and (self.sectorSets[i][1]==sectorY) for i in range(len(self.sectorSets))]
        linestring = ''
        if not self.np.any(sub):
            print('sector not found, no output')
            return 0
        else:
            print('sector found')
        sectInd = self.np.arange(len(self.sectorSets))[sub][0]
        print('Sector Index: ', sectInd)
        
        #subsectors will be numbered 0-3, 4-7, 8-11, 12-15
        #they will have a coordinate of horiz = 0-3 and vert = 0-3
        #need to sector the ystr+xstr values that will be culled
        xset = self.np.zeros(2)
        yset = self.np.zeros(2)
        
        xset[0] = int(subsectorX*sectorSizeY/4)
        xset[1] = xset[0] + int(sectorSizeY/4)
        
        yset[0] = int(subsectorY*sectorSizeX/4)
        yset[1] = yset[0] + int(sectorSizeX/4)
        
        #print('xset: ', xset)
        #print('yset: ', yset)
        
        #the strings are (xset + 1)(yset + 1)
        sub = self.np.zeros( len(self.sectorSets[sectInd][2]), dtype=bool)
        
        for i in range( len(self.sectorSets[sectInd][2]) ):
            hexValue = self.sectorSets[sectInd][2][i]
            thisXstr = hexValue[:2]
            thisYstr = hexValue[2:]
        
            xvalue = int(thisXstr) - 1
            yvalue = int(thisYstr) - 1
            
            if ((xvalue >= xset[0] and xvalue < xset[1]) and \
                (yvalue >= yset[0] and yvalue < yset[1])):
                sub[i] = True
        
        #print('sub 2: ', sub)
        #sub2 = self.np.arange( len(sub) )[sub]
        subSector = []
        subSectInd = []
        for i in range( len(sub) ):
            if sub[i]:
                subSector.append( self.sectorSets[sectInd][2][i] )
                subSectInd.append( i )
            
        #subSector = self.sectorSets[sectInd][2][sub2]
        #for i in range( len(subSector) ):
        #    print('Hex value: ', subSector[i], ' index value: ', subSectInd[i])
        
        #scrub the old sub-sector
        for i in range( len(subSectInd) ):
            del self.sectorSets[sectInd][2][subSectInd[-(i+1)]]
            del self.sectorStarData[sectInd][subSectInd[-(i+1)]]
            del self.sectorSeedList[sectInd][subSectInd[-(i+1)]]
            del self.sectorTerraformList[sectInd][subSectInd[-(i+1)]]
        
        #ystart = sectorX*sectorSizeY
        #xstart = -1*sectorY*sectorSizeX
        
        
        #now, read in the new data
        entries = ['Hex', \
                   'Name', \
                   'UWP', \
                   'Bases', \
                   'Remarks', \
                   'Zone', \
                   'PBG', \
                   'Allegiance', \
                   'Stars', \
                   '{Ix}', \
                   '{Ex}', \
                   '{Cx}', \
                   'Nobility', \
                   'W']
        newdata = self.pd.read_csv(inputFile, sep='\t', names=entries)
        
        for i in range( newdata.Hex.count() ):
            self.sectorSets[sectInd][2].append( str(newdata.Hex[i]).zfill(4) )
            starDat = []
            for j in range(1, len(entries)):
                if entries[j] == 'PBG':
                    starDat.append( str(newdata[entries[j]][i]).zfill(3) )
                else: 
                    starDat.append( str(newdata[entries[j]][i]) )
                
            self.sectorStarData[sectInd].append(starDat)
            self.sectorSeedList[sectInd].append(1)
            self.sectorTerraformList[sectInd].append(9)
        
        print('number of sectorSets: ', len(self.sectorSets[sectInd][2]))
        print('number of sectorStarData: ', len(self.sectorStarData[sectInd]))
        
        
    def deleteStar(self, sectorX, sectorY, \
                   starHex):
        '''
        deletes a specific, single star in a defined sector
        starHex has to be a string
        '''
        print('Attempting deletion of  star at '+starHex)
        #sectorSizeX = 40
        #sectorSizeY = 32
        
        #thisSector = self.np.zeros((sectorSizeX, sectorSizeY))
        
        sub = [ (self.sectorSets[i][0] == sectorX) and (self.sectorSets[i][1]==sectorY) for i in range(len(self.sectorSets))]
        #linestring = ''
        if not self.np.any(sub):
            print('sector not found, no output')
            return 0
        else:
            print('sector found')
        sectInd = self.np.arange(len(self.sectorSets))[sub][0]
        print('Sector Index: ', sectInd)
        
        sub = [self.sectorSets[sectInd][2][i] == starHex \
               for i in range(len(self.sectorSets[sectInd][2]))]
        
        if any(sub):
            subInd = np.arange(len(sub))[sub][0]
        
            del self.sectorSets[sectInd][2][subInd]
            del self.sectorStarData[sectInd][subInd]
            del self.sectorSeedList[sectInd][subInd]
            del self.sectorTerraformList[sectInd][subInd]
        else:
            print('No star found.')

    def evenq_to_cube(self, starHex):
        '''
        converts an evenq xy pair to cube coordinates
        Modification to handle star hex longer than 4 standard numbers - 
        this is only used for world coordinates
        '''
        if len(starHex) < 4:
            print('starHex: ', starHex)
            
        if len(starHex) == 4:
            eq_X = int(starHex[:2])
            eq_Y = int(starHex[2:])
        else:
            eq_X = int(starHex[:4])
            eq_Y = int(starHex[4:])
        
        x = eq_X
        z = eq_Y - (eq_X + (eq_X&1))/2
        y = -x-z
        return x, y, z

    def starDistance(self, startHex, endHex):
        '''
        calculates the distance between two hex values (both strings) assuming
        they are both in the same sector
        '''
        
        startCube = self.evenq_to_cube(startHex)
        endCube = self.evenq_to_cube(endHex)
        
        distance = 0.5*(abs(startCube[0]-endCube[0]) + \
                        abs(startCube[1]-endCube[1]) + \
                        abs(startCube[2]-endCube[2]))
        
        return distance

    def analyzeSysList(self, sectorX, sectorY, systemList):
        '''
        determines the total population of the list of stars.  probably some other
        metrics later on
        '''
        print('Inserting system...')
        #sectorSizeX = 40
        #sectorSizeY = 32
        
        #thisSector = self.np.zeros((sectorSizeX, sectorSizeY))
        
        sub = [ (self.sectorSets[i][0] == sectorX) and (self.sectorSets[i][1]==sectorY) for i in range(len(self.sectorSets))]
        #linestring = ''
        if not self.np.any(sub):
            print('sector not found, no output')
            return 0
        else:
            #print('sector found')
            pass
        sectInd = self.np.arange(len(self.sectorSets))[sub][0]
        
        
        systemDataSet = []
        totalPop = 0
        popExpList = []
        popList = []
        
        
        for i in range( len(systemList)):
            if systemList[i] in self.sectorSets[sectInd][2]:
                sub = [ x == systemList[i] for x in self.sectorSets[sectInd][2]]
                for j in range(len(sub)):
                    if sub[j]:
                        systemDataSet.append(self.sectorStarData[sectInd][j])
                        
        print('number of systems: ', len(systemDataSet))
        #should not have a compiled list of system data
        #can now compile populations
        for i in range( len(systemDataSet)):
            UWP = self.sectorStarData[sectInd][i][1]
            popExp = self.SS.convertFromEHEX(UWP[4])
            popExpList.append(popExp)
            PBG = self.sectorStarData[sectInd][i][5]
            popMult = max([1, self.SS.convertFromEHEX(PBG[0])])
            totalPop = totalPop + popMult*10**(popExp)
            popList.append(popMult*10**(popExp))
            
        return totalPop, popExpList, popList
    
    def insertStar(self, sectorX, sectorY, \
                   starData):
        '''
        deletes a specific, single star in a defined sector
        starHex has to be a string
        '''
        print('Inserting system...')
        #sectorSizeX = 40
        #sectorSizeY = 32
        
        #thisSector = self.np.zeros((sectorSizeX, sectorSizeY))
        
        sub = [ (self.sectorSets[i][0] == sectorX) and (self.sectorSets[i][1]==sectorY) for i in range(len(self.sectorSets))]
        #linestring = ''
        if not self.np.any(sub):
            print('sector not found, no output')
            return 0
        else:
            #print('sector found')
            pass
        sectInd = self.np.arange(len(self.sectorSets))[sub][0]
        #print('Sector Index: ', sectInd)
        
        #check if it needs to be deleted
        self.deleteStar(sectorX, sectorY, starData[0])
        
        self.sectorSets[sectInd][2].append( starData[0] )
        self.sectorStarData[sectInd].append( starData[1:] )
        self.sectorSeedList[sectInd].append( 1 )
        self.sectorTerraformList[sectInd].append( 0 )
    
    def writeSectorData(self, sectorX, sectorY, filename='default.dat'):
        '''
        writes an ascii text file that includes the starMap data for a given sector
        '''
        #sectorSizeX = 40
        #sectorSizeY = 32
        
        #thisSector = self.np.zeros((sectorSizeX, sectorSizeY))
        
        f = open(filename, 'w')
        entries = ['Hex', \
                   'Name', \
                   'UWP', \
                   'Bases', \
                   'Remarks', \
                   'Zone', \
                   'PBG', \
                   'Allegiance', \
                   'Stars', \
                   '{Ix}', \
                   '{Ex}', \
                   '{Cx}', \
                   'Nobility', \
                   'W']
        delimchar = '\t'
        linestring = ''
               
        if filename == 'default.dat':
            planetData = ['Emape', \
                          'B564500-B', \
                          'N', \
                          'Ag Ni Pi Da', \
                          'A', \
                          '503', \
                          'Im', \
                          'M0 V', \
                          '{ 2 }', \
                          '{A46+2}', \
                          '[1716]', \
                          'BcC', \
                          '6']    
        
        for i in range(len(entries)):
            linestring = linestring + entries[i] + delimchar
        linestring = linestring + '\n'
        f.write(linestring)
        
        #numX = int(self.imgSize[0]/sectorSizeX)
        #numY = int(self.imgSize[1]/sectorSizeY)
        
        sub = [ (self.sectorSets[i][0] == sectorX) and (self.sectorSets[i][1]==sectorY) for i in range(len(self.sectorSets))]
        linestring = ''
        if not self.np.any(sub):
            print('sector not found, no output')
            return 0
        else:
            print('sector found')
            
        #print('testing')    
        #now pass through starlist for sector and write out type
        sectInd = self.np.arange(len(self.sectorSets))[sub][0]
        print('Sector Index: ', sectInd)
        for i in range( len(self.sectorSets[sectInd][2]) ):
            starHex = self.sectorSets[sectInd][2][i]
            linestring = starHex + delimchar
            #print('i value: ', i)
            for j in range( len(entries)-1):
                if filename == 'default.dat':
                    thisData = planetData
                else:
                    thisData = self.sectorStarData[sectInd][i]
                linestring = linestring + thisData[j] + delimchar
            linestring = linestring + '\n'
            f.write(linestring)
    
    def getSystemOrigin(self, sectorX, sectorY, starHex):
        '''
        obtains the system characteristics after all current terraforming activities
        and puts it into the "local" b.SS.<variable> instance
        '''
        sub = [ (self.sectorSets[i][0] == sectorX) and (self.sectorSets[i][1]==sectorY) for i in range(len(self.sectorSets))]
        
        if not self.np.any(sub):
            print('sector not found, no output')
            return 0
        else:
            #print('sector found')
            pass
        sectInd = self.np.arange(len(self.sectorSets))[sub][0]
        
        sysInd = self.sectorSets[sectInd][2].index(starHex)
        
        print('Sector Index: ', sectInd)
        print('System Number: ', sysInd)
        
        print('Existing Planet Card: ')
        cardNames = ['Name', 'UWP', 'Bases', 'Remarks', \
                     'Travel Zone', 'PBG', 'Allegiance', \
                     'Star(s)', 'Importance', 'Economic Ex.', \
                     'Culture Ex.', 'Nobility', 'Worlds']
        
        for i in range(len(self.sectorStarData[sectInd][sysInd])):
            print(cardNames[i]+': ', self.sectorStarData[sectInd][sysInd][i])
        #print(self.sectorStarData[sectInd][sysInd])
        
        previousUWP = self.sectorStarData[sectInd][sysInd][1]
        previousTech = self.SS.convertFromEHEX(previousUWP[-1])
        previousPop = previousUWP[4]
        
        self.SS.setSeed(seed = self.sectorSeedList[sectInd][sysInd])
        self.SS.createTotalSystem()
        if self.sectorTerraformList[sectInd][sysInd] > 0:
            self.SS.terraformWorld(levels=self.sectorTerraformList[sectInd][sysInd], \
                                   lastUWP = previousUWP)
            print('This world is terraformed but details are not completely known!')
            print('Terraform level: ', self.sectorTerraformList[sectInd][sysInd])
        #include some output data to find the best planet
        #print('Best Habitability Score: ', max(self.SS.habScores[0]))
        print(' ')
        habIndex = self.SS.habScores[0].index( max(self.SS.habScores[0]))
        print('Best Habitability Index: ', habIndex)
        #print('Best Hab Score Type: ', self.SS.habScores[2][habIndex])
        print(' ')
        print('Hab Score values')
        habNames = ['Habitability', 'Terraformability', 'Type', 'MSPR', \
                    'Planet Orbit', 'Moon Orbit', 'Resources']
        for i in range(len(self.SS.habScores)):
            print(habNames[i]+': ', self.SS.habScores[i][habIndex])
        
        return [sectInd, sysInd]
    
    def fixLowTechAtmos(self):
        '''
        Passes through entire system map and if a system has a low tech level
        sets the atmosphere to an appropriate limiting value
        '''
        
        #cycle through sectors
        for i in range(len(self.sectorSets)):
            #cycle through star systems
            for j in range( len(self.sectorSets[i][2])):
                #check UWP
                thisUWP = self.sectorStarData[i][j][1]
                #print('This UWP: ', thisUWP)
                
                popValue = self.SS.convertFromEHEX(thisUWP[4])
                atmosValue = self.SS.convertFromEHEX(thisUWP[2])
                techValue = self.SS.convertFromEHEX(thisUWP[-1])
                newTech = 0
                #only worry about this if the population > 0
                if popValue > 0:
                    if atmosValue <= 1:
                        newTech = 8
                    elif atmosValue <= 3:
                        newTech = 5
                    elif ((atmosValue == 4) or (atmosValue == 7) or (atmosValue == 9)):
                        newTech = 3
                    elif atmosValue == 10:
                        newTech = 8
                    elif atmosValue == 11:
                        newTech = 9
                    elif atmosValue == 12:
                        newTech = 10
                    elif ((atmosValue == 13) or (atmosValue == 14)):
                        newTech = 5
                    elif atmosValue == 15:
                        newTech = 8
                    else:
                        pass
                
                if not newTech == 0:
                    settingTech = max([newTech, techValue])
                    self.sectorStarData[i][j][1] = thisUWP[:-1] + self.SS.convertEHEX(settingTech)
                    
    def multiSectorDistance(self, sect1, starhex1, sect2, starhex2):
        '''
        This function calculates distances between stars that cross between
        sector boundaries.
        
        Parameters
        ----------
        sect1 : tuple (int, int)
            Sector of origin in X/Y coordinates
        starhex1 : string
            Star system or map coordinate origin in sector 1
        sect2 : tuple (int, int)
            Sector of target in X/Y coordinates
        starhex2 : string
            Star system or map coordinate in target in sector 2

        Returns
        -------
        Distance in parsecs between origin and target

        '''
        
        '''
        This function processes the regional map by creating a world coordinate
        for each star system.  The starDistance function above calculates the 
        distance for beyond 1 hex just fine, so it just needs some modified
        world coordinates to be extended for multi-sector use.
        '''

        #s1x, s1y = sect1 #unpack sect1 tuple
        #s2x, s2y = sect2

        inputSet = [[sect1, starhex1], [sect2, starhex2]]
        
        newHex = []
        
        for i in range(len(inputSet)):
            sx, sy = inputSet[i][0]
            s_hex = inputSet[i][1]
            
            shx_x = int(s_hex[:2]) #unpack the star hex x coordinate
            shx_y = int(s_hex[2:]) #unpack the star hex y coordinate
            
            #convert sector values to an increment for the star hex
            addX = sx*32
            addY = -1*sy*40
            
            newCoordX = shx_x + addX
            newCoordY = shx_y + addY
            
            newHex.append( str(newCoordX).zfill(4)+str(newCoordY).zfill(4) )
            
            #print(newHex[-1])
            
        return self.starDistance( newHex[0], newHex[1] )

