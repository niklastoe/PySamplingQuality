#######################################################
#######################################################
# Released under the GNU General Public License version 3 by Mike Nemec
#
# Copyright (c) 2016: Mike Nemec
#
# PySampling: Python toolkit to assess the sampling quality of MD simulations 
#               using a multi-trajectory overlap approach
#
# Author:     Mike Nemec <mike.nemec@uni-due.de>
#
# Paper:      Mike Nemec, Daniel Hoffmann.
#             Quantitative Assessment of Molecular Dynamics Sampling for Flexible Systems. 
#             Journal of Chemical Theory and Computation, 13: 400-414, 2017.
#             doi:10.1021/acs.jctc.6b00823.
#             recommended by F1000Prime
#
# current version: v08.05.17-1
#######################################################
# tested with following program versions:
#        Gromacs       v4.6 | v5.1 
#        Amber         AmberTools14
#        Python        2.7.12
#        Anaconda      2.4.1 (64-bit)
#        Matplotlib    1.5.1
#        scipy         0.17.0
#        numpy         1.10.4
#######################################################

## IMPORT NECESSARY MODULES ##
import ast
import os
import numpy as NP
import scipy.misc
import time
from collections import defaultdict
## IMPORT NECESSARY OBJECTS
from Misc import Generate_Directories
    
#######################################################
#------------------------------------------------------
class Overlap(object):
    """
v06.05.17
- this class contains all necessary attributes for the overlap calculation which is run by the wrapper
    Calc_Overlap(EventDir, EventNames, SaveDir, SaveName, CompareList, 
                 WeightDir=None, aMD_Nrs=[], sMD_Nrs=[], SameTraj=None)
- __init__ sets automatically all attributes by
    a) direct assignment                               : self.args = args
    b) extract parameters from the EventNames-files    : self.AttrDict = self._SetAttributes()
    c) set COL_TrajNrList | EventCurve | NormMatrix by merging or modifying the matrices stored in EventNames:
            self.EventCurve, self.NormMatrix, self.COL_TrajNrList = self._Set_Event_Norm_COL(ColDict)
    d) Checks whether necessary Weights do exist, and modify aMD_Nrs/sMD_Nrs to match the TrajNameList
- public bound methods:
    a) def Calculation(self)
    b) def Store(self, Time='')
    c) def Calc_densO(self, Weights, TrajNr, Threshold, mod_ComGrp)
    d) def Calc_STD_densO(self, Weights, TrajNr, Threshold, mod_ComGrp)
    e) def Calc_TotNrWeights(self, Weights, TrajNr, Threshold)
    f) def Calc_confO(self, TrajNr, mod_ComGrp)
    g) def Calc_TotFrames(self, TrajNr)
    h) def Load_Weights(self, TrajNr)
    i) def Check_SaveName(self)
- @staticmethod
    a) def Merge_Cols(EventCurveList, NrOfThresholds, COL_TrajNrList)
    b) def Merge_Rows(EventDir, EventNames)
    c) def Extract_NormName(EventName)
- @property
    a) @WeightDir.setter
    b) @EventDir.setter
    c) @SaveDir.setter
    d) @SaveName.setter
    e) @NrOfTrajs.setter
    f) @CompareList.setter
- to calculate the overlap, the following workflow can be applied:
    1) Object = Overlap(*args, **kwargs)
    2) Object.Calculation()
    3) Object.Store(Time)
    """
    def __init__(self, EventDir, EventNames, SaveDir, SaveName, CompareList, 
                 WeightDir=None, aMD_Nrs=[], sMD_Nrs=[], SameTraj=None):
        #---- INIT super class
        super(Overlap, self).__init__()
        #---- INIT Necessary parameters from Input
        self.SameTraj               = SameTraj
        self.EventNames             = EventNames
        self.aMD_Nrs                = aMD_Nrs
        self.sMD_Nrs                = sMD_Nrs
        self.FMT                    = '%i %s  '+('  %.4f %.4f %i %i')*len(CompareList)
        #---- INIT attributes with @property from Input
        self.EventDir               = EventDir
        self.SaveDir                = SaveDir
        self.WeightDir              = WeightDir
        self.CompareList            = CompareList
        #---- CHECK whether EventNames exist
        ColDict                     = self._Check_EventNames()
        #---- INIT attributes from EventNames-Files
        self.AttrDict               = self._Initialize_AttrDict()
        #---- INIT SaveName with @property: if SameTraj, SaveName is modified using self.EventNames & self.AttrDict
        self.SaveName               = SaveName
        #---- CHECK whether the overlap file already exists
        _ = self.Check_SaveName()
        #---- INIT COL_TrajNrList | EventCurve | NormMatrix by merging or modifying the matrices stored in EventNames
        self.EventCurve, self.NormMatrix, self.COL_TrajNrList = self._Initialize_Event_Norm_COL(ColDict)
        #---- INIT OverlapMatrix with correct sizes
        self.OverlapMatrix          = self._Initialize_OverlapMatrix()
        #---- INIT NrOfTrajs with @property: if SameTraj, SaveName is modified using self.AttrDict
        self.NrOfTrajs              = len(self.EventNames) 
        #---- CHECK CompareList: do the specified trajectories exist in the EventCurves
        Unique_Compare              = self._Check_CompareList()
        #---- MODIFY aMD_Nrs and sMD_Nrs to match CompareList
        self._Check_aMDsMD_Nrs(Unique_Compare)
        #---- CHECK, whether the necessary Weight-Files exist
        self._Check_aMDWeights()
        self._Check_sMDWeights()
        #---- INIT Header for the OverlapMatrix-Text-file
        self.Header                 = self._Initialize_Header()
        
#######################################################################

    @property
    def WeightDir(self):
        return self._WeightDir

    @WeightDir.setter
    def WeightDir(self, value):
        if value is None:
            self._WeightDir = None
        elif not isinstance(value, str):
            raise ValueError('WeightDir must be a string! Check your input\n\tWeightDir = %s' % value)
        elif value[-1] == '/':
            self._WeightDir = value
        else:
            self._WeightDir = '%s/' % value

#######################################################################

    @property
    def EventDir(self):
        return self._EventDir

    @EventDir.setter
    def EventDir(self, value):
        if not isinstance(value, str):
            raise ValueError('EventDir must be a string! Check your input\n\tEventDir = %s' % value)
        elif value[-1] == '/':
            self._EventDir = value
        else:
            self._EventDir = '%s/' % value

#######################################################################

    @property
    def SaveDir(self):
        return self._SaveDir

    @SaveDir.setter
    def SaveDir(self, value):
        if not isinstance(value, str):
            raise ValueError('SaveDir must be a string! Check your input\n\tSaveDir = %s' % value)
        elif value[-1] == '/':
            self._SaveDir = value
        else:
            self._SaveDir = '%s/' % value

#######################################################################

    @property
    def SaveName(self):
        return self._SaveName

    @SaveName.setter
    def SaveName(self, value):
        if self.SameTraj is None:
            ReweightList = ['_noWeight', '_MF+sMD', '_McL+sMD', '_Exp+sMD', '_sMD', '_MF', '_McL', '_Exp']
            if isinstance(self.EventNames, list):
                reweight     = [elem for elem in ReweightList if self.EventNames[0].find(elem) != -1][0]
            else:
                reweight     = [elem for elem in ReweightList if self.EventNames.find(elem) != -1][0]
            if len(reweight) == 0:
                raise ValueError('EventNames={} must contain one "reweight" specification: \n{}'.format(self.EventNames, ReweightList))
            self._SaveName = '%s_%s-%s%s.txt' % (value, self.AttrDict['StartFrame'][0], self.AttrDict['EndingFrame'][0], reweight)
        else:
            self._SaveName = value

#######################################################################

    @property
    def NrOfTrajs(self):
        return self._NrOfTrajs

    @NrOfTrajs.setter
    def NrOfTrajs(self, value):
        if self.SameTraj is not None:
            self._NrOfTrajs = value
        else:
            self._NrOfTrajs = (len(self.NormMatrix[0,:])-2)/len(self.AttrDict['ThresholdList'])
        
#######################################################################

    @property
    def CompareList(self):
        return self._CompareList

    @CompareList.setter
    def CompareList(self, value):
        if self.SameTraj is None:
            self._CompareList = value
        else:
            self._CompareList = []
            for elem in xrange(len(self.EventNames)):
                self._CompareList.append([elem+1])
            self._CompareList = [self._CompareList]
        #---- CompareList has to have same lengths to combine them
        if NP.any([len(elem) != len(self._CompareList[0]) for elem in self._CompareList]):
            raise ValueError('CompareList does not have the same lengths of TrajNrs'+\
                             '\n\tCompareList lengths = {}'.format([len(elem) for elem in self._CompareList]))
        
#######################################################################

    def Store(self, Time=''):
        """
v06.05.17
This bound method stores the OverlapMatrix with the specified Header, Spacing using SaveDir+SaveName
INPUT:
    self.SaveDir             : {STRING}    storing directory, e.g. 'Overlap/';
    self.SaveName            : {STRING}    savename OverlapMatrix, e.g. 'Overlap_R5_Pairs_0-10000_noWeight.txt';
    self.OverlapMatrix       : {NP.array}  stores the OverlapMatrix as numpy array;
    self.FMT                 : {STRING}    spacing for the columns of the OverlapMatrix stored as text-file;
    self.Header              : {STRING}    Header of the text-file;
OUTPUT:
    '%s%s' % (SaveDir, self.SaveName) containing the NP.array OverlapMatrix with the spacings self.FMT and self.AttrDict['Header']
        """
        Generate_Directories(self.SaveDir)
        if Time != '':
            CalcTime = 'calculation time = %s seconds' % Time
        else:
            CalcTime = ''
        NP.savetxt('%s%s' % (self.SaveDir, self.SaveName), self.OverlapMatrix, fmt=self.FMT, header=self.Header.format(CalcTime))
        
#######################################################################

    def _Generate_ColumnList(self, Threshold):
        """
v06.05.17
Helper function: 
    - Generates a ColumnList corresponding to the columns of the EventCurve-file for a certain Threshold 
    - equivalent/correspond to COL_TrajNrList
    - these columns are extracted from EventCurve[:, ColumnList]
INPUT:
    Threshold                        :  {FLOAT}      threshold which was used for the eventcurve generation, e.g. 0.2;
    self.COL_TrajNrList              :  {INT-LIST}   defines which trajectories are stored in the columns of the eventcurve;
    self.NrOfTrajs                   :  {INT}        total number of trajectories in the eventcurve;
    self.AttrDict['ThresholdList']   :  {FLOAT-LIST} all thresholds used for the eventcurve generation;
OUTPUT:
    return ColumnList
        """
        ColumnList = {}
        if self.COL_TrajNrList is None:
            Index = 0
            for Tr in xrange(1,self.NrOfTrajs+1):
                ColumnList[Tr] = Index+2+self.AttrDict['ThresholdList'].index(Threshold)*(self.NrOfTrajs)
                Index += 1
        else:
            Index = 0
            for Tr in self.COL_TrajNrList:
                ColumnList[Tr] = Index+2+self.AttrDict['ThresholdList'].index(Threshold)*(self.NrOfTrajs)
                Index += 1
        return ColumnList

#######################################################################

    def Load_Weights(self, TrajNr):
        """
v06.05.17
This bound method loads the corresponding Weights (aMD or sMD) for the submitted trajectory (TrajNr)
- for aMD: different re-weighting schemes are tested ['MF', 'McL', 'Exp']
- for sMD: only one re-weighting scheme exists
- for cMD: Weights are NP.ones( Size )
- returns the corresponding Weight slizes
INPUT:
    TrajNr
    self.WeightDir {STRING}    <default None>, FOR RE-WEIGHTING ONLY, Directory, where the aMD/sMD weights are located, e.g. 'EventCurves/Weights/';
    self.AttrDict['TrajNameList'] : {LIST}      List of used trajectory names, WITHOUT ENDING;
    self.AttrDict['StartFrame']   : {INT-LIST}  Starting frame for each EventCurve, if SameTraj==None, all StartFrame are equal;
    self.AttrDict['EndingFrame']  : {INT-LIST}  Ending frame for each EventCurve, if SameTraj==None, all EndingFrame are equal;
    self.AttrDict['Iterations']   : {INT}       Number of Iterations for mean-field (MF) reweighting of aMD trajectories;
    self.AttrDict['aMD_reweight'] : {STRING}    re-weighting algorithm, e.g. MF, McL or Exp;
    self.AttrDict['Lambda']       : {FLOAT}     re-scaling factor for the sMD reweighting scheme;
    self.AttrDict['Order']        : {INT}       Order of the MacLaurin expansion reweighting of aMD trajectories;
    
OUTPUT:
    return Weights
        """
        if self.aMD_Nrs.count(TrajNr) > 0:
        #---- Load Weights for aMD references
            if self.AttrDict['aMD_reweight'] == 'MF':
                Weights = NP.exp(NP.genfromtxt('%saMD_Weight_MF_%s_%s-%s_Iter%s.txt' % \
                           (self.WeightDir, self.AttrDict['TrajNameList'][TrajNr-1], self.AttrDict['StartFrame'][TrajNr-1], 
                            self.AttrDict['EndingFrame'][TrajNr-1], self.AttrDict['Iterations'][TrajNr-1])))
            elif self.AttrDict['aMD_reweight'] == 'McL':
                tempWeights = NP.genfromtxt('%saMD_Weight_%s.txt' % \
                    (self.WeightDir, self.AttrDict['TrajNameList'][TrajNr-1]), usecols=(0))[self.AttrDict['StartFrame'][TrajNr-1]:self.AttrDict['EndingFrame'][TrajNr-1]]
                Weights = NP.zeros(len(tempWeights))
                for Ord in xrange(0,self.AttrDict['Order']+1):
                    Weights = NP.add(Weights, 
                                     NP.divide( NP.power( tempWeights , Ord), 
                                                float(scipy.misc.factorial(Ord)) ) )
            else:
                Weights = NP.exp(NP.genfromtxt('%saMD_Weight_%s.txt' % \
                    (self.WeightDir, self.AttrDict['TrajNameList'][TrajNr-1]), usecols=(0))[self.AttrDict['StartFrame'][TrajNr-1]:self.AttrDict['EndingFrame'][TrajNr-1]])
        elif self.sMD_Nrs.count(TrajNr) > 0:
        #---- Load Weights for sMD references
            Weights = NP.genfromtxt('%ssMD_Weight_lambda%s_%s_%s-%s_Iter%s.txt' % \
                   (self.WeightDir, self.AttrDict['Lambda'], self.AttrDict['TrajNameList'][TrajNr-1], self.AttrDict['StartFrame'][TrajNr-1], 
                    self.AttrDict['EndingFrame'][TrajNr-1], self.AttrDict['Iterations'][TrajNr-1]))
        else:
        #---- Load Weights = 1 for cMD references
            Weights = NP.ones( len(self.EventCurve[self.EventCurve[:,1]==TrajNr][:,0]) )
        return Weights

#######################################################################

    def Calc_densO(self, Weights, TrajNr, Threshold, mod_ComGrp):
        """
v06.05.17
This bound method calculates the "density overlap" <f_dens(k,L;r)> (NOT NORMALIZED) following Eq.(17) of the Paper:

                                                        / min{norm(Events_{r,kappa l}): l \in L}                \
    return <sum_k w_{r,k} * f_dens(k,L;r)> = sum_kappa | --------------------------------------- * w_{r, kappa}  |
                                                        \ max{norm(Events_{r,kappa l}): l \in L}                /
INPUT:
    Weights                        :   {NP.array}   corresponding weights for the reference frames kappa in trajectory k = TrajNr;
    TrajNr                         :   {INT}        reference trajectory k which defines the reference frames kappa;
    Threshold                      :   {FLOAT}      threshold which was used for the eventcurve generation, e.g. 0.2;
    mod_ComGrp                     :   {INT-LIST}   modified CompareGrp to match the Columns of the EventCurve, 
                                                    these trajectories are concatenated/summed for a possible Group-Overlap;
    self.EventCurve                :   {NP.array}   events stored for different Thresholds and trajectories in one NP.array;
    self.NormMatrix                :   {NP.array}   number of events stored for different Thresholds in one NP.array;
    self.AttrDict['ThresholdList'] :   {FLOAT-LIST} Threshold parameter list used to calculate the EventCurves;
OUTPUT: 
    return <sum_k w_{r,k} * f_dens(k,L;r)>
        """
        with NP.errstate(divide='ignore', invalid='ignore'):
            #---- calculates the normalized Events (array) with respect to the reference trajectory TrajNr <=> kappa 
            Events = [NP.divide(\
                               NP.sum(self.EventCurve[self.EventCurve[:, 1]==TrajNr][:, elem], axis=1),
                               NP.sum(self.NormMatrix[0, elem], axis=0)) \
                     for elem in mod_ComGrp]
            #---- uses the correct Weight slize
            Weights = (Weights if Weights.shape == (len(self.EventCurve[self.EventCurve[:, 1]==TrajNr][:, 0]),)\
                               else Weights[:, self.AttrDict['ThresholdList'].index(Threshold)])
            #---- calculates the NON-NORMALIZED f_dens for the reference trajectory k=TrajNr
            return  NP.sum(\
                        NP.multiply(\
                            NP.divide(\
                                NP.min(Events, axis=0),
                                NP.max(Events, axis=0)),
                            Weights))

#######################################################################

    def Calc_STD_densO(self, Weights, TrajNr, Threshold, mod_ComGrp):
        """
v06.05.17
This bound method calculates the cumulant of "density overlap" <f_dens(k,L;r)> (NOT NORMALIZED) (see def Calc_densO()):
    
                                                          / / min{norm(Events_{r,kappa l}): l \in L} \^2                 \
    return <sum_k w_{r,k} * f_dens(k,L;r)>^2 = sum_kappa | | ---------------------------------------  |   * w_{r, kappa}  |
                                                          \ \ max{norm(Events_{r,kappa l}): l \in L} /                   /
The idea is to calculate the 
        Variance = <densO^2> - <densO>^2
where the first summand is Calc_STD_densO(self, Weights, TrajNr, Threshold, mod_ComGrp)

INPUT:
    Weights                        :   {NP.array}   corresponding weights for the reference frames kappa in trajectory k = TrajNr;
    TrajNr                         :   {INT}        reference trajectory k which defines the reference frames kappa;
    Threshold                      :   {FLOAT}      threshold which was used for the eventcurve generation, e.g. 0.2;
    mod_ComGrp                     :   {INT-LIST}   modified CompareGrp to match the Columns of the EventCurve, 
                                                    these trajectories are concatenated/summed for a possible Group-Overlap;
    self.EventCurve                :   {NP.array}   events stored for different Thresholds and trajectories in one NP.array;
    self.NormMatrix                :   {NP.array}   number of events stored for different Thresholds in one NP.array;
    self.AttrDict['ThresholdList'] :   {FLOAT-LIST} Threshold parameter list used to calculate the EventCurves;                             
OUTPUT: 
    return <sum_k w_{r,k} * f_dens(k,L;r)>^2
        """
        with NP.errstate(divide='ignore', invalid='ignore'):
            #---- calculates the normalized Events (array) with respect to the reference trajectory TrajNr <=> kappa 
            Events = [NP.divide(\
                               NP.sum(self.EventCurve[self.EventCurve[:,1]==TrajNr][:,elem], axis=1),
                               NP.sum(self.NormMatrix[0,elem], axis=0)) \
                         for elem in mod_ComGrp]
            #---- uses the correct Weight slize
            Weights = (Weights if Weights.shape == (len(self.EventCurve[self.EventCurve[:, 1]==TrajNr][:, 0]),)\
                               else Weights[:, self.AttrDict['ThresholdList'].index(Threshold)])
            #---- calculates the NON-NORMALIZED f_dens for the reference trajectory k=TrajNr
            return  NP.sum(\
                        NP.multiply(\
                            NP.power(\
                                NP.divide(\
                                    NP.min(Events, axis=0),
                                    NP.max(Events, axis=0)),
                                2),
                            Weights))

#######################################################################

    def Calc_TotNrWeights(self, Weights, TrajNr, Threshold):
        """
v06.05.17
This bound method calculates the sum of Weights <sum_k w_{r,k}>, which is the normalization factor for 
    a) densO      (see Calc_densO())
    b) STD_densO  (see Calc_STD_densO())

- for cMD, sMD or aMD with 'Exp' or 'McL' reweighting scheme: Weights.shape = (len(Events), )
- for             aMD with 'MF'           reweighting scheme: Weights.shape = (len(Events), len(ThresholdList))

INPUT:
    Weights                        :   {NP.array}   corresponding weights for the reference frames kappa in trajectory k = TrajNr;
    TrajNr                         :   {INT}        reference trajectory k which defines the reference frames kappa;
    Threshold                      :   {FLOAT}      threshold which was used for the eventcurve generation, e.g. 0.2;
    self.EventCurve                :   {NP.array}   events stored for different Thresholds and trajectories in one NP.array;
    self.AttrDict['ThresholdList'] :   {FLOAT-LIST} Threshold parameter list used to calculate the EventCurves;
OUTPUT:
    return <sum_k w_{r,k}>
        """
        #---- uses the correct Weight slize
        Weights = (Weights if Weights.shape == (len(self.EventCurve[self.EventCurve[:, 1]==TrajNr][:, 0]),)\
                           else Weights[:, self.AttrDict['ThresholdList'].index(Threshold)])
        return NP.sum(Weights)
#######################################################################

    def Calc_confO(self, TrajNr, mod_ComGrp):
        """
v06.05.17
This bound method calculates the "conformational overlap" <f_dens(k,L;r)> (NOT NORMALIZED) corresponding to Eq.(15) of the Paper:

    return <TotFrames * f_conf(k,L;r)> = sum_kappa( sign_l(Events_{r,kappa l}) )

- the signum function mimics Eq.(16) of the Paper
- the NP.min() mimics the product of Eq.(15)
- TotFrames = n_K of Eq.(15), total number of frames in Events

INPUT:
    TrajNr           :   {INT}      reference trajectory k which defines the reference frames kappa;
    mod_ComGrp       :   {INT-LIST} modified CompareGrp to match the Columns of the EventCurve, 
                                    these trajectories are concatenated/summed for a possible Group-Overlap;
    self.EventCurve  :   {NP.array} events stored for different Thresholds and trajectories in one NP.array;
OUTPUT: 
    return <TotFrames * f_conf(k,L;r)>
        """
        Events = [NP.sum(self.EventCurve[self.EventCurve[:,1]==TrajNr][:,elem], axis=1)\
                  for elem in mod_ComGrp]
        return  NP.sum(\
                    NP.sign(\
                        NP.min(Events, axis=0) ) )

#######################################################################

    def Calc_TotFrames(self, TrajNr):
        """
v06.05.17
This bound method calculates the total number of Frames n_k for one single reference trajectory k = TrajNr (see Eq.(15) of the Paper)

INPUT:
    TrajNr           :   {INT}      reference trajectory k which defines the reference frames kappa;
    self.EventCurve  :   {NP.array} events stored for different Thresholds and trajectories in one NP.array;    
OUTPUT: 
    return <TotFrames for one trajectory k = TrajNr>
        """
        return len(self.EventCurve[self.EventCurve[:,1]==TrajNr][:,0])

#######################################################################

    def Calculation(self):
        """
v06.05.17
- This bound method calculates the overlap and stores the result in self.OverlapMatrix
- the overlap is calculated for different Thresholds and different groups of trajectory comparisons L stored in self.CompareList
INPUT:
    self.CompareList               : {LIST,TUPLE,LIST} list of groups of trajectories, which are compared and the overlap is calculated
                                            e.g. [([1],[3]), ([1,2],[3,4]), ([1,2],[3])] calculates the overlap between trajectories
                                                a) (1)vs(3) 
                                                b) (1+2)vs(3+4)
                                                c) (1+2)vs(3)
                                            and stores them in consecutive columns;
    self.EventCurve                : {NP.array}        events stored for different Thresholds and trajectories in one NP.array;
    self.NormMatrix                : {NP.array}        number of events stored for different Thresholds in one NP.array;
    self.AttrDict['ThresholdList'] : {FLOAT-LIST}      Threshold parameter list used to calculate the EventCurves;                -
    self.OverlapMatrix             : {NP.array}        stores the OverlapMatrix as numpy array;
OUTPUT:
    self.OverlapMatrix[:, 2::4] = densO
    self.OverlapMatrix[:, 3::4] = STD_densO
    self.OverlapMatrix[:, 4::4] = confO
    self.OverlapMatrix[:, 5::4] = TotFrames
        """
        RowIndex = 0
        for Threshold in self.AttrDict['ThresholdList']:
            #---- Generate ColumnList corresponding to the rows of the EventCurve-file <-> COL_TrajNrList
            ColumnList = self._Generate_ColumnList(Threshold)
            CompareIndex = 0
            for CompareGrp in self.CompareList: ## comparison GroupX vs GroupY vs ...
    #### #### ####
    #---- define the actual index of OverlapMatrix[RowIndex,:]
                RowIndex = len(CompareGrp)*self.AttrDict['ThresholdList'].index(Threshold)
    #### #### ####
    #---- generate modified CompareGrp to match the Columns of the EventCurve
                mod_ComGrp = [NP.array([ColumnList[ele] for ele in elem]) for elem in CompareGrp]
    #### #### ####
                for SinglCompGrp in CompareGrp:
    #---- init densO; TotnrWeights; confO; TotalNrOfFrames
                    densO = 0
                    STD_densO = 0
                    TotNrWeights = 0
                    confO = 0
                    TotFrames = 0
        #### #### ####
        #---- run through all trajectories of one comparison group <SinglCompGrp>
                    for TrajNr in SinglCompGrp:  ## TrajNr of one comparison Group, which has to be summed up
                        Weights = self.Load_Weights(TrajNr)
            #### #### ####
            #---- densO
                        densO += self.Calc_densO(Weights, TrajNr, Threshold, mod_ComGrp)
            #### #### #### v20.10.16
            #---- STD_densO
                        STD_densO += self.Calc_STD_densO(Weights, TrajNr, Threshold, mod_ComGrp)
            #### #### ####
            #---- Total Nr of Weights for the average of densO
                        TotNrWeights += self.Calc_TotNrWeights(Weights, TrajNr, Threshold)
            #### #### ####
            #---- confO
                        confO += self.Calc_confO(TrajNr, mod_ComGrp)
            #### #### ####
            #---- Total Nr of Frames of the reference trajectory
                        TotFrames += self.Calc_TotFrames(TrajNr)
        #### #### ####
        #---- calculate STD_densO for one reference trajectory   
                    STD_densO = (float(STD_densO)/float(TotNrWeights) - NP.power(float(densO)/float(TotNrWeights),2))
    #### #### ####
    #---- "normalize" densO by averaging over all reference frames, which is equal to the sum of Weights
                    ### STORE densO, confO, TotFrames to OverlapMatrix
                    self.OverlapMatrix[RowIndex, 2+CompareIndex+0] = float(densO)/float(TotNrWeights)
                    self.OverlapMatrix[RowIndex, 2+CompareIndex+1] = STD_densO
                    self.OverlapMatrix[RowIndex, 2+CompareIndex+2] = confO
                    self.OverlapMatrix[RowIndex, 2+CompareIndex+3] = TotFrames
                    RowIndex += 1
            #---- raise the Column Index for every new CompareGroup        
                CompareIndex += 4
        
#######################################################################

    def _Initialize_AttrDict(self):
        """ 
v06.05.17
Helper Function: This (private) bound method calls the EventNames-Parser to extract the attributes specified in the Header of the EventNames-File(s)
- It generates the necessary <NormName> using the staticmethod Extract_NormName(Name)
- all extracted attributes are stored as a dictionary to obtain a simple, grouped representation

INPUT:
    self.SameTraj          : {INT}    if SameTraj is not None, different parts of the SameTraj-th trajectory 
                                        are compared returning different simulation time parts ;
    self.EventDir          : {STRING} Directory, where the Norm_EventCurves.txt are stored;
    self.EventNames        : {STRING/LIST}     Name of the EventCurve, can also be a list to combine different _RowX, _ColY files |
                                        if SameTraj is not None, submit multiple simulation times of SAME trajectory,
                                        to calculate the overlap between different time parts of the SAME trajectory | 
                                        e.g. 'V3_S1-S10_0-500_noWeight.npy';
OUTPUT:
    return AttrDict
        with AttrDict.keys() = ['Header', 'TrajNameList', 'ThresholdList', 'StartFrame', 'EndingFrame', 
                                'Iterations', 'Lambda', 'Order', 'aMD_reweight', 'updated_TrajLengthList'] 
        """
        if self.SameTraj is not None:
            NormName = [Overlap.Extract_NormName(Name) for Name in self.EventNames]
        elif isinstance(self.EventNames, list):
            NormName = Overlap.Extract_NormName(self.EventNames[0])
        else:
            NormName = Overlap.Extract_NormName(self.EventNames)

        return self._Extract_AttributesFromFile(NormName)
        
#######################################################################

    def _Initialize_Event_Norm_COL(self, ColDict):
        """
v06.05.17
Helper Function: This (private) bound method loads <EventCurve, NormMatrix, COL_TrajNrList> using one of the two methods for
    a) SameTraj is not None
    b) SameTraj is     None

INPUT:
    ColDict                         : {DEFAULTDICT}  stores the _RowsX and _ColsY information from the self.EventNames-Strings,
                                                        Keys = Rows/Cols traj numbers | Values = Position in the self.EventNames-List;
    self.EventDir                   : {STRING}       Directory, where the Norm_EventCurves.txt are stored;
    self.EventNames                 : {STRING/LIST}  Name of the EventCurve, can also be a list to combine different _RowX, _ColY files |
                                                     if SameTraj is not None, submit multiple simulation times of SAME trajectory,
                                                     to calculate the overlap between different time parts of the SAME trajectory | 
                                                     e.g. 'V3_S1-S10_0-500_noWeight.npy';
    self.SameTraj                   : {INT}          if SameTraj is not None, different parts of the SameTraj-th trajectory 
                                                        are compared returning different simulation time parts ;
    self.AttrDict['TrajNameList']   : {LIST}         List of used trajectory names, WITHOUT ENDING;
    self.AttrDict['ThresholdList']  : {FLOAT-LIST}   Threshold parameter list used to calculate the EventCurves;
    self.AttrDict['StartFrame']     : {INT-LIST}     Starting frame for each EventCurve, if SameTraj==None, all StartFrame are equal;
    self.AttrDict['EndingFrame']    : {INT-LIST}     Ending frame for each EventCurve, if SameTraj==None, all EndingFrame are equal;
OUTPUT:
    return EventCurve, NormMatrix, COL_TrajNrList
        """
        if self.SameTraj is not None:
            return self._Initialize_SameTrajAttributes()
        else:
            return self._Initialize_DiffTrajAttributes(ColDict)

#######################################################################

    def _Initialize_SameTrajAttributes(self):
        """
v06.05.17
Helper Function: This (private) bound method modifies the EventCurve for SameTraj not None:
- different parts of the Events of ONE trajectory are extracted and stored into different columns
- the same is done for NormMatrix
- then, EventCurve and NormMatrix can be used for the standard overlap calculation 

INPUT:
    self.EventDir                   : {STRING}       Directory, where the Norm_EventCurves.txt are stored;
    self.EventNames                 : {STRING/LIST}  Name of the EventCurve, can also be a list to combine different _RowX, _ColY files |
                                                     if SameTraj is not None, submit multiple simulation times of SAME trajectory,
                                                     to calculate the overlap between different time parts of the SAME trajectory | 
                                                     e.g. 'V3_S1-S10_0-500_noWeight.npy';
    self.SameTraj                   : {INT}          if SameTraj is not None, different parts of the SameTraj-th trajectory 
                                                        are compared returning different simulation time parts ;
    self.AttrDict['TrajNameList']   : {LIST}         List of used trajectory names, WITHOUT ENDING;
    self.AttrDict['ThresholdList']  : {FLOAT-LIST}   Threshold parameter list used to calculate the EventCurves;
    self.AttrDict['StartFrame']     : {INT-LIST}     Starting frame for each EventCurve, if SameTraj==None, all StartFrame are equal;
    self.AttrDict['EndingFrame']    : {INT-LIST}     Ending frame for each EventCurve, if SameTraj==None, all EndingFrame are equal;
OUTPUT:
    return EventCurve, NormMatrix, COL_TrajNrList    
        """
        #--- ERROR DETECTION
        ColS = []
        refColS = []
        for Name in self.EventNames:
            if Name.find('Col') != -1:
                ColS.extend([int(elem) for elem in Name.split('_Col')[1].split('.')[0].split('_')])
                refColS = [int(elem) for elem in Name.split('_Col')[1].split('.')[0].split('_')]
        if NP.unique(refColS) != NP.unique(ColS):
            raise('EventNames have to share the same Columns. Check your input\nEventNames = {}'.format(self.EventNames))
        if refColS != []:
            COL_TrajNrList = NP.unique(refColS)
            temporary_NrOfTrajs      = len(COL_TrajNrList)
        else:
            temporary_NrOfTrajs  = len(self.AttrDict['TrajNameList'])
            COL_TrajNrList = range(1,temporary_NrOfTrajs+1)
    #--- Generate ColumnList for SameTraj corresponding to the rows of the EventCurve-file <-> COL_TrajNrList
        ColumnList_ST = [0,1];
        for Threshold in self.AttrDict['ThresholdList']:
            Index = 0
            for Tr in COL_TrajNrList:
                if Tr == self.SameTraj:
                    ColumnList_ST.append(Index+2+self.AttrDict['ThresholdList'].index(Threshold)*(temporary_NrOfTrajs))
                Index += 1
    #--- extract Starting and Ending frames
        LEN = NP.sum(NP.subtract(self.AttrDict['EndingFrame'], self.AttrDict['StartFrame']))
    #--- Combine EventCurves & NormMatrices
        EventCurve = NP.zeros( (LEN, 2+len(self.EventNames)*len(self.AttrDict['ThresholdList'])) )
        NormMatrix = NP.zeros( (1, 2+len(self.EventNames)*len(self.AttrDict['ThresholdList'])) )
        for Kai in xrange(len(self.EventNames)):
        #---- define NormName v23.06.16
            NormName = Overlap.Extract_NormName(self.EventNames[Kai])
        #----
            if self.EventNames[Kai][-3:] == 'npy':
                temp = NP.load('%s%s' % (self.EventDir, self.EventNames[Kai]))[:,ColumnList_ST]
            else:
                temp = NP.genfromtxt('%s%s' % (self.EventDir, self.EventNames[Kai]), usecols=ColumnList_ST)
            tempNorm = NP.genfromtxt('%s%s' % (self.EventDir, NormName), usecols=ColumnList_ST)
        #--- CHECK if SameTraj is present
            if [elem for elem in NP.unique(temp[:,1])].count(self.SameTraj) == 0:
                raise ValueError('SameTraj = %s is not present in the EventCurve\n\tEventDir = %s\n\tEventNames = %s' % \
                                        (self.SameTraj, self.EventDir, EventNames))
            temp = temp[temp[:,1] == self.SameTraj]
            StEvent = 0
            tempGroupNr = 0
            for St, En in zip(self.AttrDict['StartFrame'], self.AttrDict['EndingFrame']):
        #--- init Frames
                EventCurve[StEvent:(StEvent+En-St),0] = temp[St:En, 0]
        #--- init GroupNr
                EventCurve[StEvent:(StEvent+En-St),1] = tempGroupNr + 1
                tempGroupNr += 1
                for Koi in xrange(len(self.AttrDict['ThresholdList'])):
        #--- copy TrajNr to different Columns corresponding to the EventNames
                    EventCurve[StEvent:(StEvent+En-St),2+Kai+Koi*len(self.EventNames)] = temp[St:En, 2+Koi]
                    if (St,En) == (self.AttrDict['StartFrame'][0], self.AttrDict['EndingFrame'][0]):
                        NormMatrix[0, 2+Kai+Koi*len(self.EventNames)] = tempNorm[2+Koi]
                StEvent += En-St
        return EventCurve, NormMatrix, COL_TrajNrList
    
#######################################################################
    
    @staticmethod
    def Merge_Cols(EventCurveList, NrOfThresholds, COL_TrajNrList):
        """
v25.07.16
    - this supporting function merges different EventCurves with different COL_TrajNrList together by concatenation
    - COL_TrajNrList is sorted in ascending order
    - duplicate columns (trajectories) are automatically removed
INPUT:
    EventCurveList  :  multiple list with same X-dim, ColNrList Y-dim, e.g. [EventCurve_Col1_2.npy, EventCurve_Col3_4.npy]
    NrOfThresholds  :  number of present thresholds in the EventCurves
    COL_TrajNrList  :  represents the trajectory numbers for which the Events are counted with respect to a certain reference
                        e.g. [1,2,67,97]
OUTPUT:
    returns
        1. merged EventCurve with sorted COL_TrajNrList without duplicates
        2. sorted & unique COL_TrajNrList representing the columns in the EventCurve
        """
        if len(EventCurveList) == 1 or COL_TrajNrList is None: 
            ## this means, in every EventCurve from the EventCurveList, there are ALL trajs & thresholds
            return EventCurveList[0], COL_TrajNrList
        else:
          #### Concatenate FIRST trajectories of each threshold together, THEN follow with the next threshold
            for Rad in xrange(NrOfThresholds):
                for EV in EventCurveList:
                    if 'EventCurve' not in locals():
                        EventCurve = EV[:,0:(2+(len(EV[0,:])-2)/NrOfThresholds)]
                    else:
                        EventCurve = NP.concatenate( (EventCurve, EV[:,(2+Rad*(len(EV[0,:])-2)/NrOfThresholds):(2+(Rad+1)*(len(EV[0,:])-2)/NrOfThresholds)]), axis=1)
            new_COL_TrajNrList, Ind = NP.unique(COL_TrajNrList, return_index=True)
            new_Indices = [0,1]+[2+elem+Rad*len(COL_TrajNrList) for Rad in xrange(NrOfThresholds) for elem in Ind]
        #-----
            return EventCurve[:,new_Indices], new_COL_TrajNrList
 ################################

    @staticmethod
    def Merge_Rows(EventDir, EventNames):
        """
v25.07.16
    - this supporting function merges different EventCurves with different ROW_TrajNrList together by concatenation
    - concatenate (Row, axis = 0) different EventCurves corresponding to entries in EventNames-List
INPUT:
    EventDir    :  directory
    EventNames  :  EventNames with same Cols, different Rows are merged
OUTPUT:
    return EventCurve (merged)
        """
        for EV in EventNames:
            if 'EventCurve' not in locals():
                if EV[-3:] == 'npy':
                    EventCurve = NP.load('%s%s' % (EventDir, EV))
                else:
                    EventCurve = NP.genfromtxt('%s%s' % (EventDir, EV))
                RefTrajs = [elem for elem in NP.unique(EventCurve[:,1])]
            else:
                if EV[-3:] == 'npy':
                    temp = NP.load('%s%s' % (EventDir, EV))
                else:
                    temp = NP.genfromtxt('%s%s' % (EventDir, EV))
                for Ref in NP.unique(temp[:,1]):
                    if RefTrajs.count(Ref) == 0:
                        EventCurve = NP.concatenate( (EventCurve, temp[temp[:,1]==Ref]) )
        return EventCurve

#######################################################################

    def _Initialize_DiffTrajAttributes(self, ColDict):
        """
v06.05.17
Helper Function: This (private) bound method modifies the EventCurve for SameTraj is None:
- if EventNames has a single entry:
    a) the EventCurve is loaded 
    b) only the rows corresponding between self.StartFrame - self.EndingFrame are extracted
- if EventNames have multiple entries:
    a) merge first all EventCurves with same ROW_TrajNrList but different COL_TrajNrList specified in the names
    b) merge then all remaining EventCurves which corresponds to merging different Rows with same Columns

INPUT:
    ColDict                         : {DEFAULTDICT}  stores the _RowsX and _ColsY information from the self.EventNames-Strings,
                                                        Keys = Rows/Cols traj numbers | Values = Position in the self.EventNames-List;
    self.EventDir                   : {STRING}       Directory, where the Norm_EventCurves.txt are stored;
    self.EventNames                 : {STRING/LIST}  Name of the EventCurve, can also be a list to combine different _RowX, _ColY files |
                                                     if SameTraj is not None, submit multiple simulation times of SAME trajectory,
                                                     to calculate the overlap between different time parts of the SAME trajectory | 
                                                     e.g. 'V3_S1-S10_0-500_noWeight.npy';
    self.SameTraj                   : {INT}          if SameTraj is not None, different parts of the SameTraj-th trajectory 
                                                        are compared returning different simulation time parts ;
    self.AttrDict['TrajNameList']   : {LIST}         List of used trajectory names, WITHOUT ENDING;
    self.AttrDict['ThresholdList']  : {FLOAT-LIST}   Threshold parameter list used to calculate the EventCurves;
    self.AttrDict['StartFrame']     : {INT-LIST}     Starting frame for each EventCurve, if SameTraj==None, all StartFrame are equal;
    self.AttrDict['EndingFrame']    : {INT-LIST}     Ending frame for each EventCurve, if SameTraj==None, all EndingFrame are equal;
OUTPUT:
    return EventCurve, NormMatrix, COL_TrajNrList   
        """
        if isinstance(self.EventNames, list):
      ##########  
            if len(ColDict.keys()) == 1 and ColDict.keys()[0] == '':
                COL_TrajNrList = None
            elif len(ColDict.keys()) > 1 and ColDict.keys().count('') > 0:
                raise ValueError('You try to merge EventCurves containing all trajectories\n\t(no >_ColX<)\n and '+\
                                 'EventCurves containing only some trajectories\n\t(>_ColX<).\n This might '+\
                                 'be an error, check your input!\n\tEventNames = {}'.format(self.EventNames))
            else:
                COL_TrajNrList = [int(elem) for elem in '_'.join([elem2 for elem2 in ColDict.keys()]).split('_')]
          ##########
            EventCurve_List = [Overlap.Merge_Rows(self.EventDir, [self.EventNames[elem] for elem in ColM]) \
                           for ColM in ColDict.values()]
            EventCurve, _ = Overlap.Merge_Cols(EventCurve_List, len(self.AttrDict['ThresholdList']), COL_TrajNrList)
            NormMatrix_List = [NP.reshape(NP.genfromtxt('%s%s' % (self.EventDir, 
                                                                  Overlap.Extract_NormName(self.EventNames[ColM[0]]))), (1,-1)) \
                               for ColM in ColDict.values()]
            NormMatrix, COL_TrajNrList = Overlap.Merge_Cols(NormMatrix_List, len(self.AttrDict['ThresholdList']), COL_TrajNrList)
    #######################
        else:
            #---- define COL_TrajNrList
            if self.EventNames.find('_Col') == -1:
                COL_TrajNrList = None
            else:
                COL_TrajNrList = [int(elem) for elem in self.EventNames.split('_Col')[1].split('.')[0].split('_')]
            #---- define NormName v23.06.16
            NormNames = Overlap.Extract_NormName(self.EventNames)
            #---- load EventCurves
            NormMatrix = NP.genfromtxt('%s%s' % (self.EventDir, NormNames))
            NormMatrix = NP.reshape(NormMatrix, (1,len(NormMatrix)))
            if self.EventNames[-3:] == 'npy':
                EventCurve = NP.load('%s%s' % (self.EventDir, self.EventNames))
            else:
                EventCurve = NP.genfromtxt('%s%s' % (self.EventDir, self.EventNames))
    #---- MODIFY EventCurves (17.06.16): extract only StartFrame to EndingFrame of all trajectories
        if NP.sum(self.AttrDict['updated_TrajLengthList']) != len(EventCurve[:,0]):
            temp = NP.copy(EventCurve)
            EventCurve = NP.zeros( (NP.sum(self.AttrDict['updated_TrajLengthList']), len(temp[0,:])) )
            Unique_Events = NP.unique(temp[:,1])
            EvInd = 0; 
            StEndInd = 0;
            for TTs in Unique_Events:
                EventCurve[EvInd:(EvInd+len(temp[temp[:,1] == TTs][self.AttrDict['StartFrame'][StEndInd]:self.AttrDict['EndingFrame'][StEndInd],0])),:] = \
                        temp[temp[:,1] == TTs][self.AttrDict['StartFrame'][StEndInd]:self.AttrDict['EndingFrame'][StEndInd],:]
                EvInd += len(temp[temp[:,1] == TTs][self.AttrDict['StartFrame'][StEndInd]:self.AttrDict['EndingFrame'][StEndInd],0])
                StEndInd += 1
        return EventCurve, NormMatrix, COL_TrajNrList

#######################################################################

    def _Initialize_OverlapMatrix(self):
        """
v06.05.17        
Helper Function: This (private) bound method initialize the OverlapMatrix 
- init OverlapMatrix: GroupNr | Threshold | (densO | confO | TotalNr) x len(CompareList)
    - GroupNr: projection on the GroupNr-th trajectory/ies
    - densO   : sum of min()/max() ratios, NO AVERAGING
    - densStd : standard error of min()/max() values
    - confO   : Nr of overlapping frames
    - TotalNr: total number of involved frames
- to extract the density overlap & conformational overlap (to the corresp. GroupNr), divide densO & confO by TotalNr

INPUT:
    self.AttrDict['ThresholdList']  : {FLOAT-LIST}  Threshold parameter list used to calculate the EventCurves;
    self.CompareList                : {LIST,TUPLE,LIST} list of groups of trajectories, which are compared and the overlap is calculated
                                            e.g. [([1],[3]), ([1,2],[3,4]), ([1,2],[3])] calculates the overlap between trajectories
                                                a) (1)vs(3) 
                                                b) (1+2)vs(3+4)
                                                c) (1+2)vs(3)
                                            and stores them in consecutive columns;
OUTPUT:
    return OverlapMatrix
        """
        OverlapMatrix = NP.zeros( (len(self.AttrDict['ThresholdList'])*len(self.CompareList[0]), 1+1+4*len(self.CompareList)) )-1
        OverlapMatrix[:,0] = range(1,len(self.CompareList[0])+1)*len(self.AttrDict['ThresholdList'])
        OverlapMatrix[:,1] = NP.concatenate([[elem]*len(self.CompareList[0]) for elem in self.AttrDict['ThresholdList']])
        return OverlapMatrix

#######################################################################

    def _Extract_AttributesFromFile(self, NormNameList):
        """
v26.09.16
Helper Function: This (private) bound method extracts the Header and Parameters from the EventCurve(s), which are necessary for the Overlap calculation
- since EventCurves are stored in .npy python-binary-format, the information is extracted from Norm_EventCurves.txt
- if len(EventNamesList) == 1: same parts with same parameters are used but different simulation times
- if len(EventNamesList) >  1: the overlap of different parts of THE SAME TRAJECTORY is calculated
INPUT:
    NormNameList           : {LIST}   List of names of Norm_EventCurves.txt, which are used for extraction
                                    if len() == 1, using same configurations/parts but different simulation times
                                    if len() >  1, overlap is calculated for ONE TRAJECTORY but different parts;
    self.EventDir          : {STRING} Directory, where the Norm_EventCurves.txt are stored;
    self.SameTraj          : {INT}    <default None> if SameTraj is not None, different parts of the SameTraj-th trajectory 
                                        are compared returning different simulation time parts ;
OUTPUT:
    self.AttrDict['TrajNameList']           : {LIST}        List of used trajectory names, WITHOUT ENDING;
    self.AttrDict['StartFrame']             : {INT-LIST}    Starting frame for each EventCurve, if SameTraj==None, all StartFrame are equal;
    self.AttrDict['EndingFrame']            : {INT-LIST}    Ending frame for each EventCurve, if SameTraj==None, all EndingFrame are equal;
    self.AttrDict['Iterations']             : {INT}         Number of Iterations for mean-field (MF) reweighting of aMD trajectories;
    self.AttrDict['aMD_reweight']           : {STRING}      re-weighting algorithm, e.g. MF, McL or Exp;
    self.AttrDict['Lambda']                 : {FLOAT}       re-scaling factor for the sMD reweighting scheme;
    self.AttrDict['Order']                  : {INT}         Order of the MacLaurin expansion reweighting of aMD trajectories;
    self.AttrDict['ThresholdList']          : {FLOAT-LIST}  Threshold parameter list used to calculate the EventCurves;
    self.AttrDict['updated_TrajLengthList'] : {INT-LIST}    Lenghts of each trajectory [in Frames] with respect to the simulation time parts;
        """
        AttrDict = {}
        AttrDict['TrajNameList']           = []
        AttrDict['StartFrame']             = []
        AttrDict['EndingFrame']            = []
        AttrDict['updated_TrajLengthList'] = []

      ## NormNames must be stored in a List
        if isinstance(NormNameList, list):
            LoopList = NormNameList
        else:
            LoopList = [NormNameList]
      ## Extract all relevant Attributes from the file NormMatrix
        for NormName in LoopList:
            with open('%s%s' % (self.EventDir, NormName), 'r') as INPUT:
                for line in INPUT:
                    if len(line.split()) > 2 and line.split()[0] == '#':
              #-------- Abort Criterion
                        if line.split()[1] == 'each':
                            break
              #-------- AttrDict['TrajNameList']
                        elif line.split()[1] == 'TrajNameList' and AttrDict['TrajNameList'] == []:
                            if self.SameTraj is None:
                                AttrDict['TrajNameList'] = [elem.replace('\'','').replace(',','') \
                                            for elem in line[line.find('[')+1:line.find(']')].split()]
                            elif self.SameTraj is not None:
                                AttrDict['TrajNameList'].append([elem.replace('\'','').replace(',','') \
                                            for elem in line[line.find('[')+1:line.find(']')].split()][self.SameTraj-1])
              #-------- AttrDict['ThresholdList']
                        elif line.split()[1] == 'ThresholdList' and not AttrDict.has_key('ThresholdList'):
                            AttrDict['ThresholdList'] = [float(elem.replace(',','')) for elem in line[line.find('[')+1:line.find(']')].split()]
              #-------- AttrDict['StartFrame']
                        elif line.split()[1] == 'StartFrame':
                            if (self.SameTraj is not None) or (len(AttrDict['TrajNameList']) == len(LoopList)):
                                AttrDict['StartFrame'].append(int(line.split()[-1]))
                            elif AttrDict['StartFrame'] == []:
                                AttrDict['StartFrame'] = [int(line.split()[-1]) for elem in AttrDict['TrajNameList']]
              #-------- AttrDict['EndingFrame']
                        elif line.split()[1] == 'EndingFrame':
                            if (self.SameTraj is not None) or (len(AttrDict['TrajNameList']) == len(LoopList)):
                                AttrDict['EndingFrame'].append(int(line.split()[-1]))
                            elif AttrDict['EndingFrame'] == []:
                                AttrDict['EndingFrame'] = [int(line.split()[-1]) if line.split()[-1] != 'inf' \
                                                                         else NP.infty for elem in AttrDict['TrajNameList']]
              #-------- AttrDict['Iterations']
                        elif line.split()[1] == 'Weight' and dir(self).count('Iterations') == 0:
                            AttrDict['Iterations'] = [int(line.split()[-1]) for elem in AttrDict['TrajNameList']]
              #-------- AttrDict['aMD_reweight']
                        elif line.split()[1] == 'aMD_reweight' and dir(self).count('aMD_reweight') == 0:
                            AttrDict['aMD_reweight'] = (line.split()[-1])
              #-------- AttrDict['Lambda']
                        elif line.split()[1] == 'Lambda' and dir(self).count('Lambda') == 0:
                            AttrDict['Lambda'] = float(line.split()[-1])
              #-------- AttrDict['Order']
                        elif line.split()[1] == 'Order' and dir(self).count('Order') == 0:
                            AttrDict['Order'] = int(line.split()[-1])
              #-------- AttrDict['updated_TrajLengthList']
                        elif line.split()[1] == 'updated_TrajLengthList':
                            if self.SameTraj is None and AttrDict['updated_TrajLengthList'] == []:
                                AttrDict['updated_TrajLengthList'] = [int(elem.replace(',','')) \
                                             for elem in line[line.find('[')+1:line.find(']')].split()]
                            elif self.SameTraj is not None:
                                AttrDict['updated_TrajLengthList'].append([int(elem.replace(',','')) \
                                             for elem in line[line.find('[')+1:line.find(']')].split()][self.SameTraj-1])
      ## CHECK, if every parameter could be loaded from the Norm_EventCurves.txt
        Attributes = set(AttrDict.keys())
        Params = set(['TrajNameList', 'ThresholdList', 'StartFrame', 'EndingFrame', 
                      'Iterations', 'Lambda', 'Order', 'aMD_reweight', 'updated_TrajLengthList'])
        Missing = Params - Attributes
        if len(Missing) != 0:
                raise ValueError('The parameters <%s> are not specified in %s! The parameters could not be loaded. Check that <%s> are specified in %s' % \
                                    ([elem if elem != 'Iteration' else 'Weight Iterations' for elem in Missing], LoopList,
                                     [elem if elem != 'Iteration' else 'Weight Iterations' for elem in Missing], LoopList))
        return AttrDict

#######################################################################

    def _Initialize_Header(self):
        """
v06.05.17
- This bound method initialize the Header, which is used for the Overlap-Text-File, using all Class.Attributes
- it gives all information and descriptions about the Overlap-File
- to be able to set a calculation time, a format placeholder is used '{}' which is set in Store(Time)

INPUT:
    Time
    self.EventDir                           : {STRING}      Directory, where the Norm_EventCurves.txt are stored;
    self.EventNames                         : {STRING/LIST} Name of the EventCurve, can also be a list to combine different _RowX, _ColY files |
                                                            if SameTraj is not None, submit multiple simulation times of SAME trajectory,
                                                            to calculate the overlap between different time parts of the SAME trajectory | 
                                                            e.g. 'V3_S1-S10_0-500_noWeight.npy';
    self.CompareList                        : {LIST,TUPLE,LIST} list of groups of trajectories, which are compared and the overlap is calculated
                                                                e.g. [([1],[3]), ([1,2],[3,4]), ([1,2],[3])] calculates the overlap between trajectories
                                                                    a) (1)vs(3) 
                                                                    b) (1+2)vs(3+4)
                                                                    c) (1+2)vs(3)
                                                                and stores them in consecutive columns;
    self.COL_TrajNrList                     : {INT-LIST}    defines which trajectories are stored in the columns of the eventcurve;
    self.aMD_Nrs                            : {INT-LIST}    FOR RE-WEIGHTING ONLY, trajectory numbers which are generated with aMD,
                                                            numbering MUST correspond to the trajectories stored in ClusterFile under "TrajNameList = [...]",
                                                                e.g. [1,3,5] means trajNr 1,3,5 of TrajNameList are aMD trajectories | 
                                                                     has to correspond to possible _ColY EventCurves;
    self.sMD_Nrs                            : {INT-LIST}    FOR RE-WEIGHTING ONLY, trajectory numbers which are generated with scaledMD, 
                                                            numbering MUST correspond to the trajectories stored in ClusterFile under "TrajNameList = [...]",
                                                                e.g. [1,3,5] means trajNr 1,3,5 of TrajNameList are scaledMD trajectories 
                                                                     | has to correspond to possible _ColY EventCurves;
    self.AttrDict['TrajNameList']           : {LIST}        List of used trajectory names, WITHOUT ENDING;
    self.AttrDict['ThresholdList']          : {FLOAT-LIST}  Threshold parameter list used to calculate the EventCurves;
    self.AttrDict['StartFrame']             : {INT-LIST}    Starting frame for each EventCurve, if SameTraj==None, all StartFrame are equal;
    self.AttrDict['EndingFrame']            : {INT-LIST}    Ending frame for each EventCurve, if SameTraj==None, all EndingFrame are equal;
    self.AttrDict['Iterations']             : {INT}         Number of Iterations for mean-field (MF) reweighting of aMD trajectories;
    self.AttrDict['updated_TrajLengthList'] : {INT-LIST}    Lenghts of each trajectory [in Frames] with respect to the simulation time parts;
    self.AttrDict['aMD_reweight']           : {STRING}      re-weighting algorithm, e.g. MF, McL or Exp;
    self.AttrDict['Lambda']                 : {FLOAT}       re-scaling factor for the sMD reweighting scheme;
    self.AttrDict['Order']                  : {INT}         Order of the MacLaurin expansion reweighting of aMD trajectories;
OUTPUT:
    return Header
       """
      ########
      ## HEADER initialization
      ########      
        Header =  'this file contains the <density overlap | conformational overlap | total # frames> '+\
                  'for the different CompareList \n\n'+\
                  'ensure, that the columns and the submitted CompareList do match\n\n'+\
                  'INPUT:\n'+\
                      '\tEventDir               = {}\n'.format(self.EventDir)+\
                      '\tEventNames             = {}\n'.format(self.EventNames)+\
                      '\tCompareList            = {}\n'.format(self.CompareList)+\
                      '\tCOL_TrajNrList         = {}\n'.format(self.COL_TrajNrList)+\
                      '\taMD_Nrs                = {}\n'.format(self.aMD_Nrs)+\
                      '\tsMD_Nrs                = {}\n'.format(self.sMD_Nrs)+\
                      '\tTrajNameList           = {}\n'.format(self.AttrDict['TrajNameList'])+\
                      '\tThresholdList          = {}\n'.format(self.AttrDict['ThresholdList'])+\
                      '\tStartFrame             = {}\n'.format(self.AttrDict['StartFrame'])+\
                      '\tEndingFrame            = {}\n'.format(self.AttrDict['EndingFrame'])+\
                      '\taMD/sMD MF Iterations  = {}\n'.format(self.AttrDict['Iterations'])+\
                      '\tupdated_TrajLengthList = {}\n'.format(self.AttrDict['updated_TrajLengthList'])+\
                      '\taMD_reweight           = {}\n'.format(self.AttrDict['aMD_reweight'])+\
                      '\tLambda                 = {}\n'.format(self.AttrDict['Lambda'])+\
                      '\tOrder                  = {}\n'.format(self.AttrDict['Order'])+\
                   '{}\n\n'
        Header = Header + """
    densO has to be averaged for different trajectory Nr projections
    Err   is the standard error of the mean, which is taken for the weighted average calculation for different trajectory projections
    confO means the number of frames which have at least 1 event for each trajectory group
    TotFrames means the total number of frames of the certain trajectory group
    confO and TotFrames have to be summed for different trajectory Nr projections
        and then divided to obtain the real conformational overlap value
    GroupNr monitors the number of the comparing group of [X]vs[Y]vs... (K of O(K,L:r)) meaning that
        the overlap value corresponds to the reference trajectory (set) of the K-th group

GroupNr | Threshold | [densO|Err|confO|TotFrames] of """
        for Sep in self.CompareList:
            Header = Header + '%s | ' % ('vs.'.join(['%s' % elem for elem in Sep]))

        return Header 

#######################################################################

    def Check_SaveName(self):
        """
v06.05.17
This bound method checks, whether the overlap-file already exists, checking self.SaveDir and self.SaveName
INPUT:
    self.SaveDir             : {STRING}    storing directory, e.g. 'Overlap/';
    self.SaveName            : {STRING}    savename OverlapMatrix, e.g. 'Overlap_R5_Pairs_0-10000_noWeight.txt';
OUTPUT:
    return os.path.exists('%s%s' % (self.SaveDir, self.SaveName))
        """
        if os.path.exists('%s%s' % (self.SaveDir, self.SaveName)):
            print 'The overlap file already exists!\n\tSaveDir = %s\n\tSaveName = %s' % \
                      (self.SaveDir, self.SaveName)
            return True
        else:
            return False

    def _Check_EventNames(self):
        """
v06.05.17
Helper function: This (private) bound method checks, whether EventNames are properly set and do exist
- returns a defaultdict, which stores the information
    1) (Keys)   which ROW_TrajNrList and which COL_TrajNrList are used
    2) (Values) corresponding position in the EventNames-list
INPUT:
    self.EventDir   : {STRING}      Directory, where the Norm_EventCurves.txt are stored;
    self.EventNames : {STRING/LIST} Name of the EventCurve, can also be a list to combine different _RowX, _ColY files |
                                    if SameTraj is not None, submit multiple simulation times of SAME trajectory,
                                    to calculate the overlap between different time parts of the SAME trajectory | 
                                    e.g. 'V3_S1-S10_0-500_noWeight.npy';
    self.SameTraj   : {INT}         if SameTraj is not None, different parts of the SameTraj-th trajectory 
                                      are compared returning different simulation time parts;
OUTPUT:
    return ColDict  : {DEFAULTDICT} stores the _RowsX and _ColsY information from the self.EventNames-Strings,
                                      Keys = Rows/Cols traj numbers | Values = Position in the self.EventNames-List;
        """
        if isinstance(self.EventNames, list):
            ColDict = defaultdict(list); 
            Index = 0;
            EV_Prefix = []
            for Name in self.EventNames:
                NormName = Overlap.Extract_NormName(Name)
                if not os.path.exists('%s%s' % (self.EventDir, Name)) or \
                   not os.path.exists('%s%s' % (self.EventDir, NormName)):
                    raise NameError('EventCurve/NormMatrix does not exist\n\t%s\n\t%s' % (self.EventDir, Name))
                #--- CHECK if EventNames have the same simulation times
                if Name.find('_Row') != -1:
                    EV_Prefix.append(Name.split('_Row')[0])
                elif Name.find('_Col') != -1:
                    EV_Prefix.append(Name.split('_Col')[0])
                else:
                    EV_Prefix.append(Name)
            #--- Generate ColDict
                if Name.find('_Col') != -1:
                    ColDict['%s' % Name.split('_Col')[1].split('.')[0]].append(Index)
                else:
                    ColDict[''].append(Index)
                Index += 1
          ##########
            if len(NP.unique(EV_Prefix)) > 1:
                raise NameError('You try to merge EventCurves with different simulation times, '+\
                                'this might be an error. Check your Input!\n\tunique EV_Prefix = %s' % \
                                       NP.unique(EV_Prefix))
          ##########  
            elif len(ColDict.keys()) > 1 and ColDict.keys().count('') > 0:
                raise ValueError('You try to merge EventCurves containing all trajectories\n\t(no >_ColX<)\n and '+\
                                 'EventCurves containing only some trajectories\n\t(>_ColX<).\n This might '+\
                                 'be an error, check your input!\n\tEventNames = {}'.format(self.EventNames))
        else:
            ColDict = None
            if self.SameTraj is not None:
                raise ValueError('SameTraj (=%s) is not None! Multiple EventNames (=%s) are necessary!' % \
                                    (self.SameTraj, self.EventNames) )
            NormName = Overlap.Extract_NormName(self.EventNames)
            if not os.path.exists('%s%s' % (self.EventDir, self.EventNames)) or \
               not os.path.exists('%s%s' % (self.EventDir, NormName)):
                raise NameError('EventCurve/NormMatrix does not exist\n\t%s\n\t%s' % (self.EventDir, self.EventNames))
        return ColDict

    def _Check_CompareList(self):
        """
v06.05.17
Helper function: This (private) bound method checks, whether the submitted CompareList does math the
present trajectory numbers in <EventCurves>
- returns Unique_Compare: all (unique) trajectories which are specified in CompareList
INPUT:
    self.EventDir    : {STRING}      Directory, where the Norm_EventCurves.txt are stored;
    self.EventNames  : {STRING/LIST} Name of the EventCurve, can also be a list to combine different _RowX, _ColY files |
                                     if SameTraj is not None, submit multiple simulation times of SAME trajectory,
                                     to calculate the overlap between different time parts of the SAME trajectory | 
                                     e.g. 'V3_S1-S10_0-500_noWeight.npy';
    self.CompareList : {LIST,TUPLE,LIST} list of groups of trajectories, which are compared and the overlap is calculated
                              e.g. [([1],[3]), ([1,2],[3,4]), ([1,2],[3])] calculates the overlap between trajectories
                                  a) (1)vs(3) 
                                  b) (1+2)vs(3+4)
                                  c) (1+2)vs(3)
                              and stores them in consecutive columns;
    self.EventCurve  : {NP.array}        events stored for different Thresholds and trajectories in one NP.array;
OUTPUT:
    return Unique_Compare
        """
      #---- the trajs in CompareList have to match the TrajNr-windows of EventCurve
        Unique_Compare = []
        for First in self.CompareList:
            for Second in First:
                Unique_Compare.extend(Second)
        Unique_Compare = [int(elem) for elem in NP.unique(Unique_Compare)]
        Unique_Events = NP.unique(self.EventCurve[:,1])
        if len(NP.unique(NP.concatenate( (Unique_Compare, Unique_Events) ))) != len(Unique_Events):
            raise ValueError('submitted CompareList does not match the present trajectory numbers in <EventCurves>'+\
             ('check your input:\n\tCompareList = %s\n\tEventCurve = %s%s' % \
                (self.CompareList, self.EventDir, self.EventNames)))
        return Unique_Compare

    def _Check_aMDsMD_Nrs(self, Unique_Compare):
        """
v06.05.17
Helper function: This (private) bound method checks/modifies aMD_Nrs & sMD_Nrs to match the submitted CompareList

INPUT:
    self.EventNames : {STRING/LIST} Name of the EventCurve, can also be a list to combine different _RowX, _ColY files |
                                    if SameTraj is not None, submit multiple simulation times of SAME trajectory,
                                    to calculate the overlap between different time parts of the SAME trajectory | 
                                    e.g. 'V3_S1-S10_0-500_noWeight.npy';
    self.SameTraj   : {INT}         if SameTraj is not None, different parts of the SameTraj-th trajectory 
                                      are compared returning different simulation time parts;
    aMD_Nrs         : {INT-LIST}  <default []>,   FOR RE-WEIGHTING ONLY, trajectory numbers which are generated with aMD,
                                                  numbering MUST correspond to the trajectories stored in ClusterFile under "TrajNameList = [...]",
                                        e.g. [1,3,5] means trajNr 1,3,5 of TrajNameList are aMD trajectories | has to correspond to possible _ColY EventCurves;
    sMD_Nrs         : {INT-LIST}  <default []>,   FOR RE-WEIGHTING ONLY, trajectory numbers which are generated with scaledMD, 
                                                  numbering MUST correspond to the trajectories stored in ClusterFile under "TrajNameList = [...]",
                                        e.g. [1,3,5] means trajNr 1,3,5 of TrajNameList are scaledMD trajectories | has to correspond to possible _ColY EventCurves;
OUTPUT:
    modified self.aMD_Nrs, self.sMD_Nrs
        """
        #--- modify aMD_Nrs/sMD_Nrs: because one uses only ONE trajectory, if it is accelerated, then all trajs are
        #    additionally: trajectories are counted consecutively 1,2,3,...,len(EventNames)
        if self.SameTraj is not None:
            if self.aMD_Nrs.count(self.SameTraj) > 0:
                self.aMD_Nrs = [elem+1 for elem in xrange(len(self.EventNames))]
            else:
                self.aMD_Nrs = []
            if self.sMD_Nrs.count(self.SameTraj) > 0:
                self.sMD_Nrs = [elem+1 for elem in xrange(len(self.EventNames))]
            else:
                self.sMD_Nrs = []
#---- MODIFY aMD_Nrs and sMD_Nrs, that they match CompareList
        else:
            self.aMD_Nrs = [elem for elem in self.aMD_Nrs if Unique_Compare.count(elem) > 0]
            self.sMD_Nrs = [elem for elem in self.sMD_Nrs if Unique_Compare.count(elem) > 0]

    def _Check_aMDWeights(self):
        """
v06.05.17
Helper function: This (private) bound method checks whether the Weights do exist for the initialized aMD_Nrs
INPUT:
    aMD_Nrs                        : {INT-LIST}  FOR RE-WEIGHTING ONLY, trajectory numbers which are generated with aMD,
                                         numbering MUST correspond to the trajectories stored in ClusterFile under "TrajNameList = [...]",
                                        e.g. [1,3,5] means trajNr 1,3,5 of TrajNameList are aMD trajectories | has to correspond to possible _ColY EventCurves;
    self.WeightDir                 : {STRING}    FOR RE-WEIGHTING ONLY, Directory, where the aMD/sMD weights are located, e.g. 'EventCurves/Weights/';
    self.AttrDict['TrajNameList']  : {LIST}      List of used trajectory names, WITHOUT ENDING;
    self.AttrDict['StartFrame']    : {INT-LIST}  Starting frame for each EventCurve, if SameTraj==None, all StartFrame are equal;
    self.AttrDict['EndingFrame']   : {INT-LIST}  Ending frame for each EventCurve, if SameTraj==None, all EndingFrame are equal;
    self.AttrDict['Iterations']    : {INT}       Number of Iterations for mean-field (MF) reweighting of aMD trajectories;
    self.AttrDict['aMD_reweight']  : {STRING}    re-weighting algorithm, e.g. MF, McL or Exp;
OUTPUT:
        """
        if self.aMD_Nrs != []:
            for aA in self.aMD_Nrs:
                if self.AttrDict['aMD_reweight'] == 'MF':
                    if not os.path.exists('%saMD_Weight_MF_%s_%s-%s_Iter%s.txt' % \
                                           (self.WeightDir, self.AttrDict['TrajNameList'][aA-1], self.AttrDict['StartFrame'][aA-1], 
                                            self.AttrDict['EndingFrame'][aA-1], self.AttrDict['Iterations'][aA-1])):
                        raise NameError(('The Weight-Vector for %s does not exist, please check your input' % \
                                            self.AttrDict['TrajNameList'][aA-1])+\
                                        ('\n\taMD trajectories = {}'.format(self.aMD_Nrs))+\
                                        ('\n\t%saMD_Weight_MF_%s_%s-%s_Iter%s.txt' % \
                                           (self.WeightDir, self.AttrDict['TrajNameList'][aA-1], self.AttrDict['StartFrame'][aA-1], 
                                            self.AttrDict['EndingFrame'][aA-1], self.AttrDict['Iterations'][aA-1])))
                else:
                    if not os.path.exists('%saMD_Weight_%s.txt' % (self.WeightDir, self.AttrDict['TrajNameList'][aA-1])):
                        raise NameError(('The Weight-Vector for %s does not exist, please check your input' % \
                                            self.AttrDict['TrajNameList'][aA-1])+\
                                        ('\n\taMD trajectories = {}'.format(self.aMD_Nrs))+\
                                        ('\n\t%saMD_Weight_%s.txt' % (self.WeightDir, self.AttrDict['TrajNameList'][aA-1])))
    def _Check_sMDWeights(self):
        """
v06.05.17
Helper function: This (private) bound method checks whether the Weights do exist for the initialized aMD_Nrs
INPUT:
    aMD_Nrs                        : {INT-LIST}  FOR RE-WEIGHTING ONLY, trajectory numbers which are generated with aMD,
                                         numbering MUST correspond to the trajectories stored in ClusterFile under "TrajNameList = [...]",
                                        e.g. [1,3,5] means trajNr 1,3,5 of TrajNameList are aMD trajectories | has to correspond to possible _ColY EventCurves;
    self.WeightDir                 : {STRING}    FOR RE-WEIGHTING ONLY, Directory, where the aMD/sMD weights are located, e.g. 'EventCurves/Weights/';
    self.AttrDict['TrajNameList']  : {LIST}      List of used trajectory names, WITHOUT ENDING;
    self.AttrDict['StartFrame']    : {INT-LIST}  Starting frame for each EventCurve, if SameTraj==None, all StartFrame are equal;
    self.AttrDict['EndingFrame']   : {INT-LIST}  Ending frame for each EventCurve, if SameTraj==None, all EndingFrame are equal;
    self.AttrDict['Iterations']    : {INT}       Number of Iterations for mean-field (MF) reweighting of aMD trajectories;
    self.AttrDict['Lambda']        : {FLOAT}     re-scaling factor for the sMD reweighting scheme;
OUTPUT:
        """
        if self.sMD_Nrs != []:
            for sS in self.sMD_Nrs:
                if not os.path.exists('%ssMD_Weight_lambda%s_%s_%s-%s_Iter%s.txt' % \
                               (self.WeightDir, self.AttrDict['Lambda'], self.AttrDict['TrajNameList'][sS-1], self.AttrDict['StartFrame'][sS-1], 
                                self.AttrDict['EndingFrame'][sS-1], self.AttrDict['Iterations'][sS-1])):
                    raise NameError(('The Weight-Vector for %s does not exist, please check your input' % \
                                            self.AttrDict['TrajNameList'][aA-1])+\
                                    ('\n\taMD trajectories = {}'.format(self.aMD_Nrs))+\
                                    ('\n\t%ssMD_Weight_lambda%s_%s_%s-%s_Iter%s.txt' % \
                                       (self.WeightDir, self.AttrDict['Lambda'], self.AttrDict['TrajNameList'][sS-1], self.AttrDict['StartFrame'][sS-1], 
                                        self.AttrDict['EndingFrame'][sS-1], self.AttrDict['Iterations'][sS-1])))

#####################################################################################

    @staticmethod
    def Extract_NormName(EventName):
        """
    v23.06.16
    This function returns the corresponding Norm-file of the submitted EventCurve
    - if EventCurves are splitted into different Rows, meaning that they store only a certain amount of reference trajectories
        only one Norm-file is necessary, because it stores the total number of frames for every Column, meaning the
        trajectories for which the Events are counted
    - if EventCurves are splitted into different Cols, meaning trajectories for which the Events are counted, every EventCurve
        has its own Norm-file
    INPUT:
        EventName   : {STRING} name of the corresponding EventCurve [.npy format];
    OUTPUT:    
        NormName    : {STRING} returns the corresponding Norm-file Name [.txt format];
        """
    #---- define NormName
        if EventName.find('Col') == -1:
            if EventName.find('Row') == -1:
                NormName = 'Norm_'+EventName.replace('.npy','.txt')
            else:
                NormName = 'Norm_'+EventName.split('_Row')[0]+'.txt'
        else:
            if EventName.find('Row') == -1:
                NormName = 'Norm_'+EventName.split('_Col')[0]+'_Col'+EventName.split('Col')[1].replace('.npy','.txt')
            else:
                NormName = 'Norm_'+EventName.split('_Row')[0]+'_Col'+EventName.split('Col')[1].replace('.npy','.txt')
        #----
        return NormName


