#######################################################
#######################################################
# Released under the GNU General Public License version 3 by Mike Nemec
#
# Copyright (c) 2016: Mike Nemec
#
# PySampling: Python scripts to assess the sampling quality of MD simulations 
#               using a multi-trajectory overlap approach
#
# Author:     Mike Nemec <mike.nemec@uni-due.de>
#
# current version: v29.11.16-1
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
import os, sys
import subprocess as SB
#import matplotlib.pyplot as plt
import numpy as NP
import scipy.misc
import time
import string
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from multiprocessing import Pool
from argparse import ArgumentParser
## String to literal
import ast
## Simpson integration
from scipy.integrate import simps
## linear regression
from scipy.stats import linregress as LR
## T-test, critical values confidence interval
from scipy.stats import t as TPPF

#######################################################
#------------------------------------------------------
#---          OVERLAP CALCULATION
#------------------------------------------------------
#######################################################
def Calc_Overlap(EventDir, EventNames, SaveDir, SaveName, CompareList, WeightDir=None, aMD_Nrs=[], sMD_Nrs=[], SameTraj=None):
    """ 
v28.11.16
- calculates <conformational overlap> and <density overlap> : Overlap between different trajectories/groups 
  for different Threshold and reference Trajectories
- CompareList has to match the TrajNr in the EventCurves: the overlap is than calculated between the sets of trajectory numbers defined in CompareList
- EventNames can be splitted into different reference trajectories [_RowX] or different trajectories for which the 
    Events are counted for [_ColX]
- currently: (1) EventCurves with different reference trajectories [_RowX] are combined according to the submitted order
                 of EventNames, e.g. = [EventCurve_Row1_2_3.npy, EventCurve_Row4_5_6.npy] are merged to one 
                                       corresponding EventCurve containing all 6 reference TrajNrs
             (2) EventCurves with different trajectories, for which the Events are counted for [_ColX] are ALSO merged, 
                 defining (sorted) COL_TrajNrList (automatically by the EventNames name) 
                     e.g. = [EventCurve_Col2_3.npy, EventCurve_Col3_52_65.npy] are merged to one _Col2_3_52_45
                 -> COL_TrajNrList is detected by the EventNames automatically, 
                    defining which column represents which trajectory (number)
- error handling: Odens: storing standard deviation of the min/max ratios for each reference trajectory
- overlap values are stored for every K of O(K,L;r) separately, which is called group here
- stores the overlap in a text-file
        GroupNr | Threshold | [densO | densStd | confO | TotFrames] | [...] 

INPUT:
    EventDir       : {STRING}          Directory, where the EventCurves are stored, e.g. 'EventCurves/';
    EventNames     : {STRING/LIST}     Name of the EventCurve, can also be a list to combine different _RowX, _ColY files |
                                        if SameTraj is not None, submit multiple simulation times of SAME trajectory,
                                        to calculate the overlap between different time parts of the SAME trajectory | 
                                        e.g. 'V3_S1-S10_0-500_noWeight.npy';
    SaveDir        : {STRING}          storing directory, e.g. 'Overlap/';
    SaveName       : {STRING}          savename PREFIX for the overlap, 
                            e.g. 'Overlap_R5_Pairs' -> 'Overlap_R5_Pairs_StartFrame-EndingFrame_reweight.txt',
                            if SameTraj == True, then savename is the COMPLETE savename;
    CompareList    : {LIST,TUPLE,LIST} list of groups of trajectories, which are compared and the overlap is calculated
                        e.g. [([1],[3]), ([1,2],[3,4]), ([1,2],[3])] for (1)vs(3) AND (1+2)vs(3+4) AND (1+2)vs(3);
    WeightDir      : {STRING}          <default None>, FOR RE-WEIGHTING ONLY, directory, where the Weights are located for aMD OR sMD 
                                        trajectory re-weighting;
    aMD_Nrs        : {INT-LIST}        <default []>, FOR RE-WEIGHTING ONLY, trajectory numbers which are generated with aMD,
                                        e.g. [1,3,5] means trajNr 1,3,5 of TrajNameList are aMD trajectories | has to correspond to possible _ColY EventCurves;
    sMD_Nrs        : {INT-LIST}        <default []>, FOR RE-WEIGHTING ONLY, trajectory numbers which are generated with scaledMD,
                                        e.g. [1,3,5] means trajNr 1,3,5 of TrajNameList are scaledMD trajectories | has to correspond to possible _ColY EventCurves;
    SameTraj       : {INT}             <default None> IF MULTIPLE EventNames are submitted with different sim times,
                                        SameTraj defines TrajNr, for which different parts are compared, 
                                        e.g. 1 for comparing traj1 of different simulation times;
OUTPUT:
    stores the overlap in a text-file
        GroupNr | Threshold | [densO | densStd | confO | TotFrames] | [...] of full submitted CompareList
    GroupNr monitors the number of the comparing group of [X]vs[Y]vs... meaning that
      GroupNr monitors the number of the comparing group of [X]vs[Y]vs... (K of O(K,L;r)) meaning that
            the overlap value corresponds to this specific reference trajectory (set)
    """
#### #### #### #### #### 
#### INITIALIZATION
#### #### #### #### ####    
    #-- set complete SaveName for SameTraj != True 
    if not SameTraj:
        ReweightList = ['_noWeight', '_MF+sMD', '_McL+sMD', '_Exp+sMD', '_sMD', '_MF', '_McL', '_Exp']
        if type(EventNames) == list:
            try:
                reweight    = [elem for elem in ReweightList if EventNames[0].find(elem) != -1][0]
                EndingFrame = int(EventNames[0].split('%s' % reweight)[0].split('-')[-1])
                StartFrame  = int(EventNames[0].split('%s' % reweight)[0].split('-')[-2].split('_')[-1])
            except IndexError:
                raise ValueError('EventNames={} must contain one "reweight" specification: \n{}'.format(EventNames, ReweightList))
            except ValueError:
                raise ValueError('EventNames={} must contain one "reweight" specification, BUT there is a conflict with the filename!\n'.format(EventNames)+\
                                 'Check, if one of the "reweight" algorithms are not specified twice in the EventNames:\n{}'.format(ReweightList))
            SaveName    = '%s_%s-%s%s.txt' % (SaveName, StartFrame, EndingFrame, reweight)
        else:
            try:
                reweight    = [elem for elem in ReweightList if EventNames.find(elem) != -1][0]
                EndingFrame = int(EventNames.split('%s' % reweight)[0].split('-')[-1])
                StartFrame  = int(EventNames.split('%s' % reweight)[0].split('-')[-2].split('_')[-1])
            except IndexError:
                raise ValueError('EventNames{} must contain one "reweight" specification: \n{}'.format(EventNames, ReweightList))
            except ValueError:
                raise ValueError('EventNames={} must contain one "reweight" specification, BUT there is a conflict with the filename!\n'.format(EventNames)+\
                                 'Check, if one of the "reweight" algorithms are not specified twice in the EventNames:\n{}'.format(ReweightList))
            SaveName    = '%s_%s-%s%s.txt' % (SaveName, StartFrame, EndingFrame, reweight)
    #####
    if os.path.exists('%s%s' % (SaveDir, SaveName)):
        print 'The overlap file already exist\n\tSaveDir = %s\n\tSaveName = %s' % (SaveDir, SaveName)
    else:
        t1=time.time()
    #### #### ####
    #---- generate Directories  
        Generate_Directories(SaveDir)

#### #### #### #### ####
# SAME TRAJECTORY, DIFFERENT SIMULATION TIMES/PARTS
#### #### #### #### ####
        if SameTraj != None:
        #--- ERROR DETECTION
            ColS = []
            refColS = []
            for Name in EventNames:
                temp_NormName = Extract_NormName(Name)
                if not os.path.exists('%s%s' % (EventDir, Name)) or \
                   not os.path.exists('%s%s' % (EventDir, temp_NormName)):
                    raise NameError('EventCurve does not exist\n\t%s\n\t%s' % (EventDir, Name))
                if Name.find('Col') != -1:
                    ColS.extend([int(elem) for elem in Name.split('_Col')[1].split('.')[0].split('_')])
                    refColS = [int(elem) for elem in Name.split('_Col')[1].split('.')[0].split('_')]
            if NP.unique(refColS) != NP.unique(ColS):
                raise('EventNames have to share the same Columns. Check your input\nEventNames = {}'.format(EventNames))
            if refColS == []:
                COL_TrajNrList = None
            else:
                COL_TrajNrList = NP.unique(refColS)
        #### #### ####
        #--- Generate HEADER, & extract all relevant values like TrajNameList, ThresholdList, ...
        #---- extract Header and necessary Parameters from the EventCurve
            tempoNormName = []
            for EE in EventNames:
                tempoNormName.append( Extract_NormName(EE) )
        #-------
            Header2, TrajNameList, StartFrame, EndingFrame, Iterations, aMD_reweight, Lambda, Order, ThresholdList, _=\
                        Extract_Header_and_Parameters(EventDir, tempoNormName, SameTraj=SameTraj)
            NrOfTrajs  = (len(NP.genfromtxt('%s%s' % (EventDir, tempoNormName[0])))-2)/len(ThresholdList)
            if COL_TrajNrList is None:
                COL_TrajNrList = range(1,NrOfTrajs+1)
        #--- generate new 1by1 CompareList: each part vs each part vs each part vs ...
            CompareList = []
            for kai in range(len(EventNames)):
                CompareList.append([kai+1])
            CompareList = [CompareList]
        #--- Generate ColumnList for SameTraj corresponding to the rows of the EventCurve-file <-> COL_TrajNrList
            ColumnList_ST = [0,1];
            for Threshold in ThresholdList:
                Index = 0
                for Tr in COL_TrajNrList:
                    if Tr == SameTraj:
                        ColumnList_ST.append(Index+2+ThresholdList.index(Threshold)*(NrOfTrajs))
                    Index += 1
        #--- extract Starting and Ending frames
            LEN = NP.sum(NP.subtract(EndingFrame, StartFrame))
        #--- Combine EventCurves & NormMatrices
            EventCurve = NP.zeros( (LEN, 2+len(EventNames)*len(ThresholdList)) )
            NormMatrix = NP.zeros( (1, 2+len(EventNames)*len(ThresholdList)) )
            for Kai in range(len(EventNames)):
            #---- define NormName v23.06.16
                NormName = Extract_NormName(EventNames[Kai])
            #----
                if EventNames[Kai][-3:] == 'npy':
                    temp = NP.load('%s%s' % (EventDir, EventNames[Kai]))[:,ColumnList_ST]
                else:
                    temp = NP.genfromtxt('%s%s' % (EventDir, EventNames[Kai]), usecols=ColumnList_ST)
                tempNorm = NP.genfromtxt('%s%s' % (EventDir, NormName), usecols=ColumnList_ST)
            #--- CHECK if SameTraj is present
                if [elem for elem in NP.unique(temp[:,1])].count(SameTraj) == 0:
                    raise ValueError('SameTraj = %s is not present in the EventCurve\n\tEventDir = %s\n\tEventNames = %s' % \
                                            (SameTraj, EventDir, EventNames))
                temp = temp[temp[:,1] == SameTraj]
                #if Kai == 0:
                #    EventCurve[:,0] = temp[:,0]
                StEvent = 0
                tempGroupNr = 0
                for St, En in zip(StartFrame, EndingFrame):
            #--- init Frames
                    EventCurve[StEvent:(StEvent+En-St),0] = temp[St:En, 0]
            #--- init GroupNr
                    EventCurve[StEvent:(StEvent+En-St),1] = tempGroupNr + 1
                    tempGroupNr += 1
                    for Koi in range(len(ThresholdList)):
            #--- copy TrajNr to different Columns corresponding to the EventNames
                        EventCurve[StEvent:(StEvent+En-St),2+Kai+Koi*len(EventNames)] = temp[St:En, 2+Koi]
                        if (St,En) == (StartFrame[0], EndingFrame[0]):
                            NormMatrix[0, 2+Kai+Koi*len(EventNames)] = tempNorm[2+Koi]
                    StEvent += En-St
        #--- modify aMD_Nrs/sMD_Nrs: because one uses only ONE trajectory, if it is accelerated, then all trajs are
        #    additionally: trajectories are counted consecutively 1,2,3,...,len(EventNames)
            if aMD_Nrs.count(SameTraj) > 0:
                aMD_Nrs = [elem+1 for elem in range(len(EventNames))]
            else:
                aMD_Nrs = []
            if sMD_Nrs.count(SameTraj) > 0:
                sMD_Nrs = [elem+1 for elem in range(len(EventNames))]
            else:
                sMD_Nrs = []
        #--- DEFINE NrOfTrajs for the modified EventCurve comparing only SAME TRAJECTORIES
            NrOfTrajs = len(EventNames)
#### #### #### #### ####
# DIFFERENT TRAJECTORIES; SAME SIMULATION TIMES/PARTS  -  diff trajs + diff times not implemented (yet?)
#### #### #### #### ####
        else:
            EventCurve, NormMatrix, COL_TrajNrList, Header2, TrajNameList, StartFrame, EndingFrame, Iterations, \
            aMD_reweight, Lambda, Order, ThresholdList, updateTrjLenList = Return_Merged_EventCurve(EventDir, EventNames)
            NrOfTrajs  = (len(NormMatrix[0,:])-2)/len(ThresholdList)
    #-----------------------
        TT2 = time.time()
    #### #### ####
    #---- define HEADER
        HEADER = 'this file contains the <density overlap | conformational overlap | total # frames> '+\
                 'for the different CompareList \n\n'+\
                 'ensure, that the columns and the submitted CompareList do match\n\n'+\
                 'INPUT:\n'+\
                     '\tEventDir                 = {}\n'.format(EventDir)+\
                     '\tEventNames               = {}\n'.format(EventNames)+\
                     '\tCompareList              = {}\n'.format(CompareList)+\
                     '\tCOL_TrajNrList           = {}\n'.format(COL_TrajNrList)+\
                     '\taMD_Nrs                  = {}\n'.format(aMD_Nrs)+\
                     '\tsMD_Nrs                  = {}\n'.format(sMD_Nrs)
        HEADER = HEADER + Header2 + '\n'
    #### #### ####
    #---- init OverlapMatrix: GroupNr | Threshold | (densO | confO | TotalNr) x len(CompareList)
    #     GroupNr: projection on the GroupNr-th trajectory/ies
    #     densO   : sum of min()/max() ratios, NO AVERAGING
    #     densStd : standard error of min()/max() values
    #     confO   : Nr of overlapping frames
    #     TotalNr: total number of involved frames
    #     to extract the density overlap & conformational overlap (to the corresp. GroupNr), divide densO & confO by TotalNr
        OverlapMatrix = NP.zeros( (len(ThresholdList)*len(CompareList[0]), 1+1+4*len(CompareList)) )-1
        OverlapMatrix[:,0] = range(1,len(CompareList[0])+1)*len(ThresholdList)
        OverlapMatrix[:,1] = NP.concatenate([[elem]*len(CompareList[0]) for elem in ThresholdList])
        FMT = '%i %s  '+('  %.4f %.4f %i %i')*len(CompareList)
#### #### #### #### #### 
#### ERROR DETECTION
#### #### #### #### ####
    #---- CompareList has to have same lengths to combine them
        if NP.any([len(elem)-len(CompareList[0]) for elem in CompareList]):
            raise ValueError('CompareList does not have the same lengths of TrajNrs'+\
                             '\n\tCompareList lengths = {}'.format([len(elem) for elem in CompareList]))
    #---- the trajs in CompareList have to match the TrajNr-windows of EventCurve
        Unique_Compare = []
        for First in CompareList:
            for Second in First:
                Unique_Compare.extend(Second)
        Unique_Compare = [int(elem) for elem in NP.unique(Unique_Compare)]
        Unique_Events = NP.unique(EventCurve[:,1])
        if len(NP.unique(NP.concatenate( (Unique_Compare, Unique_Events) ))) != len(Unique_Events):
            raise ValueError('submitted CompareList does not match the present trajectory numbers in <EventCurves>'+\
             ('check your input:\n\tCompareList = %s\n\tEventCurve = %s%s' % (CompareList, EventDir, EventNames)))
    #---- MODIFY aMD_Nrs and sMD_Nrs, that they match CompareList
        aMD_Nrs = [elem for elem in aMD_Nrs if Unique_Compare.count(elem) > 0]
        sMD_Nrs = [elem for elem in sMD_Nrs if Unique_Compare.count(elem) > 0]
    #---- if aMD/sMD are submitted, the corresponding Weights have to exist
        if aMD_Nrs != []:
            for aA in aMD_Nrs:
                if aMD_reweight == 'MF':
                    if not os.path.exists('%saMD_Weight_MF_%s_%s-%s_Iter%s.txt' % \
                                           (WeightDir, TrajNameList[aA-1], StartFrame[aA-1], 
                                            EndingFrame[aA-1], Iterations[aA-1])):
                        raise NameError(('The Weight-Vector for %s does not exist, please check your input' % \
                                            TrajNameList[aA-1])+\
                                        ('\n\taMD trajectories = {}'.format(aMD_Nrs))+\
                                        ('\n\t%saMD_Weight_MF_%s_%s-%s_Iter%s.txt' % \
                                           (WeightDir, TrajNameList[aA-1], StartFrame[aA-1], 
                                            EndingFrame[aA-1], Iterations[aA-1])))
                else:
                    if not os.path.exists('%saMD_Weight_%s.txt' % (WeightDir, TrajNameList[aA-1])):
                        raise NameError(('The Weight-Vector for %s does not exist, please check your input' % \
                                            TrajNameList[aA-1])+\
                                        ('\n\taMD trajectories = {}'.format(aMD_Nrs))+\
                                        ('\n\t%saMD_Weight_%s.txt' % (WeightDir, TrajNameList[aA-1])))
        if sMD_Nrs != []:
            for sS in sMD_Nrs:
                if not os.path.exists('%ssMD_Weight_lambda%s_%s_%s-%s_Iter%s.txt' % \
                               (WeightDir, Lambda, TrajNameList[sS-1], StartFrame[sS-1], 
                                EndingFrame[sS-1], Iterations[sS-1])):
                    raise NameError(('The Weight-Vector for %s does not exist, please check your input' % \
                                            TrajNameList[aA-1])+\
                                    ('\n\taMD trajectories = {}'.format(aMD_Nrs))+\
                                    ('\n\t%ssMD_Weight_lambda%s_%s_%s-%s_Iter%s.txt' % \
                                       (WeightDir, Lambda, TrajNameList[sS-1], StartFrame[sS-1], 
                                        EndingFrame[sS-1], Iterations[sS-1])))                
#### #### #### #### ####
#####    AnalogousOverlap
#### #### #### #### ####
    #####################################
    ## CompareList:
    #  e.g.: [((1,2),(3,4),(5,6), ((1),(2,3),(4,5,6))] for 1+2 vs 3+4 vs 5+6 AND 1 vs 2+3 vs 4+5+6
    #####################################
        RowIndex = 0
        for Threshold in ThresholdList:
            #---- Generate ColumnList corresponding to the rows of the EventCurve-file <-> COL_TrajNrList
            ColumnList = {}
            if COL_TrajNrList is None:
                Index = 0
                for Tr in range(1,NrOfTrajs+1):
                    ColumnList[Tr] = Index+2+ThresholdList.index(Threshold)*(NrOfTrajs)
                    Index += 1
            else:
                Index = 0
                for Tr in COL_TrajNrList:
                    ColumnList[Tr] = Index+2+ThresholdList.index(Threshold)*(NrOfTrajs)
                    Index += 1
            CompareIndex = 0
            for CompareGrp in CompareList: ## comparison GroupX vs GroupY vs ...
    #### #### ####
    #---- define the actual index of OverlapMatrix[RowIndex,:]
                RowIndex = len(CompareGrp)*ThresholdList.index(Threshold)
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
                        if aMD_Nrs.count(TrajNr) > 0:
                #---- Load Weights for aMD references
                            if aMD_reweight == 'MF':
                                Weights = NP.exp(NP.genfromtxt('%saMD_Weight_MF_%s_%s-%s_Iter%s.txt' % \
                                           (WeightDir, TrajNameList[TrajNr-1], StartFrame[TrajNr-1], 
                                            EndingFrame[TrajNr-1], Iterations[TrajNr-1])))
                            elif aMD_reweight == 'McL':
                                tempWeights = NP.genfromtxt('%saMD_Weight_%s.txt' % \
                                    (WeightDir, TrajNameList[TrajNr-1]), usecols=(0))[StartFrame[TrajNr-1]:EndingFrame[TrajNr-1]]
                                Weights = NP.zeros(len(tempWeights))
                                for Ord in range(0,Order+1):
                                    Weights = NP.add(Weights, 
                                                     NP.divide( NP.power( tempWeights , Ord), 
                                                                float(scipy.misc.factorial(Ord)) ) )
                            else:
                                Weights = NP.exp(NP.genfromtxt('%saMD_Weight_%s.txt' % \
                                    (WeightDir, TrajNameList[TrajNr-1]), usecols=(0))[StartFrame[TrajNr-1]:EndingFrame[TrajNr-1]])
                        elif sMD_Nrs.count(TrajNr) > 0:
                #---- Load Weights for sMD references
                            Weights = NP.genfromtxt('%ssMD_Weight_lambda%s_%s_%s-%s_Iter%s.txt' % \
                                   (WeightDir, Lambda, TrajNameList[TrajNr-1], StartFrame[TrajNr-1], 
                                    EndingFrame[TrajNr-1], Iterations[TrajNr-1]))
                        else:
                #---- Load Weights = 1 for cMD references
                            Weights = NP.ones( len(EventCurve[EventCurve[:,1]==TrajNr][:,0]) )
            #### #### ####
            #---- densO
                        with NP.errstate(divide='ignore', invalid='ignore'):
                            densO += NP.sum(\
                                      NP.multiply(\
                                        NP.divide(\
                                          NP.min([NP.divide(\
                                                        NP.sum(EventCurve[EventCurve[:,1]==TrajNr][:,elem], axis=1),
                                                        NP.sum(NormMatrix[0,elem], axis=0)) \
                                                  for elem in mod_ComGrp], axis=0),
                                          NP.max([NP.divide(\
                                                        NP.sum(EventCurve[EventCurve[:,1]==TrajNr][:,elem], axis=1),
                                                        NP.sum(NormMatrix[0,elem], axis=0)) \
                                                  for elem in mod_ComGrp], axis=0)),
                                        (Weights if Weights.shape == (len(EventCurve[EventCurve[:,1]==TrajNr][:,0]),)\
                                                 else Weights[:,ThresholdList.index(Threshold)])))
            #### #### #### v20.10.16
            #---- STD_densO
                        with NP.errstate(divide='ignore', invalid='ignore'):
                            STD_densO += NP.sum(NP.multiply(\
                                      NP.power(\
                                        NP.divide(\
                                          NP.min([NP.divide(\
                                                        NP.sum(EventCurve[EventCurve[:,1]==TrajNr][:,elem], axis=1),
                                                        NP.sum(NormMatrix[0,elem], axis=0)) \
                                                  for elem in mod_ComGrp], axis=0),
                                          NP.max([NP.divide(\
                                                        NP.sum(EventCurve[EventCurve[:,1]==TrajNr][:,elem], axis=1),
                                                        NP.sum(NormMatrix[0,elem], axis=0)) \
                                                  for elem in mod_ComGrp], axis=0)),2),
                                        (Weights if Weights.shape == (len(EventCurve[EventCurve[:,1]==TrajNr][:,0]),)\
                                                 else Weights[:,ThresholdList.index(Threshold)])))

            #### #### ####
            #---- Total Nr of Weights for the average of densO
                        TotNrWeights += NP.sum((Weights if Weights.shape == (len(EventCurve[EventCurve[:,1]==TrajNr][:,0]),)\
                                                        else Weights[:,ThresholdList.index(Threshold)]))
            #### #### ####
            #---- confO
                        confO += NP.sum(\
                                   NP.sign(\
                                      NP.min([NP.sum(EventCurve[EventCurve[:,1]==TrajNr][:,elem], axis=1)\
                                              for elem in mod_ComGrp], axis=0) ) )
            #### #### ####
            #---- Total Nr of Frames of the reference trajectory
                        TotFrames += len(EventCurve[EventCurve[:,1]==TrajNr][:,0])
                 #### calculate STD_densO for one reference trajectory   
                    STD_densO = (float(STD_densO)/float(TotNrWeights) - NP.power(float(densO)/float(TotNrWeights),2))
    #### #### ####
    #---- "normalize" densO by averaging over all reference frames, which is equal to the sum of Weights
                    ### STORE densO, confO, TotFrames to OverlapMatrix
                    OverlapMatrix[RowIndex, 2+CompareIndex+0] = float(densO)/float(TotNrWeights)
                    OverlapMatrix[RowIndex, 2+CompareIndex+1] = STD_densO
                    OverlapMatrix[RowIndex, 2+CompareIndex+2] = confO
                    OverlapMatrix[RowIndex, 2+CompareIndex+3] = TotFrames
                    RowIndex += 1
            #---- raise the Column Index for every new CompareGroup        
                CompareIndex += 4
###########################
##   STORE Overlap
###########################
        t2=time.time()
        ########
        ## HEADER specification for CompareList
        ########
        HEADER = HEADER + ('calculation time = %s seconds\n\n' % round(t2-t1,2))
        HEADER = HEADER + '\tdensO has to be averaged for different trajectory Nr projections\n'+\
                          '\tErr   is the standard error of the mean, which is taken for the weighted average calculation for different trajectory projections\n'+\
                          '\tconfO means the number of frames which have at least 1 event for each trajectory group\n'+\
                          '\tTotFrames means the total number of frames of the certain trajectory group\n'+\
                          '\tconfO and TotFrames have to be summed for different trajectory Nr projections\n'+\
                              '\t\tand then divided to obtain the real conformational overlap value\n'+\
                              '\tGroupNr monitors the number of the comparing group of [X]vs[Y]vs... (K of O(K,L:r)) meaning that\n'+\
                              '\t\tthe overlap value corresponds to the reference trajectory (set) of the K-th group\n'
        HEADER = HEADER + '\nGroupNr | Threshold | [densO|Err|confO|TotFrames] of '
        for Sep in CompareList:
            HEADER = HEADER + '%s | ' % ('vs.'.join(['%s' % elem for elem in Sep]))
        NP.savetxt('%s%s' % (SaveDir, SaveName), OverlapMatrix, fmt=FMT, header=HEADER)

###############
#-------------
###############

def Return_Merged_EventCurve(EventDir, EventNames):
    """
v22.07.16
    - supporting function for <Calc_Overlap()>, to return (a possible merged) EventCurve for the overlap calculation
      of different trajectories
    - merging reference trajectories <_RowX> and column trajectories <_ColY> to a whole EventCurve
    - commit also the Parameters from the <Extract_Header_and_Parameters()>, because they are already needed
INPUT:
    EventDir   : {STRING}      directory, where the EventCurves are located, e.g. 'EventCurves/'
    EventNames : {STRING/LIST} Name of the EventCurve, can also be a list to combine different _RowX, _ColY files,
                                    e.g. ['Events_0-500_MF_Row1_Col1.txt', 'Events_0-500_MF_Row1_Col2.txt',
                                          'Events_0-500_MF_Row2_Col1.txt', 'Events_0-500_MF_Row2_Col2.txt'];
OUTPUT:
    return EventCurve, NormMatrix, COL_TrajNrList, Header2, TrajNameList, StartFrame, EndingFrame, Iterations, \
           aMD_reweight, Lambda, Order, ThresholdList, updateTrjLenList
    """
 ################################
    def MergeCols(EventCurveList, NrOfThresholds, COL_TrajNrList):
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
            for Rad in range(NrOfThresholds):
                for EV in EventCurveList:
                    if 'EventCurve' not in locals():
                        EventCurve = EV[:,0:(2+(len(EV[0,:])-2)/NrOfThresholds)]
                    else:
                        EventCurve = NP.concatenate( (EventCurve, EV[:,(2+Rad*(len(EV[0,:])-2)/NrOfThresholds):(2+(Rad+1)*(len(EV[0,:])-2)/NrOfThresholds)]), axis=1)
            COL_T, Ind = NP.unique(COL_TrajNrList, return_index=True)
            Boja = [0,1]+[2+elem+Rad*len(COL_TrajNrList) for Rad in range(NrOfThresholds) for elem in Ind]
        #-----
            return EventCurve[:,Boja], COL_T
 ################################
    def MergeRows(EventDir, EventNames):
        """
    v25.07.16
    EventDir    :  directory
    EventNames  :  EventNames with same Cols merges different Rows
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
 ################################
    if type(EventNames) == list:
      ##########  
        ColDict = defaultdict(list); Index = 0;
        EV_Prefix = []
        for EV in EventNames:
            if not os.path.exists('%s%s' % (EventDir, EV)):
                raise NameError('The submitted <EventCurve> does not exist, check the input\n\t%s%s' % \
                                (EventDir, EV))
        #--- CHECK if EventNames have the same simulation times
            if EV.find('_Row') != -1:
                EV_Prefix.append(EV.split('_Row')[0])
            elif EV.find('_Col') != -1:
                EV_Prefix.append(EV.split('_Col')[0])
            else:
                EV_Prefix.append(EV)
        #--- Generate ColDict
            if EV.find('_Col') != -1:
                ColDict['%s' % EV.split('_Col')[1].split('.')[0]].append(Index)
            else:
                ColDict[''].append(Index)
            Index += 1
      ##########
        if len(NP.unique(EV_Prefix)) > 1:
            raise NameError('You try to merge EventCurves with different simulation times, '+\
                            'this might be an error. Check your Input!\n\tunique EV_Prefix = %s' % \
                                   NP.unique(EV_Prefix))
      ##########  
        if len(ColDict.keys()) == 1 and ColDict.keys()[0] == '':
            COL_TrajNrList = None
        elif len(ColDict.keys()) > 1 and ColDict.keys().count('') > 0:
            raise ValueError('You try to merge EventCurves containing all trajectories\n\t(no >_ColX<)\n and '+\
                             'EventCurves containing only some trajectories\n\t(>_ColX<).\n This might '+\
                             'be an error, check your input!\n\tEventNames = {}'.format(EventNames))
        else:
            COL_TrajNrList = [int(elem) for elem in '_'.join([elem2 for elem2 in ColDict.keys()]).split('_')]
      ##########
        #--- Extract Header + Parameters
        NormNames = Extract_NormName(EventNames[0])
        Header2, TrajNameList, StartFrame, EndingFrame, Iterations, aMD_reweight, Lambda, Order, ThresholdList, \
                updateTrjLenList = Extract_Header_and_Parameters(EventDir, [NormNames], SameTraj=None)
      ##########
        EventCurve_List = [MergeRows(EventDir, [EventNames[elem] for elem in ColM]) \
                           for ColM in ColDict.values()]
        EventCurve, _ = MergeCols(EventCurve_List, len(ThresholdList), COL_TrajNrList)
        try:
            NormMatrix_List = [NP.reshape(NP.genfromtxt('%s%s' % (EventDir, 
                                                                  Extract_NormName(EventNames[ColM[0]]))), 
                                          (1,-1)) for ColM in ColDict.values()]
            NormMatrix, COL_TrajNrList = MergeCols(NormMatrix_List, len(ThresholdList), COL_TrajNrList)
        except NameError:
            raise NameError('The submitted <NormMatrix> does not exist, check the input\n\t%s%s' % \
                            (EventDir, [Extract_NormName(EventNames[ColM[0]]) for ColM in ColDict.values()]))
        #return EventCurve_List, NormMatrix_List, COL_TrajNrList
#######################
    else:
        #---- define COL_TrajNrList
        if EventNames.find('_Col') == -1:
            COL_TrajNrList = None
        else:
            COL_TrajNrList = [int(elem) for elem in EventNames.split('_Col')[1].split('.')[0].split('_')]
        #---- define NormName v23.06.16
        NormNames = Extract_NormName(EventNames)
        #---- load EventCurves
        if os.path.exists('%s%s' % (EventDir, EventNames)) and os.path.exists('%s%s' % (EventDir, NormNames)):
            NormMatrix = NP.genfromtxt('%s%s' % (EventDir, NormNames))
            NormMatrix = NP.reshape(NormMatrix, (1,len(NormMatrix)))
            if EventNames[-3:] == 'npy':
                EventCurve = NP.load('%s%s' % (EventDir, EventNames))
            else:
                EventCurve = NP.genfromtxt('%s%s' % (EventDir, EventNames))
            # del NormMatrix: direct division does not work if trajs are later "concatenated"
        else:
            raise NameError('submitted <EventCurve/NormMatrix> do not exist, check the input\n\t%s%s\n\t%s%s' % \
                            (EventDir, EventNames, EventDir, NormNames))
        #---- extract Header and necessary Parameters from the EventCurve
        Header2, TrajNameList, StartFrame, EndingFrame, Iterations, aMD_reweight, Lambda, Order, ThresholdList, \
            updateTrjLenList = Extract_Header_and_Parameters(EventDir, [NormNames])
#---- MODIFY EventCurves (17.06.16): extract only StartFrame to EndingFrame of all trajectories
    if NP.sum(updateTrjLenList) != len(EventCurve[:,0]):
        temp = NP.copy(EventCurve)
        EventCurve = NP.zeros( (NP.sum(updateTrjLenList), len(temp[0,:])) )
        Unique_Events = NP.unique(temp[:,1])
        EvInd = 0; StEndInd = 0
        for TTs in Unique_Events:
            EventCurve[EvInd:(EvInd+len(temp[temp[:,1]==TTs][StartFrame[StEndInd]:EndingFrame[StEndInd],0])),:]=\
                    temp[temp[:,1]==TTs][StartFrame[StEndInd]:EndingFrame[StEndInd],:]
            EvInd += len(temp[temp[:,1]==TTs][StartFrame[StEndInd]:EndingFrame[StEndInd],0])
            StEndInd += 1
        del temp
    #----         
    return EventCurve, NormMatrix, COL_TrajNrList, Header2, TrajNameList, StartFrame, EndingFrame, Iterations, \
           aMD_reweight, Lambda, Order, ThresholdList, updateTrjLenList

###############
#-------------
###############

def Extract_Header_and_Parameters(EventDir, EventNamesList, SameTraj=None):
    """
v26.09.16
This function extracts the Header and Parameters from the EventCurve(s), which are necessary for the Overlap calculation
- since EventCurves are stored in .npy python-binary-format, the information is extracted from Norm_EventCurves.txt
- if len(EventNamesList) == 1: same parts with same parameters are used but different simulation times
- if len(EventNamesList) >  1: the overlap of different parts of THE SAME TRAJECTORY is calculated
INPUT:
    EventDir               : {STRING} Directory, where the Norm_EventCurves.txt are stored;
    EventNamesList         : {LIST}   List of names of Norm_EventCurves.txt, which are used for extraction
                                    if len() == 1, using same configurations/parts but different simulation times
                                    if len() >  1, overlap is calculated for ONE TRAJECTORY but different parts;
    SameTraj               : {INT}    <default None> if SameTraj is not None, different parts of the SameTraj-th trajectory 
                                        are compared returning different simulation time parts ;
OUTPUT:
    HEADER                 : {STRING}      Documentation of the corresponding EventCurves;
    TrajNameList           : {LIST}        List of used trajectory names, WITHOUT ENDING;
    StartFrame             : {INT-LIST}    Starting frame for each EventCurve, if SameTraj==None, all StartFrame are equal;
    EndingFrame            : {INT-LIST}    Ending frame for each EventCurve, if SameTraj==None, all EndingFrame are equal;
    Iterations             : {INT}         Number of Iterations for mean-field (MF) reweighting of aMD trajectories;
    aMD_reweight           : {STRING}      re-weighting algorithm, e.g. MF, McL or Exp;
    Lambda                 : {FLOAT}       re-scaling factor for the sMD reweighting scheme;
    Order                  : {INT}         Order of the MacLaurin expansion reweighting of aMD trajectories;
    ThresholdList          : {FLOAT-LIST}  Threshold parameter list used to calculate the EventCurves;
    updated_TrajLengthList : {INT-LIST}    Lenghts of each trajectory [in Frames] with respect to the simulation time parts;
    """
    #if SameTraj is not None:
    TrajNameList = []
    StartFrame = []
    EndingFrame = []
    Iterations = []
    TrajLengthList = []
    updated_TrajLengthList = []
    for EventNames in EventNamesList:
        HEADER = ''
        with open('%s%s' % (EventDir, EventNames), 'r') as INPUT:
            for line in INPUT:
                if len(line.split()) > 2 and line.split()[0] == '#' and line.split()[1] == 'each':
                    break
                if len(line.split()) > 2 and line.split()[0] == '#' and \
                   ['aMD_reweight', 'Lambda', 'Order'].count(line.split()[1]) > 0:
                    HEADER = HEADER + '\t\t'+line.split()[1].ljust(23)+'= '+line.split()[-1]+'\n'
            #### #### ####
            #---- load TrajNameList                    
                if len(line.split()) > 2 and line.split()[0] == '#' and line.split()[1] == 'TrajNameList':
                    if SameTraj is None:
                        TrajNameList = [elem.replace('\'','').replace(',','') \
                                        for elem in line[line.find('[')+1:line.find(']')].split()]
                    else:
                        TrajNameList.append([elem.replace('\'','').replace(',','') \
                                             for elem in line[line.find('[')+1:line.find(']')].split()][SameTraj-1])
            #### #### ####
            #---- load ThresholdList
                if len(line.split()) > 2 and line.split()[0] == '#' and line.split()[1] == 'ThresholdList':
                    ThresholdList = [float(elem.replace(',','')) for elem in line[line.find('[')+1:line.find(']')].split()]
            #### #### ####
            #---- load StartFrame
                if len(line.split()) > 2 and line.split()[0] == '#' and line.split()[1] == 'StartFrame':
                    if SameTraj is None:
                        if len(EventNamesList) != len(TrajNameList):
                            StartFrame = [int(line.split()[-1]) for elem in TrajNameList]
                        else:
                            StartFrame.append(int(line.split()[-1]))
                    else:
                        StartFrame.append(int(line.split()[-1]))
            #### #### ####
            #---- load EndingFrame
                if len(line.split()) > 2 and line.split()[0] == '#' and line.split()[1] == 'EndingFrame':
                    if SameTraj is None:
                        if len(EventNamesList) != len(TrajNameList):
                            EndingFrame = [int(line.split()[-1]) if line.split()[-1] != 'inf' else NP.infty for elem in TrajNameList]
                        else:
                            EndingFrame.append(int(line.split()[-1]))
                    else:
                        EndingFrame.append(int(line.split()[-1]))
            #### #### ####
            #---- load Iterations                        
                if len(line.split()) > 2 and line.split()[0] == '#' and line.split()[1] == 'Weight':    
                    Iterations = [int(line.split()[-1]) for elem in TrajNameList]
            #### #### ####
            #---- load aMD_reweight                        
                if len(line.split()) > 2 and line.split()[0] == '#' and line.split()[1] == 'aMD_reweight':    
                    aMD_reweight = (line.split()[-1])
            #### #### ####
            #---- load Lambda                        
                if len(line.split()) > 2 and line.split()[0] == '#' and line.split()[1] == 'Lambda':    
                    Lambda = float(line.split()[-1])
            #### #### ####
            #---- load Order                        
                if len(line.split()) > 2 and line.split()[0] == '#' and line.split()[1] == 'Order':    
                    Order = int(line.split()[-1])   
            #### #### ####
            #---- load updated_TrajLengthList                        
                if len(line.split()) > 2 and line.split()[0] == '#' and line.split()[1] == 'updated_TrajLengthList':    
                    if SameTraj is None:
                        updated_TrajLengthList = [int(elem.replace(',','')) \
                                                  for elem in line[line.find('[')+1:line.find(']')].split()]
                    else:
                        updated_TrajLengthList.append([int(elem.replace(',','')) \
                                                       for elem in line[line.find('[')+1:line.find(']')].split()][SameTraj-1])
    ## CHECK, if every parameter could be loaded from the Norm_EventCurves.txt
    for Params in ['TrajNameList', 'ThresholdList', 'StartFrame', 'EndingFrame', 'Iterations', 'aMD_reweight', 'Lambda', 
                   'Order', 'updated_TrajLengthList']:
        if locals().keys().count(Params) == 0:
            raise ValueError('The parameter %s is not specified in %s! The parameter could not be loaded. Check, that %s is specified in %s' % \
                                (Params if Params != 'Iterations' else 'Weight Iterations', EventNamesList, 
                                 Params if Params != 'Iterations' else 'Weight Iterations', EventNamesList))
    #### #### ####
    HEADER = '\t\tTrajNameList           = {}\n'.format(TrajNameList)+\
             '\t\tThresholdList             = {}\n'.format(ThresholdList)+\
             '\t\tStartFrame             = {}\n'.format(StartFrame)+\
             '\t\tEndingFrame            = {}\n'.format(EndingFrame)+\
             '\t\taMD/sMD MF Iterations  = {}\n'.format(Iterations)+\
             '\t\tupdated_TrajLengthList = {}\n'.format(updated_TrajLengthList)+HEADER
    return HEADER, TrajNameList, StartFrame, EndingFrame, Iterations, aMD_reweight, \
           Lambda, Order, ThresholdList, updated_TrajLengthList

###############
#-------------
###############

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

#######################################################
#------------------------------------------------------
#---          EVENTCURVE CALCULATION
#------------------------------------------------------
#######################################################
def Generate_EventCurves(TrajNameList, TrajLengthList, MatrixDir, SaveDir, SaveName, ThresholdList, MaxNumberLines, \
     ROW_TrajNrList=None, COL_TrajNrList=None, StartFrame=0, EndingFrame=NP.infty, PartList=None,
     aMD_Nrs=[], aMD_reweight='MF', aMDlogDir=None, aMDlogName=None, AmberVersion='Amber14', WeightStep=1, Temp=300,
     sMD_Nrs=[], Lambda=1, Order=10, BinFile_precision=NP.float32, Iterations=1, RMSD_SaveAdder=''):
    """ 
v28.11.16
    This function calculates the EventCurves using calculated RMSD matrices and possible aMD/sMD Weights: 
        - Events are the number of neighboring frames with Threshold < RMSD for each reference frame and for each traj
        - Events as a function of Simulation time
        - generates Events & corresponding Normalization factors = total number of (re-weighted) frames
        - the corresponding TrajNr is monitored for corresponding frames to easily detect, which reference frames refer to which trajectory
        - different Thresholds can be collected in one file
        - uses re-weighting for aMD/sMD
        - RMSD matrices with name format "TRAJNAME1_bin.dat", "TRAJNAME2_bin.dat", ... have to exist
    uses <Return_FullColRMSD()> to load the corresponding RMSD matrices
1. possibility, to select ROW_TrajNrList, to which the Events are counted, but must contain all numbers between the maximal and minimal trajectory number
2. possibility, to select COL_TrajNrList, which defines the trajectories the Events are counted for
3. Weight Generation can be done by submitting ONLY the aMD/sMD trajectory with the corresponding number of Iterations

INPUT:
    TrajNameList   : {LIST}      List of Names of the trajectories, NAMES NEED TO CORRESPOND TO RMSD_matrix
                                    TrajName1 -> TrajName1_bin.dat & TrajName1_TrajName2_bin.dat;
    TrajLengthList : {INT-LIST}  number of frames for corresponding TrajName, have to match RMSD matrix shape & 
                                    length of TrajNameList, e.g. [2000, 2000, 1000, ...] 
                                    if PartList != None, TrajNameList must store the NrOfFrames for each SPLIT;
    MatrixDir      : {STRING}    Name of the directory, where RMSD matrices are located, e.g. 'RMSD_matrices/';
    SaveDir        : {STRING}    Name of the save directory for the EventCurves, e.g. 'EventCurves/';
    SaveName       : {STRING}    Save PREFIX, e.g. 'V3_S1-S10' -> 'V3_S1-S10_StartFrame-EndingFrame_SamplingMethod.npy';
    ThresholdList  : {FLOAT-LIST} different thresholds r for which the Events are counted [nm], e.g. [0.1, 0.2, 0.3, ...];
    MaxNumberLines : {INT}          Maximal number of lines which are loaded from the FullRMSD matrix, 
                                    it inflicts directly the memory usage, if MaxNumberLines > sum(TrajLengthList), all
                                    RMSD matrices are loaded at once, recommended using length of one RMSD block,
                                    e.g. if one block RMSDmatrix.shape == (2000,2000), try to use MaxNumberLines = 2000;
    ROW_TrajNrList : {INT-LIST}  <default None>, reference trajectories, to which the Events are counted, e.g. [4,5,6], None means ALL trajectories are used;
    COL_TrajNrList : {INT-LIST}  <default None>, the Events are counted for these trajectories, e.g. [2,3,4], None means ALL trajectories are used;
    StartFrame     : {INT}       <default 0>    starting frame of Trajectories/RMSD matrices,
                                    to select different simulation times/lengths together with 'EndingFrame';
    EndingFrame    : {INT}       <default NP.infty> ending frame of Trajectories/RMSD matrices,
                                    to select different simulation times/lengths together with 'StartingFrame';
    PartList       : {INT-LIST}  <default None> defines into how many parts the single trajectories are split, due to
                                    memory reasons,
                                        1. len(PartList) == len(TrajNameList) !!  ||   
                                        2. PartList = [1,2,3], (MD1.xtc -> MD1.xtc),
                                                               (MD2.xtc -> MD2_part1.xtc, MD2_part2.xtc),
                                                               (MD3.xtc -> MD3_part1.xtc, MD3_part2.xtc, MD3_part3.xtc), ||
                                        3. default PartList = None -> PartList = [1]*len(TrajNameList);    
    aMD_Nrs        : {INT-LIST}  <default []> FOR RE-WEIGHTING ONLY, trajectory numbers which are generated with aMD
                                    e.g. [1,3,5] means trajNr 1,3,5 of TrajNameList are aMD trajectories;
    aMD_reweight   : {STRING}    <default MF> FOR RE-WEIGHTING ONLY, reweighting method if aMD trajs are present, default Mean-Field-Approach | 
                                    possibilities - 'MF', 'Exp', 'McL';
    aMDlogDir      : {LIST}      <default None> FOR RE-WEIGHTING ONLY,e.g. ['directory/'], where aMD.log from AMBER14/12 are located for each aMD trajectory,
                                    searching files with the following syntax '%s%s' % (aMDlogDir, aMDlogName) from AMBER14/12,
                                    len(aMDlogDir) == 1 OR len(aMDlogDir) == len(aMD_Nrs) for different traj directories;
    aMDlogName     : {LIST}      <default None> FOR RE-WEIGHTING ONLY, e.g. ['aMD.log', 'aMD.log'] for aMD_Nrs = [2,5] in ascending order,
                                    Names for aMD.log following the syntax '%s%s' % (aMDlogDir, aMDlogName) from AMBER14/12,
                                    necessary for re-weighting, if weights are NOT already present,
                                    len(aMDlogName) == len(aMD_Nrs), each name corresponds to the trajectory specified in aMD_Nrs;
    AmberVersion   : {STRING}    <default Amber14> FOR RE-WEIGHTING ONLY, Amber14 OR Amber12, because the units are different in both versions;
    WeightStep     : {INT}       <default 1> FOR RE-WEIGHTING ONLY, every WeightStep-th Row is used from the aMD.log, this HAS TO correspond
                                    to the same frames from the trajectories which are used for RMSD matrices;
    Temp           : {FLOAT}     <default 300> FOR RE-WEIGHTING ONLY, simulation temperature of the system (for correct units);
    sMD_Nrs        : {INT-LIST}  <default []> FOR RE-WEIGHTING ONLY, trajectory numbers which are generated with scaledMD
                                   e.g. [1,3,5] means trajNr 1,3,5 of TrajNameList are scaledMD trajectories;
    Lambda         : {FLOAT}     <default 1> FOR RE-WEIGHTING ONLY, scaling factor for scaledMD, e.g. 0.7, 1 means no scaling;
    Order          : {INT}       <default 10> FOR RE-WEIGHTING ONLY, Order for the MacLaurin expansion, ONLY NECESSARY IF aMD_reweight = 'McL';
 BinFile_precision : {TYPE}      FORMAT, [GROMACS/BINARY] single precision 'float32', double precision 'float64', [AMBER/ELSE] 'None';
    Iterations     : {INT}       <default 1> FOR RE-WEIGHTING ONLY, defines the number of MF iterations if aMD (aMD_reweight=MF) or sMD trajectories are present,
                                             for Iterations=-1, it iterates until convergence or 100000 steps;
OUTPUT:
    STORE noWeight & possible enhancedMatrix
    """
#### #### #### #### #### 
#### INITIALIZATION
#### #### #### #### #### 
    #---- INIT trajXList & trajYList for the "reference"- and "event calc"-trajs
    if ROW_TrajNrList is None: ## USE ALL TRAJECTORY CONFIGURATIONS
        trajXList = range(len(TrajNameList))
    else: 
        ROW_TrajNrList.sort()
        trajXList = [elem-1 for elem in ROW_TrajNrList]
        if ROW_TrajNrList != range(trajXList[0]+1, trajXList[-1]+2):
            raise ValueError('ROW_TrajNrList MUST contain all trajectory numbers between the maximal and minimal submitted integers! Check your input:\n\t'+\
                              ('ROW_TrajNrList = %s\nis not valid! Try to use\n\tROW_TrajNrList = %s' % (ROW_TrajNrList,range(trajXList[0]+1, trajXList[-1]+2))))
    #---- SORT ThresholdList
    ThresholdList.sort()
    #---- DELETE dubplicates
    temp = []
    [temp.append(elem) for elem in ThresholdList if not elem in temp]
    ThresholdList = [elem for elem in temp]
    del temp
    #---
    if COL_TrajNrList is None: ## USE ALL TRAJECTORIES TO COUNT EVENTS FOR
        trajYList = range(len(TrajNameList))
        NrOfCols = 1+1+len(TrajNameList)*len(ThresholdList)
    else:
        COL_TrajNrList.sort()
        trajYList = [elem-1 for elem in COL_TrajNrList]
        NrOfCols = 1+1+len(COL_TrajNrList)*len(ThresholdList)
    #---- INIT PartList
    if PartList is None:
        PartList = [1]*len(TrajNameList)
    #---- INIT TrajLenghtList to a dictionary
    TrajLenDict = Generate_TrajLenDict(len(TrajNameList), PartList, TrajLengthList)
    MaxTrajLength = max(max([max([NP.sum([TrajLenDict['%s%s' % (Kai, (('_part%s' % elem) if PartList[Kai] > 1 else ''))][0] \
                                      for elem in range(1,PartList[Kai]+1)])]) \
                         for Kai in trajYList]),
                        max([max([NP.sum([TrajLenDict['%s%s' % (Kai, (('_part%s' % elem) if PartList[Kai] > 1 else ''))][0] \
                                      for elem in range(1,PartList[Kai]+1)])]) \
                         for Kai in trajXList]))
    #  v25.05.16: use ALL trajX frames, BeginX = 0 | EndX = TrajLength
    NrOfRows = NP.sum([TrajLenDict['%s%s' % (elem, (('_part%s' % elem2) if PartList[Kai] > 1 else ''))][0] \
                       for elem in trajXList for elem2 in range(1,PartList[elem]+1)])
    #---- adjust EndingFrame to a int value if "infinity"
    EndingFrame = min(EndingFrame,MaxTrajLength)
  #---- aMD_Nrs & sMD_Nrs HAVE TO BE UNIQUE
    aMD_Nrs = NP.unique(aMD_Nrs); sMD_Nrs = NP.unique(sMD_Nrs); 
  #---- ADJUST AND CHECK aMDlogDir & aMDlogName
    # (1) if aMDlogDir/aMDlogName NOT None: len(aMDlogDir)+len(aMDlogName) == len(aMD_Nrs)
    # (2) merge both to ONE list, adjust that they match COL_TrajNrList
    # (3) use same "list-indices" like aMD_Nrs for Generate_Weights()
    if (aMDlogName is None or aMDlogDir is None) or len(aMDlogName) != len(aMD_Nrs) or len(aMDlogDir) > len(aMDlogName):
        aMDlogCombo = None;
    elif len(aMDlogDir) == 1:
        aMDlogCombo = ['%s%s' % (aMDlogDir[0], elem) for elem in aMDlogName]
    elif len(aMDlogDir) == len(aMDlogName):
        aMDlogCombo = ['%s%s' % (aMDlogDir[elem], aMDlogName[elem]) for elem in range(len(aMDlogName))]
    else:
        raise ValueError('aMDlogDir & aMDlogName have to be '+\
                         '\n\tEITHER both none'+\
                         '\n\tOR len(aMDlogDir) == 1, len(aMDlogName) == len(aMD_Nrs)'+\
                         '\n\tOR len(aMDlogDir) == len(aMDlogName) == len(aMD_Nrs)!\nCheck your input!')
  #---- MODIFY aMD_Nrs and sMD_Nrs, that they match COL_TrajNrList
    if aMDlogCombo is not None:
        aMDlogCombo = [aMDlogCombo[elem] for elem in range(len(aMDlogCombo)) if trajYList.count(aMD_Nrs[elem]-1) > 0 and \
               NP.sum([TrajLenDict['%s%s' % (aMD_Nrs[elem]-1, (('_part%s' % elem2) if PartList[Kai] > 1 else ''))][0] \
                       for elem2 in range(1,PartList[aMD_Nrs[elem]-1]+1)])>StartFrame]
    aMD_Nrs = [elem for elem in aMD_Nrs if trajYList.count(elem-1) > 0 and \
                       NP.sum([TrajLenDict['%s%s' % (elem-1, (('_part%s' % elem2) if PartList[Kai] > 1 else ''))][0] \
                               for elem2 in range(1,PartList[elem-1]+1)])>StartFrame]
    sMD_Nrs = [elem for elem in sMD_Nrs if trajYList.count(elem-1) > 0 and \
                       NP.sum([TrajLenDict['%s%s' % (elem-1, (('_part%s' % elem2) if PartList[Kai] > 1 else ''))][0] \
                               for elem2 in range(1,PartList[elem-1]+1)])>StartFrame]
  #---- Define SaveName for <EventMatrix> and <enhancedMatrix>
    SaveAdd1 = ''
    SaveAdd2 = ''
    if ROW_TrajNrList != None:
        SaveAdd1 = SaveAdd1 + '_Row'+'_'.join([str(elem) for elem in ROW_TrajNrList])
    if COL_TrajNrList != None:
        SaveAdd1 = SaveAdd1 + '_Col'+'_'.join([str(elem) for elem in COL_TrajNrList])
        SaveAdd2 = SaveAdd2 + '_Col'+'_'.join([str(elem) for elem in COL_TrajNrList])
    SaveEventMatrix    = '%s_%s-%s_noWeight%s.npy' % (SaveName, StartFrame, EndingFrame, SaveAdd1)
    SaveNormMatrix     = 'Norm_%s_%s-%s_noWeight%s.txt' % (SaveName, StartFrame, EndingFrame, SaveAdd2)
    if aMD_Nrs == [] and sMD_Nrs != []:
        SaveAdd1 = '_sMD' + SaveAdd1
        SaveAdd2 = '_sMD' + SaveAdd2
    elif sMD_Nrs == [] and aMD_Nrs != []:
        SaveAdd1 = '_'+aMD_reweight + SaveAdd1
        SaveAdd2 = '_'+aMD_reweight + SaveAdd2
    else:
        SaveAdd1 = '_'+aMD_reweight+'+sMD' + SaveAdd1
        SaveAdd2 = '_'+aMD_reweight+'+sMD' + SaveAdd2
    if aMD_Nrs == [] and sMD_Nrs == []:
        SaveEnhancedMatrix = ''
        SaveEnhancedNorm   = ''
    else:
        SaveEnhancedMatrix = '%s_%s-%s%s.npy' % (SaveName, StartFrame, EndingFrame, SaveAdd1)
        SaveEnhancedNorm   = 'Norm_%s_%s-%s%s.txt' % (SaveName, StartFrame, EndingFrame, SaveAdd2)
    if os.path.exists('%s%s' % (SaveDir, SaveEventMatrix)) and \
       ((aMD_Nrs == [] and sMD_Nrs == []) or os.path.exists('%s%s' % (SaveDir, SaveEnhancedMatrix))):
        print '\t>%s%s<\nand\n\t>%s%s<\nalready exists! Calculation stopped.' % \
                (SaveDir, SaveEventMatrix, SaveDir, SaveEnhancedMatrix)
    else:
        t1=time.time()
    #---- generate Directories  
        Generate_Directories(SaveDir)
        if (aMD_Nrs != [] or sMD_Nrs != []):
            Generate_Directories('%sWeights/' % SaveDir)
    #---- INIT NP.ndarray for EVENTS & NORM
        #                 |                  r=r1             |             r=r2                  |
        # Frames | TrajNr | Events traj1 | Events traj2 | ... | Events traj1 | Events traj2 | ... |
        EventMatrix = NP.zeros( (NrOfRows, NrOfCols), dtype=int )  
        NormMatrix  = NP.zeros( (   1    , NrOfCols), dtype=int )
        NormMatrix[0,0:2] = 1 # first two values are 1 to use is whole for NP.divide(EventMatrix,NormMatrix)
        if aMD_Nrs != [] or sMD_Nrs != []:
            enhancedMatrix = NP.zeros( (NrOfRows, NrOfCols) )
            enhancedNorm   = NP.zeros( (    1   , NrOfCols) )
#### #### #### #### #### 
#### ERROR DETECTION
#### #### #### #### ####         
    #---- CHECK if correct aMD_reweight is submitted
        if ['MF', 'McL', 'Exp'].count(aMD_reweight) == 0:
            raise ValueError('wrong aMD_reweight = %s is set! Use "MF" or "McL" or "Exp"' % aMD_reweight)
    #---- CHECK and define PartList for trajectory Splits
        if len(TrajNameList) != len(PartList):
            raise ValueError('length of >TrajNameList (%s)< and >PartList (%s)< must be the same' % \
                                (len(TrajNameList), len(PartList)) )
    #---- CHECK correct TrajLength's
        if NP.sum(PartList) != len(TrajLengthList):
            raise ValueError('length of >TrajNameList [with SPLITS] (%s)< and >TrajLengthList (%s)< must be the same' % \
                                (NP.sum(PartList), len(TrajLengthList)) )
    #---- CHECK TrajLength's with StartFrame
        if EndingFrame <= StartFrame:
            raise ValueError('StartFrame (%s) is larger than EndingFrame (%s)' % (StartFrame, EndingFrame))
        if MaxTrajLength <= StartFrame:
            raise ValueError('StartFrame (%s) is larger then the trajectory lengths (%s)' % (StartFrame, MaxTrajLength))
    #---- CHECK, if RMSD matrices exist
        for Kai in range(len(TrajNameList)):
            for PartX in [(('_part%s' % Part) if PartList[Kai] != 1 else '') for Part in range(1,PartList[Kai]+1)]:
                if not os.path.exists('%s%s%s%s_bin.dat' % (MatrixDir, TrajNameList[Kai], PartX, RMSD_SaveAdder)):
                    raise NameError('RMSD matrix \n\t>%s%s%s%s_bin.dat<\nnot found' % \
                                    (MatrixDir,TrajNameList[Kai],PartX, RMSD_SaveAdder))
                for Kai2 in range(Kai+1,len(TrajNameList),1):
                    if Kai2 > Kai:
                        for PartY in [(('_part%s' % Part) if PartList[Kai2] != 1 else '') \
                                      for Part in range(1,PartList[Kai2]+1)]:
                            if not os.path.exists('%s%s%s_%s%s%s_bin.dat' % \
                                      (MatrixDir, TrajNameList[Kai], PartX, TrajNameList[Kai2], PartY, RMSD_SaveAdder)) \
                               and not os.path.exists('%s%s%s_%s%s%s_bin.dat' % \
                                      (MatrixDir, TrajNameList[Kai2], PartY, TrajNameList[Kai], PartX, RMSD_SaveAdder)):
                                raise NameError('RMSD matrix \n\t>%s%s%s_%s%s%s_bin.dat<\nand\n\t>%s%s%s_%s%s%s_bin.dat<\n not found' % \
                                     (MatrixDir, TrajNameList[Kai], PartX, TrajNameList[Kai2], PartY, RMSD_SaveAdder, 
                                      MatrixDir, TrajNameList[Kai2], PartY, TrajNameList[Kai], PartX, RMSD_SaveAdder))
    #---- CHECK, if aMD.log are present for aMD_Nrs != []
    #----        and have the correct lengths as TrajLengthLis
        if aMD_Nrs != []:
            for Kai in aMD_Nrs:
                if trajYList.count(Kai-1) > 0:
                    for PartY in [(('_part%s' % Part) if PartList[Kai-1] != 1 else '') \
                                  for Part in range(1,PartList[Kai-1]+1)]:
                        if  (aMD_reweight == 'MF' and os.path.exists('%sWeights/aMD_Weight_MF_%s_%s-%s%s_Iter%s.txt' % \
                                  (SaveDir, TrajNameList[Kai-1], StartFrame, EndingFrame, PartY, Iterations))) \
                            or \
                            os.path.exists('%sWeights/aMD_Weight_%s%s.txt' % (SaveDir, TrajNameList[Kai-1], PartY)):
                            pass
                        elif aMD_reweight == 'MF' and trajXList.count(Kai-1) == 0:
                            raise ValueError('aMD Weights for <MF> cannot be calculated: not all aMD trajectories '+\
                                             'are present in ROW_TrajNrList\n'+\
                                             ('\t%s not in %s\nTrajName = %s, StartFrame, EndingFrame, Iterations = (%s, %s, %s)' % \
                                                  (Kai, [elem+1 for elem in trajXList], TrajNameList[Kai-1], 
                                                   StartFrame, EndingFrame, Iterations)))
                        elif aMDlogCombo is not None and os.path.exists(aMDlogCombo[aMD_Nrs.index(Kai)]):
                            temp = NP.genfromtxt(aMDlogCombo[aMD_Nrs.index(Kai)])[0::WeightStep]
                            if PartList[Kai-1] > 1 and len(temp[:,0])==NP.sum([TrajLenDict['%s_part%s' % (Kai-1, elem)][0] \
                                              for elem in range(1,1+PartList[Kai-1])]):
                                pass
                            elif PartList[Kai-1] == 1 and len(temp[:,0]) == TrajLenDict['%s' % (Kai-1)][0]:
                                pass
                            else:
                                if PartList[Kai-1] > 1: 
                                    LENGO = NP.sum([TrajLenDict['%s_part%s' % (Kai-1, elem)][0] \
                                                    for elem in range(1,1+PartList[Kai-1])])
                                else:
                                    LENGO = TrajLenDict['%s' % (Kai-1)][0]
                                raise ValueError('aMD Weight Check failed: length of the trajectory does not match the '+\
                                                 ('length of the aMD.log file\n\t%s\n\t%s != %s\n' % \
                                                  (aMDlogCombo[aMD_Nrs.index(Kai)], len(temp[:,0]), LENGO))+\
                                                 'check the trajectory and aMD.log lengths and select the correct frames')
                        else:
                            raise NameError('aMD Weight Check failed: no appropriate aMD-Weight-File is submitted!\n'+\
                                    ('Neither >%sWeights/aMD_Weight_MF_%s_%s-%s%s_Iter%s.txt<\n' % \
                                  (SaveDir, TrajNameList[trajY], StartFrame, EndingFrame, PartY, Iterations))+\
                                    ('nor >%sWeights/aMD_Weight_%s%s.txt\n' % (SaveDir, TrajNameList[trajY], PartY))+\
                                    ('nor >%s<\nexist! Calculation stopped' % \
                         (aMDlogCombo[aMD_Nrs.index(Kai)] if aMDlogCombo is not None else 'aMDlogDir/aMDlogName')))
    #---- CHECK if all necessary trajectories are present for sMD weight calculation
    #----       OR if the weights already exist
        if sMD_Nrs != []:
            for Kai in sMD_Nrs:
                if trajYList.count(Kai-1) > 0:
                    for PartY in [(('_part%s' % Part) if PartList[Kai-1] != 1 else '') \
                                  for Part in range(1,PartList[Kai-1]+1)]:
                        if trajXList.count(Kai-1) == 0 and \
                            not os.path.exists('%sWeights/sMD_Weight_lambda%s_%s_%s-%s%s_Iter%s.txt' % \
                                      (SaveDir, Lambda, TrajNameList[Kai-1], StartFrame, EndingFrame, PartY, Iterations)):
                            raise ValueError('<sMD> Weights cannot be calculated because not all sMD trajectories '+\
                                             'are present in ROW_TrajNrList\n'+\
                                             ('\t%s not in %s\nsMD_Weight_lambda%s_%s_%s-%s%s_Iter%s.txt' % \
                                                  (Kai, [elem+1 for elem in trajXList], Lambda, 
                                                   TrajNameList[Kai-1], StartFrame, EndingFrame, PartY, Iterations))) 
#### #### #### #### #### #### 
# if trajectories are split into parts, the EventCurve HAS TO BE built completely before generating the aMD/sMD MF weights
#### #### #### #### #### 
#### LOAD RMSD matrix
#### #### #### #### ####
    #---- INIT LOOP for (trajX, trajY) to have only one easy readable loop
    #---- use trajX and trajY loops to define the ROW (trajX) as reference configurations  
    #                                     and the COL (trajY), for which trajs the events are calculated
    #     GENERATE FIRST all diagonals, because this is necessary for weight generation
        Looper = []
        Looper2= []
        for trajX in trajXList:      ## "Reference"-trajectory loop, to which Events are counted
            for trajY in trajYList:  ## "Reference"-trajectory loop, to which Events are counted
                if trajX == trajY:
                    Looper.append((trajX, trajY))
                else:
                    Looper2.append((trajX, trajY))
        Looper = Looper+Looper2
        del Looper2
        FullCumTrajLenList = [max(0,min(NP.infty,Return_TrajLen_for_trajX(elem,
                                                                                 TrajLenDict,
                                                                                 PartList, 
                                                                                 range(len(TrajNameList))))-0) \
                              for elem in range(len(TrajNameList))]
        FullCumTrajLenList = [sum(FullCumTrajLenList[:elem2]) for elem2 in range(0,len(TrajNameList)+1)]
      ############
        for (trajX, trajY) in Looper:
            FullRMSDBlock = False
            CurrentRow=FullCumTrajLenList[trajX]
            EmergencyCancel = 0
## v24.08.16: IDEA OF PARTITIONING AND MF WEIGHTS
# 1. one NEEDS full EventMatrix & full INDICES (for all ref-frames)
# 2. i.e. <while not FullRMSDBlock:> should be done ONCE COMPLETELY to obtain full EventMatrix
# 3.      a) either STORE INDICES_LowerEnd-UpperEnd to load it then in the Weight MF generation
# 3.      b) or     load FullColRMSD, extract again INDICES, SORT THEM, and process to Weight MF generation
#####
# WORKFLOW:
# 1. <while not FullRMSDBlock:> for EventMatrix only
# 2. then do Weight MF generation, submit FULL EventMatrix[trajY]
# 3. INSIDE Weight Generation: load corresponding INDICES_LowerEnd-UpperEnd and do the MF steps
            LowUpArray = []
            while not FullRMSDBlock:
                #if EmergencyCancel == 1000:
                #    print 'EmergencyCancel'
                #    return
                #EmergencyCancel += 1
              #-- v26.08.16: return FULL RMSD matrix for given trajectory: StartFrame=0; EndingFrame=NP.infty
              #              then store INDICES and then use only RMSD_mat[:,BeginY:EndY]
              ################
                RMSD_mat, BeginX, EndX, BeginY, EndY, LowerEnd, UpperEnd = \
                    Return_FullColRMSD(MatrixDir, TrajNameList, CurrentRow, 
                                       MaxNumberLines, TrajLenDict, FullCumTrajLenList, PartList=PartList, 
                                       BinFile_precision=BinFile_precision, GLOBAL=True, 
                                       EventCurve=True, trajYList=[trajY], RMSD_SaveAdder=RMSD_SaveAdder)
              ################
                BeginY = StartFrame;  
                if PartList[trajY] > 1:
                    EndY = min(EndingFrame, NP.sum([TrajLenDict['%s_part%s' % (trajY, PayY)][0] 
                                                    for PayY in range(1,1+PartList[trajY]) 
                                                    if TrajLenDict['%s_part%s' % (trajY, PayY)][0] > 0]))
                else:
                    EndY = min(EndingFrame, TrajLenDict['%s' % (trajY)][0])
              ################
                if MaxNumberLines >= FullCumTrajLenList[trajX+1]-FullCumTrajLenList[trajX]:
                    FullRMSDBlock = True
                elif UpperEnd == FullCumTrajLenList[trajX+1]:
                    FullRMSDBlock = True
                elif UpperEnd == FullCumTrajLenList[-1]:
                    FullRMSDBlock = True
                CurrentRow = UpperEnd
              ################
    #### #### #### #### #### 
    #### CALCULATE Nr of Events for every Threshold: >> noWeight <<
    #### ASSIGN Events to >EventMatrix[:,:]< & >NormMatrix<
    #                     |                  r=r1             |             r=r2                  |
    #     Frames | TrajNr | Events traj1 | Events traj2 | ... | Events traj1 | Events traj2 | ... |
    #### #### #### #### ####
                if BeginX < EndX and BeginY < EndY:
            #---- SORT and store INDICEs of RMSD_Mat; STORE ONLY if not FullRMSD matrix could be loaded
                    if (aMD_Nrs.count(trajY+1) > 0 or sMD_Nrs.count(trajY+1) > 0):
                        if MaxNumberLines < FullCumTrajLenList[trajX+1]-FullCumTrajLenList[trajX] and \
                           not os.path.exists('%sINDICES_%s-%s_%s-%s.npy' % \
                                      (MatrixDir, TrajNameList[trajX], TrajNameList[trajY], LowerEnd, UpperEnd)) and \
                            not os.path.exists('%sINDICES_%s-%s_%s-%s.npy' % \
                                      (MatrixDir, TrajNameList[trajY], TrajNameList[trajX], LowerEnd, UpperEnd)):
                            NP.save('%sINDICES_%s-%s_%s-%s.npy' % (MatrixDir, TrajNameList[trajX], TrajNameList[trajY], 
                                                                   LowerEnd-FullCumTrajLenList[trajX], 
                                                                   UpperEnd-FullCumTrajLenList[trajX]), 
                                    NP.argsort(RMSD_mat, axis=1))
                    #INDICES = NP.argsort(RMSD_mat, axis=1)
                    RMSD_mat = RMSD_mat[:,BeginY:EndY]
                    INDICES = NP.argsort(RMSD_mat, axis=1)
                            
                    if aMD_Nrs.count(trajY+1) > 0 or sMD_Nrs.count(trajY+1) > 0:
                        LowUpArray.append((LowerEnd-FullCumTrajLenList[trajX], UpperEnd-FullCumTrajLenList[trajX]))
      #-- v31.10.16: if ROW_TrajLenList does not start at 1, there is a wrong shift
      #--            thus use (LowerEnd-FullCumTrajLenList[trajXList[0]]):(UpperEnd-FullCumTrajLenList[trajXList[0]]) for all index usages
            #---- INIT EventMatrix[:,[0,1],0] / NormMatrix[0, [0,1],0] with Frames & TrajNr
            #     THE THIRD DIMENSION NEEDS TO BE ZERO for correct NP.sum(, axis=2) of the single parts
                    EventMatrix[(LowerEnd-FullCumTrajLenList[trajXList[0]]):(UpperEnd-FullCumTrajLenList[trajXList[0]]),0] = \
                                                                    range(LowerEnd-FullCumTrajLenList[trajX], UpperEnd-FullCumTrajLenList[trajX],1)
                    EventMatrix[(LowerEnd-FullCumTrajLenList[trajXList[0]]):(UpperEnd-FullCumTrajLenList[trajXList[0]]),1] = trajX+1
                    NormMatrix[0, [2+trajYList.index(trajY)+elem*len(trajYList) for elem in range(len(ThresholdList))]] \
                               =  max(0,min(EndingFrame,FullCumTrajLenList[trajY+1]-FullCumTrajLenList[trajY])-StartFrame)
            #---- assign Counts/events to the corresponding trajectory column for the corresponding trajectory reference
                    II = 0

                    for Rows in range((LowerEnd-FullCumTrajLenList[trajXList[0]]),(UpperEnd-FullCumTrajLenList[trajXList[0]])):
                        EventMatrix[Rows,
                                    [2+trajYList.index(trajY)+elem*len(trajYList) for elem in range(len(ThresholdList))]] = \
                                            NP.searchsorted(RMSD_mat[II, INDICES[II,:]], ThresholdList)
                        II += 1
                    #---- INIT enhancedMatrix for aMD/sMD
                    if aMD_Nrs != [] or sMD_Nrs != []:
                        enhancedMatrix[(LowerEnd-FullCumTrajLenList[trajXList[0]]):(UpperEnd-FullCumTrajLenList[trajXList[0]]), 
                                       [0,1]+[2+trajYList.index(trajY)+elem*len(trajYList) \
                                              for elem in range(len(ThresholdList))]] = \
                           EventMatrix[(LowerEnd-FullCumTrajLenList[trajXList[0]]):(UpperEnd-FullCumTrajLenList[trajXList[0]]), 
                                       [0,1]+[2+trajYList.index(trajY)+elem*len(trajYList) \
                                              for elem in range(len(ThresholdList))]]
                        enhancedNorm[0, [0,1]+[2+trajYList.index(trajY)+elem*len(trajYList) \
                                               for elem in range(len(ThresholdList))]] = \
                          NormMatrix[0, [0,1]+[2+trajYList.index(trajY)+elem*len(trajYList) \
                                               for elem in range(len(ThresholdList))]]
                else:
                    FullRMSDBlock = True                        
#### #### #### #### #### 
#### LOAD/CALCULATE possible Weights for aMD/sMD for current trajectory (trajY )
#### ASSIGN Events to >enhancedMatrix[:,:]< & >enhancedNorm<
#### #### #### #### ####
    #### #### ####         
    #---- aMD
    #### #### #### 
            if aMD_Nrs.count(trajY+1) > 0:
        ##### ##### #####
        # LOAD/GENERATE aMD Weights
        ##### ##### #####
                WeightName = 'aMD_Weight_MF_%s_%s-%s_Iter%s.txt' % \
                              (TrajNameList[trajY], StartFrame, EndingFrame, Iterations)
                if aMD_reweight == 'MF':
                    ## for Weight generation, trajX==trajY, thus the trajY specific ROWS of EventMatrix are used
                    if PartList[trajY] > 1:
                        SpecEventRow = sum(TrajLengthList[:TrajLenDict['%s_part1' % (trajY)][1]])
                    else:
                        SpecEventRow = sum(TrajLengthList[:TrajLenDict['%s' % (trajY)][1]])
                else:
                    SpecEventRow = 0
                Weights = Generate_Weights(TrajNameList[trajY], 
                              EventMatrix[(SpecEventRow+BeginY):(SpecEventRow+EndY),
                                          [2+trajYList.index(trajY)+elem*len(trajYList) \
                                           for elem in range(len(ThresholdList))]],
                              INDICES[(BeginY):(EndY), :],
                              LowUpArray,
                              '%sWeights/' % SaveDir, WeightName, MatrixDir, BeginY, EndY, ThresholdList, aMD_reweight, 
                              Iterations, Lambda,
                              aMDlogCombo[aMD_Nrs.index(trajY+1)] if aMDlogCombo is not None else None, 
                              WeightStep, AmberVersion, Temp) 
        ##### ##### #####
        # ASSIGN enhancedMatrix (aMD)
        ##### ##### #####
                for ColIndex in range(len(ThresholdList)):
                    if aMD_reweight == 'MF':
                        enhancedNorm[0, 2+trajYList.index(trajY)+ColIndex*len(trajYList)] = \
                                                                NP.sum(NP.exp( Weights[:,ColIndex] ))
                    elif aMD_reweight == 'Exp':
                        enhancedNorm[0, 2+trajYList.index(trajY)+ColIndex*len(trajYList)] = NP.sum(NP.exp( Weights ))
                    elif aMD_reweight == 'McL':
                        Weight_McL = NP.zeros(len(Weights))
                        for Ord in range(0,Order+1):
                            Weight_McL = NP.add(Weight_McL, 
                                                NP.divide( NP.power( Weights , Ord), float(scipy.misc.factorial(Ord))))
                        enhancedNorm[0, 2+trajYList.index(trajY)+ColIndex*len(trajYList)] = NP.sum(Weight_McL)
                    #---------
                    if MaxNumberLines < FullCumTrajLenList[trajX+1]-FullCumTrajLenList[trajX]: II = len(INDICES[:,0]); 
                    else: II = 0;
                    III = 0;
                    for Row in range(LowUpArray[0][0]+FullCumTrajLenList[trajX]-FullCumTrajLenList[trajXList[0]], 
                                     LowUpArray[-1][1]+FullCumTrajLenList[trajX]-FullCumTrajLenList[trajXList[0]]):
                      #--- LOAD INDICES if necessary
                        if MaxNumberLines < FullCumTrajLenList[trajX+1]-FullCumTrajLenList[trajX] and \
                           len(INDICES[:,0]) <= II:
                            if os.path.exists('%sINDICES_%s-%s_%s-%s.npy' % \
                                  (MatrixDir,TrajNameList[trajX],TrajNameList[trajY],LowUpArray[III][0],LowUpArray[III][1])):
                                INDICES = NP.load('%sINDICES_%s-%s_%s-%s.npy' % \
                                  (MatrixDir,TrajNameList[trajX],TrajNameList[trajY],LowUpArray[III][0],LowUpArray[III][1]))
                            else:
                                INDICES = NP.transpose(NP.load('%sINDICES_%s-%s_%s-%s.npy' % \
                                  (MatrixDir,TrajNameList[trajX],TrajNameList[trajY],LowUpArray[III][0],LowUpArray[III][1])))
                            III +=1; II = 0; 
                      #--- v26.08.16 adjust indices to BeginY:EndY
                        if MaxNumberLines < FullCumTrajLenList[trajX+1]-FullCumTrajLenList[trajX] and \
                            BeginY > 0 or max(INDICES[0,:]) >= EndY: 
                            temp = INDICES[II, INDICES[II,:]>=BeginY]; temp = NP.subtract(temp[temp<EndY],BeginY);
                        else:
                            temp = INDICES[II,:]
                      #-----------
                        if aMD_reweight == 'MF':
                            enhancedMatrix[Row, 2+trajYList.index(trajY)+ColIndex*len(trajYList)] = \
                                                NP.sum(NP.exp( Weights[\
                                temp[0:EventMatrix[Row, 2+trajYList.index(trajY)+ColIndex*len(trajYList)] ], ColIndex] ))
                        elif aMD_reweight == 'Exp':
                            enhancedMatrix[Row, 2+trajYList.index(trajY)+ColIndex*len(trajYList)] = \
                                                NP.sum(NP.exp( Weights[\
                                temp[0:EventMatrix[Row, 2+trajYList.index(trajY)+ColIndex*len(trajYList)] ] ] ) )
                        elif aMD_reweight == 'McL':
                            enhancedMatrix[Row, 2+trajYList.index(trajY)+ColIndex*len(trajYList)] = \
                                                NP.sum(Weight_McL[\
                                temp[0:EventMatrix[Row, 2+trajYList.index(trajY)+ColIndex*len(trajYList)] ] ])
                        II += 1;
                    #---------
        #### #### ####         
        #---- sMD
        #### #### ####
            elif sMD_Nrs.count(trajY+1) > 0:
            ##### ##### #####
            # LOAD/GENERATE sMD Weights
            ##### ##### #####
                WeightName = 'sMD_Weight_lambda%s_%s_%s-%s_Iter%s.txt' % \
                              (Lambda, TrajNameList[trajY], StartFrame, EndingFrame, Iterations)
                ## for Weight generation, trajX==trajY, thus the trajY specific ROWS of EventMatrix are used
                if PartList[trajY] > 1:
                    SpecEventRow = sum(TrajLengthList[:TrajLenDict['%s_part1' % (trajY)][1]])
                else:
                    SpecEventRow = sum(TrajLengthList[:TrajLenDict['%s' % (trajY)][1]])
                Weights = Generate_Weights(TrajNameList[trajY], 
                              EventMatrix[(SpecEventRow+BeginY):(SpecEventRow+EndY),
                                          [2+trajYList.index(trajY)+elem*len(trajYList) \
                                           for elem in range(len(ThresholdList))]],
                              INDICES[(BeginY):(EndY), :],
                              LowUpArray,
                              '%sWeights/' % SaveDir, WeightName, MatrixDir, BeginY, EndY, ThresholdList, 'sMD', 
                              Iterations, Lambda, None, None, None, Temp)
                    ##### ##### #####
                    # ASSIGN enhancedMatrix (sMD)
                    ##### ##### #####        
                for ColIndex in range(len(ThresholdList)):
                    enhancedNorm[0, 2+trajYList.index(trajY)+ColIndex*len(trajYList)] = NP.sum(Weights[:,ColIndex])
                    
                    if MaxNumberLines < FullCumTrajLenList[trajX+1]-FullCumTrajLenList[trajX]: II = len(INDICES[:,0]); 
                    else: II = 0;
                    III = 0;
                    for Row in range(LowUpArray[0][0]+FullCumTrajLenList[trajX]-FullCumTrajLenList[trajXList[0]], 
                                     LowUpArray[-1][1]+FullCumTrajLenList[trajX]-FullCumTrajLenList[trajXList[0]]):
                      #--- LOAD INDICES if necessary
                        if MaxNumberLines < FullCumTrajLenList[trajX+1]-FullCumTrajLenList[trajX] and \
                           len(INDICES[:,0]) <= II:
                            if os.path.exists('%sINDICES_%s-%s_%s-%s.npy' % \
                                  (MatrixDir,TrajNameList[trajX],TrajNameList[trajY],LowUpArray[III][0],LowUpArray[III][1])):
                                INDICES = NP.load('%sINDICES_%s-%s_%s-%s.npy' % \
                                  (MatrixDir,TrajNameList[trajX],TrajNameList[trajY],LowUpArray[III][0],LowUpArray[III][1]))
                            else:
                                INDICES = NP.transpose(NP.load('%sINDICES_%s-%s_%s-%s.npy' % \
                                  (MatrixDir,TrajNameList[trajX],TrajNameList[trajY],LowUpArray[III][0],LowUpArray[III][1])))
                            III +=1; II = 0;
                      #--- v26.08.16 adjust indices to BeginY:EndY
                        if MaxNumberLines < FullCumTrajLenList[trajX+1]-FullCumTrajLenList[trajX] and \
                            BeginY > 0 or max(INDICES[0,:]) >= EndY: 
                            temp = INDICES[II, INDICES[II,:]>=BeginY]; temp = NP.subtract(temp[temp<EndY],BeginY);
                        else:
                            temp = INDICES[II,:]
                      #-----------
                        enhancedMatrix[Row, 2+trajYList.index(trajY)+ColIndex*len(trajYList)] = NP.sum(Weights[\
                                temp[0:EventMatrix[Row, 2+trajYList.index(trajY)+ColIndex*len(trajYList)] ], ColIndex])
                        II += 1 
##############          
## STORE: OverlapMatrix
##############
        t2=time.time()
    #---- updated_TrajLengthList
        updated_TrjLenList = []
        for trajY in range(len(TrajNameList)):
            PartY = PartList[trajY]
            if PartY == 1: ## NO SPLIT
                BeginY = StartFrame;  EndY = min(EndingFrame, TrajLenDict['%s' % (trajY)][0])
                updated_TrjLenList.append(EndY - BeginY)
            else:
                for PaY in range(1,PartY+1):
                    if PaY == 1:
                        BeginY = StartFrame;  EndY = min(EndingFrame, TrajLenDict['%s_part1' % (trajY)][0])
                    else:
                        BeginY = max(0,StartFrame-NP.sum([TrajLenDict['%s_part%s' % (trajY, elem)][0] \
                                                          for elem in range(1,PaY)]))
                        EndY   = min(EndingFrame, NP.sum([TrajLenDict['%s_part%s' % (trajY, elem)][0] \
                                                      for elem in range(1,1+PaY)]))
                        EndY   = EndY - NP.sum([TrajLenDict['%s_part%s' % (trajY, elem)][0] \
                                              for elem in range(1,PaY)])
                    updated_TrjLenList.append(max(0,EndY - BeginY))
        if NP.any(updated_TrjLenList):
    #---- define Header for the stored <EventMatrix> / <NormMatrix> / <enhancedMatrix> / <enhancedNorm>
            Rlen = max([len('r=%s' % elem) for elem in ThresholdList])
            Tlen = len('%s - %s' % (trajYList[0]+1, trajYList[-1]+1))
            HEADER = 'this file contains the <EventMatrix> / <NormMatrix> / <enhancedMatrix> / <enhancedNorm> '+\
                     'for the different trajectories calculating\n'+\
                     '\t\t>> EventMatrix/enhancedMatrix: << the amount of frames (re-weighted) within a certain threshold r\n'+\
                     '\t\t>> NormMatrix/enhancedNorm:    << the total amount of frames (re-weighted) for a certain threshold\n'+\
                         '\tcalculation time: {} seconds\n'.format(round(t2-t1,2))+\
                         '\tINPUT:\n'+\
                         '\t\tTrajNameList = {}\n'.format(TrajNameList)+\
                         '\t\tThresholdList = {}\n'.format(ThresholdList)+\
                         '\t\treference trajectories = {}\n'.format(ROW_TrajNrList if ROW_TrajNrList != None else 'ALL')+\
                         '\t\tevent counting trajectories = {}\n'.format(COL_TrajNrList if COL_TrajNrList != None else 'ALL')+\
                         '\t\tStartFrame = {}\n'.format(StartFrame)+\
                         '\t\tEndingFrame = {}\n'.format(EndingFrame)+\
                         '\t\tPartList = {}\n'.format(PartList)+\
                         '\t\taMD_Nrs = {}\n'.format(aMD_Nrs)+\
                         '\t\taMD_reweight = {}\n'.format(aMD_reweight)+\
                         '\t\tAmberVersion = {}\n'.format(AmberVersion)+\
                         '\t\tTemp = {}\n'.format(Temp)+\
                         '\t\tWeightStep = {}\n'.format(WeightStep)+\
                         '\t\tsMD_Nrs = {}\n'.format(sMD_Nrs)+\
                         '\t\tLambda = {}\n'.format(Lambda)+\
                         '\t\tOrder = {}\n'.format(Order)+\
                         '\t\tBinFile_precision = {}\n'.format(BinFile_precision)+\
                         '\t\tWeight Iterations for aMD<MF>/sMD = {}\n'.format(Iterations)+\
                         '\t\tTrajLengthList = {}\n'.format(TrajLengthList)+\
                         '\t\tupdated_TrajLengthList = {}\n'.format(updated_TrjLenList)+\
         'each row corresponds to:\n\t'+\
         'Frame | reference Traj for the corresp. frame | Events of trajY1,trajY2,...,trajYN of threshold r1 | ... of r2 | ...\n\n'+\
         'Frames | TrajNr | '+' | '.join([('r=%s' % elem).ljust(max(Rlen,Tlen)) for elem in ThresholdList])+'\n'+\
         ''.rjust(15)+(' | '+('%s - %s' % (trajYList[0]+1, trajYList[-1]+1)).ljust(max(Rlen,Tlen)))*len(ThresholdList)
            if not os.path.exists('%s%s' % (SaveDir, SaveEventMatrix)):
                NP.save('%s%s' % (SaveDir, SaveEventMatrix), EventMatrix)
            if not os.path.exists('%s%s' % (SaveDir, SaveNormMatrix)):
                NP.savetxt('%s%s' % (SaveDir, SaveNormMatrix),
                           NormMatrix,  fmt='%i %i   '+'%i '*len(trajYList)*len(ThresholdList), header=HEADER)
            if aMD_Nrs != [] or sMD_Nrs != []:
                FMT = '%i %i   '+((' '.join(['%i' if (aMD_Nrs+sMD_Nrs).count(elem+1) == 0 else '%.6f' \
                                             for elem in trajYList]))+'  ')*len(ThresholdList)
                NP.save('%s%s' % (SaveDir, SaveEnhancedMatrix), enhancedMatrix)
                if not os.path.exists('%s%s' % (SaveDir, SaveEnhancedNorm)):
                    NP.savetxt('%s%s' % (SaveDir, SaveEnhancedNorm), enhancedNorm, fmt=FMT, header=HEADER)

###############
#-------------
###############

def Extract_aMDlog(SaveDir, SaveName, aMDlogCombo, WeightStep=1, AmberVersion='Amber14', Temp=300):
    """
v12.10.16
- this supporting function extracts the Weights of the aMD.log's and stores every WeightStep-th frame
- ensure, that the same time steps are used as in the corresponding RMSD matrices
- FULL aMD.log (with corresponding WeightStep) is returned, then one need to extract Begin:End + LowerEnd:UpperEnd
INPUT:
    SaveDir      : {STRING}   Directory, where the extracted weights are stored, e.g. aMD_Weights/;
    SaveName     : {STRING}   classical savename WITHOUT ENDING, SaveName = Test -> Test.txt / Test_partX.txt;
    aMDlogCombo  : {STRING}   name of the aMD reweight file WITH DIRECTORY, standard Amber output is aMD.log, 
                                where the 6th and 7th column are used, e.g. 'Directory/aMD.log';
    WeightStep   : {INT}      <default 1>       every WeightStep-th frame from the aMDlogName is used,
                                synchronization with RMSD matrix is necessary
    AmberVersion : {STRING}   <default Amber14> Amber14 OR Amber12, because the units are different in both versions;
    Temp         : {FLOAT}    <default 300>     simulation temperature of the system (for correct units)
OUTPUT:
    - returns aMD_Weights monitored in aMD.log, whereas only the WeightStep-th frames are stored
    """
    #---- Generate Directories
    Generate_Directories(SaveDir)
    #----
    if os.path.exists('%s' % (aMDlogCombo)):
        Weights = NP.genfromtxt('%s' % (aMDlogCombo), usecols=(6,1,7))[0::WeightStep,:]
        if AmberVersion == 'Amber14':
            Weights[:,2] = NP.add(Weights[:,0], Weights[:,2])
            Weights[:,0] = NP.divide(Weights[:,2], (0.001987*Temp))
        elif AmberVersion == 'Amber12':
            Weights[:,0] = NP.add(Weights[:,0], Weights[:,2])
            Weights[:,2] = NP.multiply(Weights[:,2], (0.001987*Temp))
        HEADER = 'aMD weights extracted from \n\t>%s<\nbeta*dV [unitless] | steps [*0.002=ps] | dV [kCal/mol]' % \
                    (aMDlogCombo)
        NP.savetxt('%s%s.txt' % (SaveDir, SaveName), Weights, fmt='%.12f  %i  %.12f', header=HEADER)
        return NP.genfromtxt('%s%s.txt' % (SaveDir, SaveName), usecols=(0))
    elif os.path.exists('%s%s.txt' % (SaveDir, SaveName)):
        return NP.genfromtxt('%s%s.txt' % (SaveDir, SaveName), usecols=(0))
    else:
        raise NameError('Neither \n\t%s%s.txt\nnor\n\t%s\nexist! Check your input!' % \
                           (SaveDir, SaveName, aMDlogCombo))

###############
#-------------
###############

def Generate_Weights(TrajName, noWeights, Indices, RangeArray, SaveDir, SaveName, MatrixDir, BeginY, EndY, ThresholdList, 
                    aMD_reweight='MF', Iterations=1, Lambda=1, aMDlogCombo=None, WeightStep=1, 
                    AmberVersion='Amber14', Temp=300):
    """ 
v28.11.16
    This function calculates the >Mean-Field< approach for aMD/sMD using the 
        >Events< (noWeight) and RMSD matrix >Indices< and the standard aMD.log Weights, 
        >Indices< store the frames for the corresponding Weights
    Weights from aMD.log are ONLY "beta \Delta V", the EXPONENTIAL FUNCTION IS APPLIED IN >Generate_EventCurves<
    if aMD_reweight='MF' or sMD:
        Full noWeights for the corresponding trajNr is used with all THRESHOLDS, because the weights are 
        unique for each threshold
INPUT:
    TrajName     : {STRING}     Name of the trajectory, used for aMD_Weight_TrajName.txt;
    noWeights    : {NP.ndarray} number of counts for certain threshold, ! TrajX=TrajY, Indices.shape == (X,X) !
                                    the ROWS need to correspond to the frames counted in the COLUMNS
                                    the Indices[x,.] has to correspond to Indices[.,0];
    Indices      : {NP.ndarray} sorting Indices, storing the specific frames, 
                                    Indices[x, 0-noWeights[x,y]] returns the indices of frames r < RMSD;
    SaveDir      : {STRING}     Directory to store the weights, e.g. 'Dir/' leads to 'Dir/Weights/',  '%sWeights/%s' % (SaveDir, SaveName);
    SaveName     : {STRING}     Save name to store the weights, '%sWeights/%s' % (SaveDir, SaveName);
    MatrixDir    : {STRING}     Name of the directory, where RMSD matrices are located, e.g. 'RMSD_matrices/';
    BeginY       : {INT}        Starting frame of the trajectory, correspond to BeginY of RMSD_mat, e.g. 0;
    EndY         : {INT}        Ending frame of the trajectory,   correspond to EndY   of RMSD_mat, e.g. 2000;
    ThresholdList: {FLOAT-LIST} different thresholds r for which the Events are counted, e.g. [0.1, 0.2, 0.3, ...];
    aMD_reweight : {STRING}     <default MF>      reweighting Method -> 'MF', 'Exp', 'McL' OR 'sMD';
    Iterations   : {INT}        <default 1>         number of iterations for the Mean-Field approach of 'MF' OR 'sMD',
                                                    for Iterations=-1, it iterates until convergence or 100000 steps;
    Lambda       : {FLOAT}      <default 1>         scaling factor lambda for 'sMD', 1 means no rescaling;
    aMDlogCombo  : {STRING}     <default None>      name of the reweighting file WITH DIRECTORY 'Dir/aMD.log' produced by Amber;
    WeightStep   : {INT}        <default 1>         skip every WeightStep-th frame, i.e. Weight[0-last, every WeightStep], 
                                    HAS TO MATCH THE FRAMES EXTRACTED FROM THE TRAJECTORY WHICH ARE USED FOR RMSD_mat;
    AmberVersion : {STRING}     <default Amber14> AmberVersion, either 'Amber14' OR 'Amber12', 
                                    defines the unit of the aMD.log deltaV's;
    Temp         : {FLOAT}      <default 300>       temperature of the simulation in Kelvin;
OUTPUT:
    return Weight for correct PART, which is already included in the SaveName
            and already correct simulation time part
    for scaledMD:    SAVE the weighted Events NP.divide( NP.power(Weight,1./Lambda), Weight)
    for aMD     :    SAVE averaged weights, without NP.exp()
    the EXPONENTIAL FUNCTION is applied in <Generate_EventCurve()>
    """
#### #### #### 
#-------------
#### #### ####
    if Iterations < 0: Iterations = 100000;
    #--- aMD + <'MF'> + FILE EXIST
    if aMD_reweight == 'MF' and os.path.exists('%s%s' % (SaveDir, SaveName)):
        #---- load ThresholdList
        with open('%s%s' % (SaveDir, SaveName), 'r') as INPUT:
            for line in INPUT:
                if len(line.split()) > 2 and line.split()[0] == '#' and line.split()[1] == 'ThresholdList':
                    temp = [float(elem.replace(',','')) for elem in line[line.find('[')+1:line.find(']')].split()]
                    break
        if ThresholdList == temp:
            if len(ThresholdList) == 1:
                return NP.reshape(NP.genfromtxt('%s%s' % (SaveDir, SaveName)), (-1,1))
            else:
                return NP.genfromtxt('%s%s' % (SaveDir, SaveName))
    #--- sMD          + FILE EXIST
    if aMD_reweight == 'sMD' and os.path.exists('%s%s' % (SaveDir, SaveName)):
        #---- load ThresholdList
        with open('%s%s' % (SaveDir, SaveName), 'r') as INPUT:
            for line in INPUT:
                if len(line.split()) > 2 and line.split()[0] == '#' and line.split()[1] == 'ThresholdList':
                    temp = [float(elem.replace(',','')) for elem in line[line.find('[')+1:line.find(']')].split()]
                    break
        if ThresholdList == temp:
            if len(ThresholdList) == 1:
                return NP.reshape(NP.genfromtxt('%s%s' % (SaveDir, SaveName)), (-1,1))
            else:
                return NP.genfromtxt('%s%s' % (SaveDir, SaveName))
    #--- not (aMD + <'MF'>) and not (sMD) + FILE EXIST
    if aMD_reweight != 'MF' and aMD_reweight != 'sMD' and \
         os.path.exists('%saMD_Weight_%s.txt' % (SaveDir, TrajName)):
        return NP.genfromtxt('%saMD_Weight_%s.txt' % (SaveDir, TrajName), usecols=(0))[BeginY:EndY]
    #--- not (aMD + <'MF'>) and not (sMD) and NO aMD_Weights_%s%s.txt exist
    elif aMD_reweight != 'MF' and aMD_reweight != 'sMD': 
        ## no MF approach, weights are simply extracted from <Directory/aMD.log>
        return Extract_aMDlog('%s' % SaveDir, 'aMD_Weight_%s' % (TrajName), aMDlogCombo, 
                              WeightStep, AmberVersion)[BeginY:EndY]
    #--- (aMD + <'MF'>) or (sMD) and NO file exist
    else:
#### #### #### 
#-------------
#### #### ####
        t1 = time.time()
        if aMD_reweight != 'sMD':
            if os.path.exists('%saMD_Weight_%s.txt' % (SaveDir, TrajName)):
                Weights = NP.genfromtxt('%saMD_Weight_%s.txt' % (SaveDir, TrajName), usecols=(0))[BeginY:EndY]
            else:
                Weights = Extract_aMDlog('%s' % SaveDir, 'aMD_Weight_%s' % (TrajName), aMDlogCombo, 
                                         WeightStep, AmberVersion, Temp)[BeginY:EndY]
            ## INIT Weights_MF (0. step) with "beta deltaV" from Weights
            Weights_MF1 = NP.zeros( noWeights.shape )
            for RadIndex in range(len(noWeights[0,:])):
                Weights_MF1[:,RadIndex] = NP.copy(Weights)
            #del Weights
        elif aMD_reweight == 'sMD':
            ## INIT Weights_MF (0. step) with 1: w_k^(0) = 1
            Weights_MF1 = NP.ones( noWeights.shape )
        ## INIT RadList: if for specific threshold the Weights are already converged, delete it from RadList
        RadList = range(len(noWeights[0,:]))
        Iter = 0
        ConvIter = [-1]*len(RadList)
#### #### #### 
#-------------
#### #### ####
      #### v05.09.16 variable which monitors, if Indices has to be loaded
        IndicesLoad = False
      ####
        while (Iter < Iterations) and len(RadList) != 0:
            ## Store current Weights_MF (n) step to control convergency
            Weights_MF0 = NP.copy(Weights_MF1)
         #### v25.08.16: LOAD INDICES IF NECESSARY, take care about noWeights = Ev[BeginY:EndY,:]
            III = 0
            while III < len(RangeArray) and RangeArray[III][1] < BeginY:
                III += 1
            if os.path.exists('%sINDICES_%s-%s_%s-%s.npy' % \
                                  (MatrixDir, TrajName, TrajName, RangeArray[III][0], RangeArray[III][1])):
                Indices = NP.load('%sINDICES_%s-%s_%s-%s.npy' % \
                                  (MatrixDir, TrajName, TrajName, RangeArray[III][0], RangeArray[III][1]))
                IndicesLoad = True
            #--- adjust lower index to BeginY
                if RangeArray[III][0] < BeginY and RangeArray[III][1] > BeginY:
                    Indices = Indices[(BeginY-RangeArray[III][0]):,:]
         ####
            IIII = 0
            for Rows in range(len(noWeights[:,0])):
              ####
                if RangeArray[III][1] <= BeginY+Rows and \
                   os.path.exists('%sINDICES_%s-%s_%s-%s.npy' % \
                              (MatrixDir, TrajName, TrajName, RangeArray[III+1][0], RangeArray[III+1][1])):
                    III +=1 ; Indices = NP.load('%sINDICES_%s-%s_%s-%s.npy' % \
                              (MatrixDir, TrajName, TrajName, RangeArray[III][0], RangeArray[III][1]))
                    IIII = 0
                    IndicesLoad = True
              #---26.08.16 adjust indices to BeginY:EndY
                if IndicesLoad:
                    temp = Indices[IIII, Indices[IIII,:]>=BeginY]; temp = NP.subtract(temp[temp<EndY],BeginY);
                else:
                    temp = Indices[IIII,:]
              #-----          
                for RadIndex in RadList:
                  #-----          
                    if aMD_reweight != 'sMD':
                        ## for Mean-Field: calculate average deltaV for each reference frame
                        Weights_MF1[Rows, RadIndex] = \
                                NP.divide( NP.sum(Weights_MF0[temp[0:noWeights[Rows, RadIndex]], RadIndex]),
                                                 noWeights[Rows, RadIndex] )
                    elif aMD_reweight == 'sMD':
                        ## Indices[Rows, 0:Events] returns frames -> Weights_MF0[Rows, frames] weights the frames
                        weightedEvents = NP.sum( Weights_MF0[temp[0:noWeights[Rows, RadIndex]], RadIndex] )
                        Weights_MF1[Rows, RadIndex] = NP.power(weightedEvents, 1./Lambda-1)
                        #Weights_MF1[Rows, RadIndex] = NP.divide( NP.power(weightedEvents, 1./Lambda),
                        #                                         noWeights[Rows, RadIndex] )
                IIII += 1
              ####
            Iter += 1
            ##### ##### ##### 
            # CHECK convergence of Weights_MF1 for every Threshold, pop() if convergence reached
            ##### ##### ##### 
            Delete = []
            for RadIndex in RadList:
                if NP.all(NP.absolute(Weights_MF1[:,RadIndex]-Weights_MF0[:,RadIndex]) < 10e-6):
                    Delete.append(RadIndex)
                    ConvIter[RadIndex] = Iter
            if Delete != []:
                RadList = [elem for elem in RadList if Delete.count(elem) == 0]
#### #### ####
#-------------
#### #### ####
        t2 = time.time()
        if len(RadList) == 0:
            Conv = 'The weight calculation converged in %s iterations to < 10e-6, for the corresponding threshold' % ConvIter 
        else:
            if Iterations >= 100000:
                print '####\nWARNING: the Weights did not converge for all thresholds! Check the WeightFile for further information\n###'
            Conv = 'Maximal iterations (= %s) for the weight calculation are reached. -1 means, the Weight did not converge after 100000 steps for the corresponding threshold' % ConvIter
        if SaveDir is not None and SaveName is not None:
            Generate_Directories(SaveDir)
            HEADER = 'generated Weights for '+aMD_reweight+'\n'+\
                     'ThresholdList = {}\n'.format(ThresholdList)+\
                     'elapsed time: {} seconds\n'.format(round(t2-t1,2))+\
                     'aMD is calculated by\n'+\
                     '\t\tsum_k(\Delta V_k)/N\n'+\
                     'and needs only be further proceeded by "NP.exp()"\n'+\
                     'sMD is calculated by\n'+\
                     '\t\tcounts^(1./lambda)/counts with lambda = '+str(Lambda)+'\n'+\
                     Conv+'\n'+\
                     'r='+' | r='.join([str(elem) for elem in ThresholdList])
            NP.savetxt('%s%s' % (SaveDir, SaveName), Weights_MF1, fmt='%.12f'+' %.12f'*len(Weights_MF1[0,1:]),
                       header=HEADER)
        if len(ThresholdList) == 1:
            return NP.reshape(NP.genfromtxt('%s%s' % (SaveDir, SaveName)), (-1,1))
        else:
            return NP.genfromtxt('%s%s' % (SaveDir, SaveName))

###############
#-------------
###############

def Generate_TrajLenDict(LengthTrajNameList, PartList, TrajLengthList, StartFrame=0, EndingFrame=NP.infty):
    """
v06.09.16
Helper function to return TrajLenDict[trajX_PartX] = (len(trajX_PartX), Index, EndX-BeginX)
    """
    if PartList is None:
        PartList = [1]*LengthTrajNameList
    #------    
    TrajLenDict = {}; Index = 0
    for trajX in range(LengthTrajNameList):
        for PartX in [(('_part%s' % Part) if PartList[trajX]!=1 else '') for Part in range(1,PartList[trajX]+1)]: 
            TrajLenDict['%s%s' % (trajX, PartX)] = (TrajLengthList[Index], Index, TrajLengthList[Index])
            Index += 1
  ##############
    if StartFrame != 0 or EndingFrame != NP.infty:
        for trajX in range(LengthTrajNameList):
            for PartX in [(('_part%s' % Part) if PartList[trajX]!=1 else '') for Part in range(1,PartList[trajX]+1)]: 
                if PartX == '' or PartX == '_part1': ## NO SPLIT or PART1 for trajX ##
                    BeginX = StartFrame;  EndX = min(EndingFrame, TrajLenDict['%s%s' % (trajX, PartX)][0])
                else:
                    BeginX = max(0,StartFrame-NP.sum([TrajLenDict['%s_part%s' % (trajX, elem)][0] \
                                                  for elem in range(1,int(PartX.split('_part')[1]))]))
                    EndX   = min(EndingFrame, NP.sum([TrajLenDict['%s_part%s' % (trajX, elem)][0] \
                                                  for elem in range(1,1+int(PartX.split('_part')[1]))]))
                  # v03.06.16: EndY must be subtracted by the LENGTHS of all SMALLER parts
                    EndX   = EndX - NP.sum([TrajLenDict['%s_part%s' % (trajX, elem)][0] \
                                                  for elem in range(1,int(PartX.split('_part')[1]))])
                TrajLenDict['%s%s' % (trajX, PartX)] = (TrajLenDict['%s%s' % (trajX, PartX)][0], 
                                                        TrajLenDict['%s%s' % (trajX, PartX)][1],
                                                        0 if EndX < BeginX else EndX-BeginX)
  ##############
    return TrajLenDict

#######################################################
#------------------------------------------------------
#---             RMSD matrices
#------------------------------------------------------
#######################################################
def Generate_RMSD_Matrix(TrajDir, TopologyDir, TrajName, TopologyName, DistSaveDir, MatrixSaveDir, SaveName,  
                        Select1, Select2=None, TimeStep=None, AmberHome='', GromacsHome='', Begin=None, End=None, 
                        SecondTraj=None, Fit='rot+trans', Program_Suffix='', ReferencePDB=None, Bin=True):
    """
v12.10.16
    - RMSD matrix generation for given Trajectory using Gromacs v4.6.7|5.1.2 or AmberTools14
    - it tries to automatically detect AMBER/GROMACS Trajs:
        1. if Select2 is     None or TrajName = <.netcdf or .nc> -> AMBER
        2. if Select2 is not None or TrajName = <.xtc or .trr>   -> GROMACS
    - Trajectories should already be preprocessed, making the protein whole etc.
    - if TimeStep is None: ALL Frames of the trajectory are used, otherwhise
        - INT   for AMBER  : use Frames 'first last skip', e.g. TimeStep=2, every 2nd Frame is used
        - FLOAT for GROMACS: use every TimeStep-th time in [ns]
    - for very large trajectories (>> 10000 frames) or low memory
        --> split trajectories into smaller parts BY HAND like 
            TrajName -> TrajName_partX : e.g. MD.xtc -> MD_part1.xtc, MD_part2.xtc, ...
INPUT:
    TrajDir        : {STRING}    Directory, where the trajectory is located, e.g. FullTrajs/;
    TopologyDir    : {STRING}    Directory, where the topology   is located, e.g. FullTrajs/;
    TrajName       : {STRING}    Name of the trajectory, e.g. MD.xtc | MD.netcdf
                                    it tries to automatically detect AMBER/GROMACS Trajectories by the ending
                                    1. .nc  or .netcdf -> AMBER
                                    2. .xtc or .trr    -> GROMACS
                                    3. if Select2 is None -> AMBER else GROMACS;
    TopologyName   : {STRING}    Name of the Topology file, WITH ENDING .pdb|.tpr (GROMACS) OR .top (AMBER);
    DistSaveDir    : {STRING}    Directory, where the distributions are stored (Gromacs), e.g. RMSD_files/;
    MatrixSaveDir  : {STRING}    Directory, where the RMSD matrices are stored, e.g. RMSD_matrices/;
    SaveName       : {STRING}    SaveName prefix, e.g. 'MD_All' -> RMSD matrix 'MD_All_bin.dat' | 'MD_All_dist.xvg';
    Select1        : {STRING}    FIT  selection Gromacs, e.g. 'Backbone'
                                 CALC selection Amber  , e.g. '@N,CA,C', see Amber syntax;
    Select2        : {STRING}    CALC selection Gromacs, e.g. 'Backbone';
    TimeStep       : {FLOAT/INT} <default None> GROMACS, -dt 't MOD dt = first [ns]' 
                                                AMBER    skip dt-th frame 'first last skip'
                                                if TimeStep = None, use ALL frames in the trajectory;
    AmberHome      : {STRING}    <default ''> if cpptraj is not in environmental variables, define directory to 'cpptraj',
                                    e.g. /home/user/Software/amber14/bin/;
    GromacsHome    : {STRING}    <default ''> if g_rms | gmx_suffix rms is not in environmental variables, define directory to them,
                                    e.g. /home/user/Software/gromacs/bin/;
    Begin          : {FLOAT/INT} <default None> starting Frame [ns Gromacs | frame Amber]      for Trajectory, 
                                    e.g. '0.01 | 1';
    End            : {FLOAT/INT} <default None> ending   Frame [ns Gromacs | frame/last Amber] for Trajectory, 
                                    e.g. '200  | last';
    SecondTraj     : {STRING}    <default None> Name of a possible second trajectory, e.g. 'MD2.xtc' | 'MD2.netcdf'
                                    to calculate Traj1 vs Traj2 RMSD matrix;
    Fit            : {STRING}    <default rot+trans>, selects, if trajectories are FIT on top of each other or not, 'none' or 'rot+trans';
    Program_Suffix : {STRING}    <default ''> gmx_suffix for Gromacs installation, e.g. '_467' for 'g_rms_467';
    ReferencePDB   : {STRING}    <default None>, only for AMBER, RMSD to reference calculation, WITH possible DIRECTORY
                                    e.g. Directory/Reference.pdb;
    Bin            : {BOOL}      <default True>, IF TRUE, RMSD matrix is calculated, else only RMSD curves;
OUTPUT:
    RMSD matrices (+ RMSD distribtions for Gromacs) in *SaveDir
    """
    if (not os.path.exists('%s%s.xvg' % (DistSaveDir, SaveName)) or \
        not os.path.exists('%s%s_bin.dat' % (MatrixSaveDir, SaveName))) and \
       (os.path.exists('%s%s' % (TrajDir, TrajName)) and \
        os.path.exists('%s%s' % (TopologyDir, TopologyName))):
        t1 = time.time()
     #### AMBER ####
        if Select2 is None or TrajName.split('.')[-1] == 'netcdf' or TrajName.split('.')[-1] == 'nc': # AMBER14
            if TimeStep is None:
                TimeStep = 1
          #---- Error Detection for wrong values of >Begin | End | TimeStep<
            if type(TimeStep) != int:
                print (('The >TimeStep = %s< you have specified is not an >int<\n' % TimeStep)+\
                                  'every TimeStep-th frame is used from the trajectory, try >1< for all frames')
                raise ValueError(('The >TimeStep = %s< you have specified is not an >int<\n' % TimeStep)+\
                                  'every TimeStep-th frame is used from the trajectory, try >1< for all frames')
            if Begin is not None and type(Begin) != int:
                print (('The >Begin = %s< you have specified is not an >int<\n' % Begin)+\
                                  'define the starting frame of the trajectory, try >1< for all frames')
                raise ValueError(('The >Begin = %s< you have specified is not an >int<\n' % Begin)+\
                                  'define the starting frame of the trajectory, try >1< for all frames')
            if End is not None and type(End) != int and End != 'last':
                print (('The >End = %s< you have specified is not an >int< or not "last"\n' % End)+\
                                  'define the ending frame of the trajectory, try >"last"< for all frames')
                raise ValueError(('The >End = %s< you have specified is not an >int< or not "last"\n' % End)+\
                                  'define the ending frame of the trajectory, try >"last"< for all frames')
            if not os.path.exists('%scpptraj' % AmberHome) and not os.path.exists('%s/bin/cpptraj' % (os.environ.copy()['AMBERHOME'])):
                print ('You are trying to calculate RMSD (matrix) using Amber\n\t'+\
                                ('>> %scpptraj << not found\n' % AmberHome)+\
                                ('CHECK your >> AmberHome=%s<< value' % AmberHome))
                raise NameError('You are trying to calculate RMSD (matrix) using Amber\n\t'+\
                                ('>> %scpptraj << not found\n' % AmberHome)+\
                                ('CHECK your >> AmberHome=%s<< value' % AmberHome))
          #---- generate Directories  
            Generate_Directories('%sLogs/' % MatrixSaveDir)
          #---- generate INPUT file for CPPTRAJ
            with open('%s%s.in' % (MatrixSaveDir, SaveName), 'w') as OUTPUT:
                OUTPUT.write('trajin %s%s %s %s %s\n' % (TrajDir, TrajName, 
                                                         Begin if Begin is not None else 1, 
                                                         End if End is not None else 'last',
                                                         TimeStep))
                if ReferencePDB is not None:
                    if not os.path.exists(ReferencePDB):
                        print ('>>%s<< \ndoes not exist! Check the directory and name' % (ReferencePDB))
                        raise NameError('>>%s<< \ndoes not exist! Check the directory and name' % (ReferencePDB))
                    OUTPUT.write('reference %s\n' % (ReferencePDB))
                    OUTPUT.write('rms reference mass out %s%s.xvg %s %s' % \
                                 (DistSaveDir, SaveName, 
                                  Select1, 
                                  '' if Fit is not None else 'nofit'))
                if Bin:
                    OUTPUT.write('rms2d out %s%s_bin.dat mass %s %s %s %s%s\n' % \
                                 (MatrixSaveDir, SaveName, 
                                  Select1, 
                                  '' if Fit is not None else 'nofit',
                                  '' if SecondTraj is None else 'reftraj',
                                  '' if SecondTraj is None else TrajDir,
                                  '' if SecondTraj is None else SecondTraj))
            if os.path.exists('%scpptraj' % AmberHome):
                CPPTRAJ = SB.Popen(['%scpptraj' % AmberHome, '-p', '%s%s' % (TopologyDir, TopologyName), 
                                                 '-i', '%s%s.in' % (MatrixSaveDir, SaveName)],
                                  stdout=SB.PIPE, stderr=SB.STDOUT)
            elif os.path.exists('%s/bin/cpptraj' % (os.environ.copy()['AMBERHOME'])):
                CPPTRAJ = SB.Popen(['%s/bin/cpptraj' % (os.environ.copy()['AMBERHOME']), '-p', '%s%s' % (TopologyDir, TopologyName), 
                                                 '-i', '%s%s.in' % (MatrixSaveDir, SaveName)],
                                  stdout=SB.PIPE, stderr=SB.STDOUT)
            Out, _ = CPPTRAJ.communicate()
            with open('%sLogs/LOG_%s.log' % (MatrixSaveDir, SaveName), 'w') as LOG_OUT:
                LOG_OUT.write(Out)
          #---- 
        elif Select2 is not None or TrajName.split('.')[-1] == 'xtc' or TrajName.split('.')[-1] == 'trr': # GROMACS
     #### GROMACS ####
            if not os.path.exists('%sg_rms%s' % (GromacsHome, Program_Suffix)) and \
               not os.path.exists('%s/g_rms%s' % (os.environ.copy()['GMXBIN'], Program_Suffix)) and \
               not os.path.exists('%sgmx%s' % (GromacsHome, Program_Suffix)) and \
               not os.path.exists('%s/gmx%s' % (os.environ.copy()['GMXBIN'], Program_Suffix)):
                print ('You are trying to calculate RMSD (matrix) using Gromacs\n\t'+\
                                ('Neither >>%sg_rms%s<< nor >>%sgmx%s<< found\n' % \
                                (GromacsHome, Program_Suffix, GromacsHome, Program_Suffix))+\
                                ('CHECK your >>GromacsHome=%s<< and >>Program_Suffix=%s<< value' % (GromacsHome, Program_Suffix)))
                raise NameError('You are trying to calculate RMSD (matrix) using Gromacs\n\t'+\
                                ('Neither >>%sg_rms%s<< nor >>%sgmx%s<< found\n' % \
                                (GromacsHome, Program_Suffix, GromacsHome, Program_Suffix))+\
                                ('CHECK your >>GromacsHome=%s<< and >>Program_Suffix=%s<< value' % (GromacsHome, Program_Suffix)))
          #---- generate Directories  
            Generate_Directories('%sLogs/' % MatrixSaveDir)
            Generate_Directories(DistSaveDir)
          #---- Command for RMS
            if os.path.exists('%sg_rms%s' % (GromacsHome, Program_Suffix)): # gromacs v4.6.7
                Command = ['%sg_rms%s' % (GromacsHome, Program_Suffix)]
            elif os.path.exists('%s/g_rms%s' % (os.environ.copy()['GMXBIN'], Program_Suffix)): # gromacs v4.6.7
                Command = ['%sg_rms%s' % (GromacsHome, Program_Suffix)]
            elif os.path.exists('%sgmx%s' % (GromacsHome, Program_Suffix)):
                Command = ['%sgmx%s' % (GromacsHome, Program_Suffix), 'rms']
            elif os.path.exists('%s/gmx%s' % (os.environ.copy()['GMXBIN'], Program_Suffix)):
                Command = ['%sgmx%s' % (os.environ.copy()['GMXBIN'], Program_Suffix), 'rms']
            Command.extend(['-f', '%s%s' % (TrajDir, TrajName),
                            '-s', '%s%s' % (TopologyDir, TopologyName),
                            '-o', '%s%s.xvg' % (DistSaveDir, SaveName),
                            '-tu', 'ns', '-xvg', 'none'])
            if TimeStep is not None:
                Command.extend(['-dt', '%s' % TimeStep])
            if SecondTraj is not None:
                Command.extend(['-f2', '%s%s' % (TrajDir, SecondTraj)])
            if Begin is not None:
                Command.extend(['-b', '%s' % Begin])
            if End is not None:
                Command.extend(['-b', '%s' % End])
            if Fit != 'rot+trans':
                Command.extend(['-fit', '%s' % Fit])
          #---- Command for DIST + MATRIX
            if Bin:
                Generate_Directories('%sXpmFiles/' % DistSaveDir)
                Command.extend(['-m', '%sXpmFiles/%s.xpm' % (DistSaveDir, SaveName),
                                '-bin', '%s%s_bin.dat' % (MatrixSaveDir, SaveName),
                                '-dist', '%s%s_dist.xvg' % (DistSaveDir, SaveName)])
          #---- RUN GROMACS rms generation using a subprocess
            RMSD = SB.Popen(Command, stdin=SB.PIPE, stdout=SB.PIPE, stderr=SB.STDOUT)
            Out, _ = RMSD.communicate('%s\n%s\n' % (Select1, Select2))
            with open('%sLogs/LOG_%s.log' % (MatrixSaveDir, SaveName), 'w') as LOG_OUT:
                LOG_OUT.write(Out)
        t2 = time.time()
    else:
        print 'Either \n\t>>%s%s.xvg<< &\n\t>>%s%s_bin.dat<<\nalready exists or \n\t>>%s%s<< & >>%s%s<< \ndoes NOT exist' % \
                (DistSaveDir,SaveName,  MatrixSaveDir,SaveName,  TrajDir,TrajName,  TopologyDir,TopologyName)

###############
#-------------
###############

def Generate_RMSD_Matrices(TrajDir, TopologyDir, TrajNameList, TopologyName, DistSaveDir, MatrixSaveDir, 
                          TimeStep, Select1, Select2=None, AmberHome='', GromacsHome='',
                          Fit='rot+trans', Program_Suffix='', PartList=None):
    """ 
v22.11.16 
    - using <Generate_RMSD_Matrix()>
    - Calculates every RMSD diagonal & off-diagonal matrix part for <Generate_EventCurves()>
    - needs trajectories in '.xtc', '.trr', '.pdb', '.nc', '.netcdf' or '.dhd' format, then automatically either Gromacs or AmberTools is used
        otherwise, the RMSD matrices can be generated by hand fulfilling the following syntax:
            TrajNameList = [Traj1, Traj2, Traj3, ...] -> Traj1_bin.dat, Traj1_Traj2_bin.dat, ... Traj1_TrajN_bin.dat
                                                         Traj2_bin.dat, Traj2_Traj3_bin.dat, ... Traj2_TrajN_bin.dat
                                                         TrajN_bin.dat
        TrajX_bin.dat       is the self-RMSD-matrix between all frames of trajX
        TrajX_TrajY_bin.dat is the      RMSD-matrix between all frames of trajX vs all frames of trajY                                                            
    - RMSD matrix generation for given Trajectory-List using Gromacs v4.6.7|5.1.2 or AmberTools14
    - Trajectories should already be preprocessed, making the protein whole, strip e.g. unnecessary residues, etc.
    - uses the FULL trajectories without Begin | End
    - for very large trajectories (>> 10000 frames) or low memory
        (1) split trajectories into smaller parts BY HAND like 
            TrajName -> TrajName_partX : e.g. MD.xtc -> MD_part1.xtc, MD_part2.xtc, ...
        (2) submit a >> PartList = [1,5,2,7, ...] << which defines automatically the number of splits
            -> MD1.xtc -> MD1.xtc | MD2.xtc -> MD2_part1.xtc, ..., MD2_part5.xtc | MD3.xtc -> MD3_part1.xtc, ...
        (3) This PartList works also for the EventCurves & effective Clustering
            -> loading corresponding parts one-by-one into the RAM

INPUT:
    TrajDir        : {STRING}    Directory, where the trajectory is located, e.g. FullTrajs/;
    TopologyDir    : {STRING}    Directory, where the topology   is located, e.g. FullTrajs/;
    TrajNameList   : {LIST}      List of Names of the trajectories, WITH ENDING, e.g. ['MD1.xtc', 'MD2.netcdf', ...]
                                    it tries to automatically detect AMBER/GROMACS Trajectories by the ending
                                    1. .nc  or .netcdf -> AMBER
                                    2. .xtc or .trr    -> GROMACS
                                    3. if Select2 is None -> AMBER else GROMACS;
    TopologyName   : {STRING}    Name of the Topology file, WITH ENDING .pdb|.tpr (GROMACS) OR .top (AMBER);
    DistSaveDir    : {STRING}    Directory, where the distributions are stored (Gromacs), e.g. RMSD_files/, 
                                    FOR AMBER (trajectories) THIS VALUE IS NOT NECESSARY and can be set to 'None';
    MatrixSaveDir  : {STRING}    Directory, where the RMSD matrices are stored, e.g. RMSD_matrices/;
    TimeStep       : {FLOAT/INT} -dt for GROMACS 't MOD dt = first [ns]' OR skip dt-th frame for AMBER 'first last skip',
                                            if TimeStep = None, use ALL frames in the trajectory;
    Select1        : {STRING}    FIT  selection Gromacs, e.g. 'Backbone'
                                 CALC selection Amber  , e.g. '@N,CA,C', see Amber syntax;
    Select2        : {STRING}    CALC selection Gromacs, e.g. 'Backbone', FOR AMBER (trajectories), THIS VALUE IS NOT NECESSARY and should be set to 'None';
    AmberHome      : {STRING}    <default ''> if cpptraj is not in environmental variables, define directory to 'cpptraj',
                                    e.g. /home/user/Software/amber14/bin/;
    GromacsHome    : {STRING}    <default ''> if g_rms | gmx_suffix rms is not in environmental variables, define directory to them,
                                    e.g. /home/user/Software/gromacs/bin/;
    Fit            : {STRING}    <default rot+trans>, selects, if trajectories are FIT on top of each other or not, 'none' or 'rot+trans';
    Program_Suffix : {STRING}    <default ''> gmx_suffix for Gromacs installation, e.g. '_467' for 'g_rms_467';
    PartList       : {INT-LIST}  <default None> defines into how many parts the single trajectories are split, due to
                                    memory reasons,
                                        1. len(PartList) == len(TrajNameList) !!  ||   
                                        2. PartList = [1,2,3], (MD1.xtc -> MD1.xtc),
                                                               (MD2.xtc -> MD2_part1.xtc, MD2_part2.xtc),
                                                               (MD3.xtc -> MD3_part1.xtc, MD3_part2.xtc, MD3_part3.xtc), ||
                                        3. default PartList = None -> PartList = [1]*len(TrajNameList);    
OUTPUT:
    RMSD matrices (+ RMSD distribtions for Gromacs) in *SaveDir using multiprocess
"""
  #---- generate Directories  
    Generate_Directories('%sLogs/' % MatrixSaveDir)
    Generate_Directories(DistSaveDir)
    ######## ENABLE MULTIPROCESSING: Pool() enables all free nodes
    pool = Pool()
    ######## test if the trajectories are called differently ######## 
    if len(TrajNameList) != len(NP.unique(TrajNameList)): ## test if the trajectories are called differently
        raise NameError('List of trajectories contains multiply same Names, change the trajectory names for differentiation')
    ######## generate TrajName with Splits ########
    if PartList is not None and len(PartList) == len(TrajNameList):
        if sum([elem-1. for elem in PartList]) > 10e-5:     ## PartList does not contain only 1
            TrajNameList_Parts = []
            for Kai in range(len(PartList)):
                for Parts in range(PartList[Kai]):
                    if PartList[Kai] > 1:
                        TrajNameList_Parts.append(TrajNameList[Kai]+'_part'+str(Parts+1))
                  ## UPDATED v02.10.15: if there are also NO splits with other splitted trajs, these are now included, too
                    else:
                        TrajNameList_Parts.append(TrajNameList[Kai])
            TrajNameList = [elem for elem in TrajNameList_Parts]
            del TrajNameList_Parts
    elif PartList is not None and len(PartList) != len(TrajNameList):
        raise ValueError('the submitted >PartList< does not match the length of the >TrajNameList<')
    ######## ######## ########
    for Kai in range(len(TrajNameList)):
        if not os.path.exists('%s%s' % (TrajDir, TrajNameList[Kai])):
            raise NameError('Trajectory does not exist\n\t> %s%s <' % (TrajDir, TrajNameList[Kai]))
        #----- extract possible standard endings
        for Ending in ['.xtc', '.trr', '.pdb', '.nc', '.netcdf', '.dhd']:
            if TrajNameList[Kai].find(Ending) != -1:
                SaveName1 = TrajNameList[Kai].split(Ending)[0]
        #######
        ## Diagonal = SINGLE Trajectory RMSD matrices
        #######
        if not os.path.exists('%s%s_bin.dat' % (MatrixSaveDir, SaveName1)):
            pool.apply_async(Generate_RMSD_Matrix, 
                             args=(TrajDir, TopologyDir, TrajNameList[Kai], TopologyName, DistSaveDir, MatrixSaveDir, 
                                   SaveName1, Select1, Select2, TimeStep, AmberHome, GromacsHome, None, None, None, 
                                   Fit, Program_Suffix, None, True))
        ##
        for Kai2 in range(Kai+1,len(TrajNameList),1):
            if not os.path.exists('%s%s' % (TrajDir, TrajNameList[Kai2])):
                raise NameError('Trajectory does not exist\n\t> %s%s <' % (TrajDir, TrajNameList[Kai2]))
        #######
        ## Off-Diagonal
        #######
            if Kai2 > Kai:
                #----- extract possible standard endings
                for Ending in ['.xtc', '.trr', '.pdb', '.nc', '.netcdf', '.dhd']:
                    if TrajNameList[Kai].find(Ending) != -1 and TrajNameList[Kai2].find(Ending) != -1:
                        SaveName2 = TrajNameList[Kai].split(Ending)[0]+'_'+TrajNameList[Kai2].split(Ending)[0]
                if not os.path.exists('%s%s_bin.dat' % (MatrixSaveDir, SaveName2)):
                    pool.apply_async(Generate_RMSD_Matrix,
                                     args=(TrajDir, TopologyDir, TrajNameList[Kai], TopologyName, DistSaveDir, 
                                           MatrixSaveDir, SaveName2, Select1, Select2, TimeStep, AmberHome, GromacsHome, 
                                           None, None, TrajNameList[Kai2], Fit, Program_Suffix, None, True))
    #---- Wait until all jobs are done
    pool.close()
    pool.join()

#######################################################
#------------------------------------------------------
#---          RMSD distribution
#------------------------------------------------------
#######################################################
def determineR_extract_MaxRMSD(TrajNameList, SaveName, 
                               RMSD_dist_Dir = 'Amber14Trajs/RMSD_files/',
                               MatrixDir = 'Amber14Trajs/RMSD_files/BinFiles/',
                               SaveDir = 'Amber14Trajs/RMSD_distributions/',
                               BinFile_precision=NP.float32, RMSD_SaveAdder=''):
    """
v12.10.16
    extract and store MaxRMSD for the RMSD histrograms
    (1) EITHER uses the last "BIN" of RMSD_dist (GROMACS)
    (2) OR screen through every RMSD matrix
    (3) searches for: TrajName_dist.xvg AND TrajName_bin.dat | 
                      TrajName1_TrajName2_dist.xvg AND TrajName1_TrajName2_bin.dat

INPUT:
    TrajNameList       : {LIST}    list of Trajectory name prefixes WITHOUT ENDING, 
                                    Names of the Block RMSD matrices / distributions refer to these names,
                                    e.g. ['MD1', 'MD2', ...] <-> MD1_bin.dat, MD2_bin.dat, MD1_MD2_bin.dat, ...;
    SaveName           : {STRING}  Savename, e.g. 'V3' leads to 'MaxMinRMSD_V3.txt', 'MaxMinRMSD_%s.txt' % (SaveName);
    RMSD_dist_Dir      : {STRING}  <default ''> Directory, where (possible) RMSD distributions are located (from GROMACS), e.g. 'RMSD_files/', 
                                    if it does not exist, RMSD matrices are used instead;
    MatrixDir          : {STRING}  Directory, where RMSD matrices are stored, e.g. 'RMSD_matrices/';
    SaveDir            : {STRING}  <default RMSD_distributions/>, Directory, where the maximal RMSD value is stored;
    BinFile_precision  : {TYPE}    FORMAT, [GROMACS/BINARY] single precision 'float32', double precision 'float64', [AMBER/ELSE] 'None';
OUTPUT:
    return MaxRMSD and STORE it to <SaveDir>

added: 
    - RMSD_bin.dat extraction, if not in binary from GMX v4.6.7 OR GMX v5.1.2 (generates the same _bin.dat RMSD matrices)
    - for AMBERTOOLS: RMSD_matrix is an N x (N+1) shape matrix, FIRST COLUMN = index
    
    """
    MaxRMSD = 0
 #------------- DIRECTORY
    Generate_Directories(SaveDir)
#-------------------------------------
 #---- EXTRACT MAXIMAL RMSD VALUE 
#-------------------------------------
    if os.path.exists('%sMaxMinRMSD_%s.txt' % (SaveDir, SaveName)):
      #-- TEST if the same TrajNameList is used in the File, otherwise produce it anew
        with open('%sMaxMinRMSD_%s.txt' % (SaveDir, SaveName), 'r') as INPUT:
            for line in INPUT:
                if line.split()[1] == 'TrajNameList':
                    if '# TrajNameList = {}\n'.format(TrajNameList) == line:
                        try: # first value = MaxRMSD | snd value = MinRMSD
                            MaxRMSD = NP.genfromtxt('%sMaxMinRMSD_%s.txt' % (SaveDir, SaveName))[0]
                        except IndexError: # only MaxRMSD exists
                            MaxRMSD = NP.float(NP.genfromtxt('%sMaxMinRMSD_%s.txt' % (SaveDir, SaveName)))
                    break    
   #-------- if MaxRMSD not already calculated: GENERATE MaxRMSD       
    if MaxRMSD == 0:
        for Kai in range(len(TrajNameList)):
         #-- DIAGS
            # TRY TO LOAD RMSD_dist.xvg
            if os.path.exists('%s%s%s_dist.xvg' % (RMSD_dist_Dir, TrajNameList[Kai], RMSD_SaveAdder)):
              # g_rms -dist generates always 101 bins
                temp = NP.genfromtxt('%s%s%s_dist.xvg' % (RMSD_dist_Dir, TrajNameList[Kai], RMSD_SaveAdder), 
                                     skip_header=50) # skip_header, if XVG header is used
                MaxRMSD = max(MaxRMSD, temp[-1,0])
                del temp
            # TRY TO LOAD RMSD_bin.dat
            elif os.path.exists('%s%s%s_bin.dat' % (MatrixDir, TrajNameList[Kai], RMSD_SaveAdder)):
                if BinFile_precision is None:
                  #----- TRY TO LOAD NP.NDarray like input, like generated by AmberTools14
                    try:
                        temp = NP.genfromtxt('%s%s%s_bin.dat' % (MatrixDir, TrajNameList[Kai], RMSD_SaveAdder))
                    except ValueError:
                        raise ValueError('The RMSD matrix is not stored in a txt-file.\n\t'+\
                                         ('Check your RMSD matrix format and >>BinFile_precision=%s<< input\n' % BinFile_precision)+\
                                         '\'None\' corresponds to txt-file Matrix shape format\n'+\
                                         '\'float32/64\' corresponds to GROMACS binary format')
                    if len(temp.shape) != 2:
                        raise ValueError('The RMSD matrix is not stored in Matrix form, (Rows)x(Columns)\n\tCheck your matrices stored in >>%s<<' % MatrixDir)
                   #--- if FIRST COL = Indices: strip them (AmberTools14)
                    if NP.sum(NP.absolute(temp[:,0]-range(1,len(temp[:,0])+1))) < 10e-6: 
                       #--- AmberTool14 uses Angstrom, thus /10 for NM
                        MaxRMSD = max(MaxRMSD, NP.max(temp[:,1:])/10.)
                    else:
                        MaxRMSD = max(MaxRMSD, NP.max(temp))               
                else:
                    temp = NP.fromfile('%s%s%s_bin.dat' % (MatrixDir, TrajNameList[Kai], RMSD_SaveAdder), dtype=BinFile_precision)
                    if len(temp) != int(NP.sqrt(len(temp)))*int(NP.sqrt(len(temp))):
                        raise ValueError(('The RMSD matrix generated by GROMACS is not loaded correctly!\n\tCheck your >>BinFile_precision=%s<< input\n' % \
                                BinFile_precision)+\
                              'The BinFile_precision must correspond to the precision which was used to install GROMACS'+\
                              '\n\t\'float32\' corresponds to single precision\n\t\'float64\' corresponds to double precision!\n'+\
                              'This was tested for GROMACS v4.6 and v5.1, for other versions the RMSD matrices might differ and have to be preprocessed by hand\n'+\
                              'and stored in matrix-form in a txt-file using >>BinFile_precision=None<<')
                    MaxRMSD = max(MaxRMSD, NP.max(temp))
                del temp
            else:
                raise NameError('Error extracting the maximal RMSD for the given TrajNameList: \n'+\
                    'Diagonal RMSD_dist.xvg or RMSD_bin.dat\n\t%s%s%s_dist.xvg or \n\t%s%s%s_bin.dat\nnot found' % \
                        (RMSD_dist_Dir, TrajNameList[Kai],RMSD_SaveAdder, 
                         MatrixDir, TrajNameList[Kai], RMSD_SaveAdder))
         #-- OFF-Diags   
            for Kai2 in range(Kai+1,len(TrajNameList)):
                if Kai != Kai2:
                    if os.path.exists('%s%s_%s%s_dist.xvg' % \
                                      (RMSD_dist_Dir, TrajNameList[Kai],TrajNameList[Kai2], RMSD_SaveAdder)):
                      # g_rms -dist generates always 101 bins
                        temp = NP.genfromtxt('%s%s_%s%s_dist.xvg' % \
                                             (RMSD_dist_Dir, TrajNameList[Kai],TrajNameList[Kai2], RMSD_SaveAdder), 
                                             skip_header=50) # skip_header, if XVG header is used
                        MaxRMSD = max(MaxRMSD, temp[-1,0])
                        del temp
                    elif os.path.exists('%s%s_%s%s_bin.dat' % \
                                        (MatrixDir, TrajNameList[Kai],TrajNameList[Kai2], RMSD_SaveAdder)):
                        if BinFile_precision is None:
                         #----- TRY TO LOAD NP.NDarray like input, like generated by AmberTools14
                            try:
                                temp = NP.genfromtxt('%s%s_%s%s_bin.dat' % \
                                       (MatrixDir, TrajNameList[Kai], TrajNameList[Kai2], RMSD_SaveAdder))
                            except ValueError:
                                raise ValueError('The RMSD matrix is not stored in a txt-file.\n\t'+\
                                                 ('Check your RMSD matrix format and >>BinFile_precision=%s<< input\n' % BinFile_precision)+\
                                                 '\'None\' corresponds to txt-file Matrix shape format\n'+\
                                                 '\'float32/64\' corresponds to GROMACS binary format')
                            if len(temp.shape) != 2:
                                raise ValueError('The RMSD matrix is not stored in Matrix form, (Rows)x(Columns)\n\tCheck your matrices stored in >>%s<<' % MatrixDir)
                          #--- if FIRST COL = Indices: strip them (AmberTools14)
                            if NP.sum(NP.absolute(temp[:,0]-range(1,len(temp[:,0])+1))) < 10e-6:
                              #--- AmberTool14 uses Angstrom, thus /10 for NM
                                MaxRMSD = max(MaxRMSD, NP.max(temp[:,1:])/10.)
                            else:
                                MaxRMSD = max(MaxRMSD, NP.max(temp))               
                        else: 
                            temp = NP.fromfile('%s%s_%s%s_bin.dat' % \
                                   (MatrixDir, TrajNameList[Kai], TrajNameList[Kai2], RMSD_SaveAdder), 
                                               dtype=BinFile_precision)
                            MaxRMSD = max(MaxRMSD, NP.max(temp)) 
                        del temp
                    else:
                        raise NameError('Error extracting the maximal RMSD for the given TrajNameList: \n'+\
                                        'Off-Diagonal RMSD_dist.xvg or RMSD_bin.dat\n'+\
                                        '\t%s%s_%s%s_dist.xvg or \n\t%s%s_%s%s_bin.dat\nnot found' % \
                                        (RMSD_dist_Dir, TrajNameList[Kai], TrajNameList[Kai2],RMSD_SaveAdder, 
                                         MatrixDir,  TrajNameList[Kai], TrajNameList[Kai2],RMSD_SaveAdder))
      #------- STORE MaxRMSD to File
        with open('%sMaxMinRMSD_%s.txt' % (SaveDir, SaveName), 'w') as OUTPUT:
            OUTPUT.write('# This file stores the maximal AND minimal RMSD value for the given TrajNameList\n'+\
                         '# TrajNameList = {}\n'.format(TrajNameList)+\
                         '# SaveDir = '+SaveDir+'\n'+\
                         '# RMSD_dist_Dir = '+RMSD_dist_Dir+'\n'+\
                         '# MatrixDir = '+MatrixDir+'\n'+\
                         '# BinFile_precision = {}\n'.format(BinFile_precision))
            OUTPUT.write(str(MaxRMSD)+'\t#maximal RMSD\n')
    return MaxRMSD
############################################################################################################
def determineR_generate_RMSD_distributions(TrajNameList, SaveName='V3', 
                                           MatrixDir = 'RMSD_files/BinFiles/',
                                           SaveDir = 'Amber14Trajs/RMSD_distributions/',
                                           BinFile_precision=NP.float32,
                                           Bins=250, RMSD_SaveAdder=''):
    """
v12.10.16
    generate RMSD distributions using ALL RMSD matrices | ADD MinRMSD to MaxMinRMSD_<SaveName>.txt

INPUT:
    TrajNameList       : {LIST}    list of Trajectory name prefixes WITHOUT ENDING, 
                                    Names of the Block RMSD matrices / distributions refer to these names,
                                    e.g. ['MD1', 'MD2', ...] <-> MD1_bin.dat, MD2_bin.dat, MD1_MD2_bin.dat, ...;
    SaveName           : {STRING}  Name for the RMSD distributions, e.g. 'V3' leads to 'Diag_V3_ALL_dist_Bins250.txt,
                                    'Diag_%s_ALL_dist_Bins%s.txt' % (SaveName, Bins);
    MatrixDir          : {STRING}  Directory, where RMSD matrices are stored, e.g. 'RMSD_matrices/';
    SaveDir            : {STRING}  <default RMSD_distributions/>, Directory, where RMSD distributions are stored;
    BinFile_precision  : {TYPE}    FORMAT, [GROMACS/BINARY] single precision 'float32', double precision 'float64', [AMBER/ELSE] 'None';
    Bins               : {INT}     <default 250>, number of BINS  used for RMSD histrogram between 0-MaxRMSD;
OUTPUT:
    generate (1) all single traj RMSD_dist | (2) all X vs Y traj RMSD_dist | 
             (3) concatenated single traj Diag_RMSD_dist | (4) concatenated X vs Y traj OffDiag_RMSD_dist 
             (5) concatenated ALL traj Full_RMSD_dist
    and store in <SaveDir>
    """
#-------------------------------------
 #---- Generate RMSD_distributions
#-------------------------------------
    if os.path.exists('%sDiag_%s_ALL_dist_Bins%s.txt' % (SaveDir, SaveName, Bins)) and \
       os.path.exists('%sOffDiag_%s_ALL_dist_Bins%s.txt' % (SaveDir, SaveName, Bins)) and \
       os.path.exists('%sFull_%s_ALL_dist_Bins%s.txt' % (SaveDir, SaveName, Bins)):
        
        if os.path.exists('%sMaxMinRMSD_%s.txt' % (SaveDir, SaveName)) and \
           not NP.genfromtxt('%sMaxMinRMSD_%s.txt' % (SaveDir, SaveName)).shape == (2,):
            with open('%sMaxMinRMSD_%s.txt' % (SaveDir, SaveName), 'r') as INPUT:
                for line in INPUT:
                    if line.split()[1] == 'TrajNameList':
                        if '# TrajNameList = {}\n'.format(TrajNameList) == line:
                         ### STORE MinRMSD ###
                            Full_dist = NP.genfromtxt('%sFull_%s_ALL_dist_Bins%s.txt' % (SaveDir, SaveName, Bins), 
                                                      usecols=(0,1,2), skip_footer=1)
                            MinRMSD   = 0
                            for F in Full_dist:
                                if abs(F[2]) > 10e-6:
                                    MinRMSD = F[0]
                                    break
                            with open('%sMaxMinRMSD_%s.txt' % (SaveDir, SaveName), 'a') as OUTPUT:
                                OUTPUT.write(str(MinRMSD)+'\t#minimal RMSD')
                        else:
                            print 'wrong SaveName to add MinRMSD to MaxMinRMSD_%s.txt' % (SaveName)
    else:
        if not os.path.exists('%sMaxMinRMSD_%s.txt' % (SaveDir, SaveName)):
            raise NameError('%sMaxMinRMSD_%s.txt does not exist! MaxRMSD cannot be extracted!' % (SaveDir, SaveName))
        try: # first value = MaxRMSD | snd value = MinRMSD
            MaxRMSD = NP.genfromtxt('%sMaxMinRMSD_%s.txt' % (SaveDir, SaveName))[0]
        except IndexError: # only MaxRMSD exists
            MaxRMSD = NP.float(NP.genfromtxt('%sMaxMinRMSD_%s.txt' % (SaveDir, SaveName)))
        Diag_dist    = NP.zeros( (Bins) )
        OffDiag_dist = NP.zeros( (Bins) )
       #-----
        for Kai in range(len(TrajNameList)):
         #----- GENERATE  DIAGS
            # TRY TO LOAD RMSD_dist.xvg
            if os.path.exists('%s%s%s_bin.dat' % (MatrixDir, TrajNameList[Kai], RMSD_SaveAdder)):
                if BinFile_precision is None:
                 #----- TRY TO LOAD NP.NDarray like input, like generated by AmberTools14
                    try:
                        temp = NP.genfromtxt('%s%s%s_bin.dat' % (MatrixDir, TrajNameList[Kai], RMSD_SaveAdder))
                    except ValueError:
                        raise ValueError('The RMSD matrix is not stored in a txt-file.\n\t'+\
                                         ('Check your RMSD matrix format and >>BinFile_precision=%s<< input\n' % BinFile_precision)+\
                                         '\'None\' corresponds to txt-file Matrix shape format\n'+\
                                         '\'float32/64\' corresponds to GROMACS binary format')
                    if len(temp.shape) != 2:
                        raise ValueError('The RMSD matrix is not stored in Matrix form, (Rows)x(Columns)\n\tCheck your matrices stored in >>%s<<' % MatrixDir)
                  #--- if FIRST COL = Indices: strip them (AmberTools14)
                    if NP.sum(NP.absolute(temp[:,0]-range(1,len(temp[:,0])+1))) < 10e-6:
                      #--- AmberTool14 uses Angstrom, thus /10 for NM
                        temp = temp[:,1:]/10.
                    temp_len = int(len(temp[:,0]))
                else:
                    temp = NP.fromfile('%s%s%s_bin.dat' % (MatrixDir, TrajNameList[Kai], RMSD_SaveAdder), 
                                       dtype=BinFile_precision)
                    temp_len = int(NP.sqrt(len(temp)))
                    if len(temp) != temp_len*temp_len:
                        raise ValueError(('The RMSD matrix generated by GROMACS is not loaded correctly!\n\tCheck your >>BinFile_precision=%s<< input\n' % \
                                BinFile_precision)+\
                              'The BinFile_precision must correspond to the precision which was used to install GROMACS'+\
                              '\n\t\'float32\' corresponds to single precision\n\t\'float64\' corresponds to double precision!\n'+\
                              'This was tested for GROMACS v4.6 and v5.1, for other versions the RMSD matrices might differ and have to be preprocessed by hand\n'+\
                              'and stored in matrix-form in a txt-file using >>BinFile_precision=None<<')
                    temp = NP.reshape(temp, (temp_len,temp_len))
                temp_diag_dist = NP.zeros( ( temp_len*(temp_len-1)/2 ) )
                Index = 0
                for Row in range(0,temp_len-1):
                    for Col in range(Row+1, temp_len):
                        temp_diag_dist[Index] = temp[Row, Col]
                        Index += 1                          
                temp_diag_dist, temp_diag_dist_edges2 = \
                                NP.histogram(temp_diag_dist, bins=Bins, range=(0,round(MaxRMSD*1.01,3)))
                temp_diag_dist_edges1 = [round((temp_diag_dist_edges2[elem]+temp_diag_dist_edges2[elem+1])/2,6) \
                                         for elem in range(Bins)]
              #----- STORE single DIAGS
                with open('%sDiag_%s_%s_dist_Bins%s.txt' % \
                          (SaveDir, SaveName, TrajNameList[Kai], Bins), 'w') as OUTPUT:
                    OUTPUT.write('# This file stores the RMSD distribution for\n'+\
                                 '# \tTrajectory = '+TrajNameList[Kai]+'\n'+\
                                 '# BinEdges | BinMid | RMSD distribution\n')
                    for Index in range(Bins):
                        OUTPUT.write('%.6f  %.6f  %i\n' % (temp_diag_dist_edges2[Index], 
                                                           temp_diag_dist_edges1[Index],
                                                           temp_diag_dist[Index]))
                    OUTPUT.write('%.6f\n' % temp_diag_dist_edges2[-1])
                Diag_dist = NP.add(Diag_dist, temp_diag_dist)    
                del temp, temp_len
            else:
                raise NameError('Error extracting the maximal RMSD for the given TrajNameList: \n'+\
                        'Diagonal RMSD_bin.dat\n\t%s%s%s_bin.dat\nnot found' % \
                        (MatrixDir, TrajNameList[Kai], RMSD_SaveAdder))
         #----- GENERATE  OFF-Diags   
            for Kai2 in range(Kai+1,len(TrajNameList)):
                if Kai != Kai2:
                    if os.path.exists('%s%s_%s%s_bin.dat' % \
                                      (MatrixDir, TrajNameList[Kai],TrajNameList[Kai2], RMSD_SaveAdder)):
                        if BinFile_precision is None:
                         #----- TRY TO LOAD NP.NDarray like input, like generated by AmberTools14
                            try:
                                temp = NP.genfromtxt('%s%s_%s%s_bin.dat' % \
                                                   (MatrixDir, TrajNameList[Kai], TrajNameList[Kai2], RMSD_SaveAdder))
                            except ValueError:
                                raise ValueError('The RMSD matrix is not stored in a txt-file.\n\t'+\
                                                 ('Check your RMSD matrix format and >>BinFile_precision=%s<< input\n' % BinFile_precision)+\
                                                 '\'None\' corresponds to txt-file Matrix shape format\n'+\
                                                 '\'float32/64\' corresponds to GROMACS binary format')
                            if len(temp.shape) != 2:
                                raise ValueError('The RMSD matrix is not stored in Matrix form, (Rows)x(Columns)\n\tCheck your matrices stored in >>%s<<' % MatrixDir)
                          #--- if FIRST COL = Indices: strip them (AmberTools14)
                            if NP.sum(NP.absolute(temp[:,0]-range(1,len(temp[:,0])+1))) < 10e-6:
                              #--- AmberTool14 uses Angstrom, thus /10 for NM
                                temp = temp[:,1:]/10.
                            temp = NP.reshape(temp, (len(temp[:,0])*len(temp[:,0]),))
                        else:
                            temp = NP.fromfile('%s%s_%s%s_bin.dat' % \
                                               (MatrixDir, TrajNameList[Kai], TrajNameList[Kai2], RMSD_SaveAdder), 
                                               dtype=BinFile_precision)
                      ## for OFF-Diags: one needs to take EVERY value, because no is present twice
                        temp_offdiag_dist, temp_offdiag_dist_edges2 = \
                                    NP.histogram(temp, bins=Bins, range=(0,round(MaxRMSD*1.01,3)))
               #----- STORE single OFF-DIAGS
                        with open('%sOffDiag_%s_%s_%s_dist_Bins%s.txt' % \
                                  (SaveDir, SaveName, TrajNameList[Kai], TrajNameList[Kai2], Bins), 'w') as OUTPUT:
                            OUTPUT.write('# This file stores the RMSD distribution for the off-diagonal\n'+\
                                         '# \tTrajectory = '+TrajNameList[Kai]+'_'+TrajNameList[Kai2]+'\n'+\
                                         '# BinEdges | BinMid | RMSD distribution\n')
                            for Index in range(Bins):
                                OUTPUT.write('%.6f  %.6f  %i\n' % (temp_offdiag_dist_edges2[Index], 
                                                                   temp_diag_dist_edges1[Index],
                                                                   temp_offdiag_dist[Index]))
                            OUTPUT.write('%.6f\n' % temp_offdiag_dist_edges2[-1])
                        OffDiag_dist = NP.add(OffDiag_dist, temp_offdiag_dist)
                        del temp
                    else:
                        raise NameError('Error extracting the maximal RMSD for the given TrajNameList: \n'+\
                                        'Off-Diagonal RMSD_bin.dat\n'+\
                                        '\t%s%s_%s%s_bin.dat\nnot found' % \
                                        (MatrixDir,  TrajNameList[Kai], TrajNameList[Kai2], RMSD_SaveAdder))
     #----- STORE FULL DIAG
        with open('%sDiag_%s_ALL_dist_Bins%s.txt' % (SaveDir, SaveName, Bins), 'w') as OUTPUT:
            OUTPUT.write('# This file stores the RMSD distribution for all Trajectories\n'+\
                         '# \tTrajNameList = {}\n'.format(TrajNameList)+\
                         '# BinEdges | BinMid | RMSD distribution\n')
            for Index in range(Bins):
                OUTPUT.write('%.6f  %.6f  %i\n' % (temp_diag_dist_edges2[Index], 
                                                   temp_diag_dist_edges1[Index],
                                                   Diag_dist[Index]))
            OUTPUT.write('%.6f\n' % temp_diag_dist_edges2[-1])
     #----- STORE FULL OFF-DIAG
        with open('%sOffDiag_%s_ALL_dist_Bins%s.txt' % (SaveDir, SaveName, Bins), 'w') as OUTPUT:
            OUTPUT.write('# This file stores the RMSD distribution for all Trajectories\n'+\
                         '# \tTrajNameList = {}\n'.format(TrajNameList)+\
                         '# BinEdges | BinMid | RMSD distribution\n')
            for Index in range(Bins):
                OUTPUT.write('%.6f  %.6f  %i\n' % (temp_diag_dist_edges2[Index], 
                                                   temp_diag_dist_edges1[Index],
                                                   OffDiag_dist[Index]))
            OUTPUT.write('%.6f\n' % temp_diag_dist_edges2[-1])
     #----- STORE FULL RMSD Distance
        with open('%sFull_%s_ALL_dist_Bins%s.txt' % (SaveDir, SaveName, Bins), 'w') as OUTPUT:
            OUTPUT.write('# This file stores the RMSD distribution for all Trajectories\n'+\
                         '# \tTrajNameList = {}\n'.format(TrajNameList)+\
                         '# BinEdges | BinMid | RMSD distribution\n')
            Full_dist = NP.add(Diag_dist, OffDiag_dist)
            for Index in range(Bins):
                OUTPUT.write('%.6f  %.6f  %i\n' % (temp_diag_dist_edges2[Index], 
                                                   temp_diag_dist_edges1[Index],
                                                   Full_dist[Index]))
            OUTPUT.write('%.6f\n' % temp_diag_dist_edges2[-1])
     #----- IF MinRMSD not already stored: STORE MinRMSD
        MinRMSD = 0
        for F in range(len(Full_dist)):
            if abs(Full_dist[F]) > 10e-6:
                MinRMSD = temp_diag_dist_edges2[F]
                break
        if os.path.exists('%s%s' % (SaveDir, SaveName)) and \
           not NP.genfromtxt('%s%s' % (SaveDir, SaveName)).shape == (2,):
            with open('%s%s' % (SaveDir, SaveName), 'r') as INPUT:
                for line in INPUT:
                    if line.split()[1] == 'TrajNameList':
                        if '# TrajNameList = {}\n'.format(TrajNameList) == line:
                            with open('%s%s' % (SaveDir, SaveName), 'a') as OUTPUT:
                                OUTPUT.write(str(MinRMSD)+'\t#minimal RMSD')
                        else:
                            print 'wrong SaveName to add MinRMSD: {}'.format(SaveName)
############################################################################################################
def Plot_determineR_using_RMSD_distributions(TrajNameList, SaveName='V3', SaveNamePdf=None,  
                                             SaveDir = 'Amber14Trajs/RMSD_distributions/', Bins=250, Percent=1,
                                             DiagTitle=[''], OffDiagTitle=[''], Legend=[], Indices1=None, Indices2=None, XLIM=None):
    """
v29.11.16
    Plotting RMSD distributions into different subplots

INPUT:
    TrajNameList       : {LIST}    list of Trajectory name prefixes WITHOUT ENDING, 
                                    Names of the Block RMSD matrices / distributions refer to these names,
                                    e.g. ['MD1', 'MD2', ...] <-> MD1_bin.dat, MD2_bin.dat, MD1_MD2_bin.dat, ...;
    SaveName           : {STRING}  Name for the RMSD distributions, e.g. 'V3' leads to 'Diag_V3_ALL_dist_Bins250.txt,
                                    'Diag_%s_ALL_dist_Bins%s.txt' % (SaveName, Bins);
    SaveNamePdf        : {STRING}  Savename for the PDF, e.g. 'MoleculeName+Specification.pdf';
    SaveDir            : {STRING}  <default RMSD_distributions/>, Directory, where RMSD distributions are stored;
    Bins               : {INT}     <default 250>, number of BINS  used for RMSD histrogram between 0-MaxRMSD;
    Percent            : {FLOAT}   <default 1> plotting interval, where the Percent-amount of all RMSD distributions are 
                                        located in, for Percent = 1, it plots the minimal and maximal values, e.g. 0.99;
    DiagTitle          : {LIST}    <default ['']>, title for ROWS of single traj RMSD_dist, corresponds to splitting 
                                    some trajectories into different subplots by Indices1, e.g. ['cMD','aMD'];
    OffDiagTitle       : {LIST}    <default ['']>, title for ROWS of XvsY, corresponds to splitting 
                                    some trajectories into different subplots by Indices2, e.g. ['cMDvscMD','aMDvsaMD'];
    Legend             : {LIST}    <default []>, LEGEND of the groups of single traj RMSD_dist corresponding to Indices1, 
                                    e.g. ['start C', 'start D'];
    Indices1           : {LIST,TUPLE} <default None> Rows of single RMSD_dist plotted into subfigures, 
                                    e.g. [(0,10,20), (20,30,40)], need to match dimensions of Indices2 & DiagTitle
                                    plotting 2 groups, trajs (0-10|10-20) in 1.row and trajs (20-30|30-40) in 2.row;
    Indices2           : {LIST,TUPLE} <default None> Rows of X vs. Y trajs RMSD_dist plotted into subfigures, 
                                    e.g. [(0,20), (20,40)], need to match dimensions of Indices1 & OffDiagTitle 
                                    plotting X vs. Y trajs (0-20) in 1.row and trajs (20-40) trajs in 2.row;
    XLIM               : {FLOAT-LIST} <default None>, defines the x-limits;
OUTPUT:
    plots/stores the 
        >> ALL SINGLE TRAJS | ALL X vs Y TRAJS | Concatenated Cases <<
    of the RMSD distributions for the submitted Trajectory Names
    """
    import matplotlib.pyplot as plt
    logX=False; 
  ###------- DEFAULT VALUES for Indices: simply use all Trajectories
    if Indices1 is None:
        Indices1 = [(0,len(TrajNameList))]
        Indices2 = [(0,len(TrajNameList))]
        Test2    = False
    elif Indices2 is None:
        Indices2 = [(0,len(TrajNameList))]
        Test2    = True
    else:
        Test2    = False
  ###-------
    fig = plt.figure(figsize=(15,11/3.*len(Indices1)))
  ###------- EACH DIAGONALS 
    for Kug in range(len(Indices1)):
        Min = 100.0
        Max = 0.0
        AX = plt.subplot2grid( (len(Indices1),2 if Test2 else 3), (Kug,0) ); 
        if len(Indices1) == 1:
            plt.subplots_adjust(wspace=0.1, hspace=0.3, left=0.03, right=0.98, bottom=0.18, top=0.91)
        else:
            plt.subplots_adjust(wspace=0.15, hspace=0.30, left=0.03, right=0.98, bottom=0.06, top=0.96)
        plt.title('single trajectories %s' % (DiagTitle[Kug%len(DiagTitle)]), fontsize=25)
        plt.plot([0],[0], 'r-', lw=2); plt.plot([0],[0], 'k-', lw=2); 
        if Legend != []: plt.legend(Legend, fontsize=19, framealpha=0.4);
        for Name in TrajNameList[Indices1[Kug][0]:Indices1[Kug][1]]:
            temp = NP.genfromtxt('%sDiag_%s_%s_dist_Bins%s.txt' % (SaveDir, SaveName, Name, Bins),
                                 usecols=(1,2), skip_footer=1)
            temp2= NP.genfromtxt('%sDiag_%s_%s_dist_Bins%s.txt' % (SaveDir, SaveName, Name, Bins),
                                 usecols=(0))
            if len(Indices1) == 1:
                plt.plot(temp[:,0], temp[:,1]/NP.sum(temp[:,1]), lw=2)
            else:
                plt.plot(temp[:,0], temp[:,1]/NP.sum(temp[:,1]), color='r', lw=2)
         #---------- Extract MIN & MAX values
            for F in range(len(temp[:,0])):
                if abs(temp[F,1]) > 10e-6:
                    Min = min(Min,temp2[F])
                    break
            for F in range(len(temp[:,0])):
                if abs(temp[-(1+F),1]) > 10e-6:
                    Max = max(Max,temp2[-(1+F)])
                    break
         #-----------------------------------
        for Name in TrajNameList[Indices1[Kug][1]:Indices1[Kug][-1]]:
            temp = NP.genfromtxt('%sDiag_%s_%s_dist_Bins%s.txt' % (SaveDir, SaveName, Name, Bins),
                                 usecols=(1,2), skip_footer=1)
            temp2= NP.genfromtxt('%sDiag_%s_%s_dist_Bins%s.txt' % (SaveDir, SaveName, Name, Bins),
                                 usecols=(0))
            plt.plot(temp[:,0], temp[:,1]/NP.sum(temp[:,1]), color='k', lw=2)
         #---------- Extract MIN & MAX values
            for F in range(len(temp[:,0])):
                if abs(temp[F,1]) > 10e-6:
                    Min = min(Min,temp2[F])
                    break
            for F in range(len(temp[:,0])):
                if abs(temp[-(1+F),1]) > 10e-6:
                    Max = max(Max,temp2[-(1+F)])
                    break
         #-----------------------------------
      #------- Plot Min/Max lines
        if XLIM is not None:
            POS = AX.get_position()
            plt.figtext((POS.x1-POS.x0)/AX.get_xlim()[1]*(Min*1.15)+POS.x0, (POS.y1-POS.y0)*0.85+POS.y0,
                    '%.4fnm' % Min, rotation=90, color='k', fontsize=20)
            plt.figtext((POS.x1-POS.x0)/AX.get_xlim()[1]*(Max*.94)+POS.x0, (POS.y1-POS.y0)*0.49+POS.y0,
                    '%.4fnm' % Max, rotation=90, color='k', fontsize=20)
            plt.axvline(Min, ls=':', color='grey', lw=2); plt.axvline(Max, ls=':', color='grey', lw=2);
      #--------------------------
        if Kug == len(Indices1)-1: plt.xlabel('RMSD [nm]', fontsize=22);
        plt.xticks(fontsize=20); plt.yticks([]); 
        plt.ylabel('frequency [a.u.]', fontsize=22); #plt.xlabel('threshold r', fontsize=15)
      #--------------------------  
        if logX: plt.xscale('log'); 
        if XLIM is not None: plt.xlim(XLIM);
        plt.locator_params(axis = 'x', nbins = 5)
  ###------- OFF-DIAGONALS [Same] vs [Same]
    if not Test2:
        for Kug in range(len(Indices1)):
            Min = 100.0
            Max = 0.0
            AX = plt.subplot2grid( (len(Indices1),3), (Kug,1) )
            plt.title('trajX vs trajY %s' % (OffDiagTitle[Kug%len(OffDiagTitle)]), fontsize=25)
            for Kai in range(Indices2[Kug][0],Indices2[Kug][1]-1):
                for Kai2 in range(Kai+1,Indices2[Kug][1]):
                    temp = NP.genfromtxt('%sOffDiag_%s_%s_%s_dist_Bins%s.txt' % \
                                             (SaveDir, SaveName, TrajNameList[Kai], TrajNameList[Kai2], Bins),
                                             usecols=(1,2), skip_footer=1)
                    temp2 = NP.genfromtxt('%sOffDiag_%s_%s_%s_dist_Bins%s.txt' % \
                                             (SaveDir, SaveName, TrajNameList[Kai], TrajNameList[Kai2], Bins),
                                             usecols=(0))
                 #---------- Extract MIN & MAX values
                    for F in range(len(temp[:,0])):
                        if abs(temp[F,1]) > 10e-6:
                            Min = min(Min,temp2[F])
                            break
                    for F in range(len(temp[:,0])):
                        if abs(temp[-(1+F),1]) > 10e-6:
                            Max = max(Max,temp2[-(1+F)])
                            break
                 #-----------------------------------  
                    plt.plot(temp[:,0], temp[:,1]/NP.sum(temp[:,1]))
          #------- Plot Min/Max lines
            if XLIM is not None:
                POS = AX.get_position()
                plt.figtext((POS.x1-POS.x0)/AX.get_xlim()[1]*(Min*1.15)+POS.x0, (POS.y1-POS.y0)*0.85+POS.y0,
                        '%.4fnm' % Min, rotation=90, color='k', fontsize=20)
                plt.figtext((POS.x1-POS.x0)/AX.get_xlim()[1]*(Max*.94)+POS.x0, (POS.y1-POS.y0)*0.49+POS.y0,
                        '%.4fnm' % Max, rotation=90, color='k', fontsize=20)
                plt.axvline(Min, ls=':', color='grey', lw=2); plt.axvline(Max, ls=':', color='grey', lw=2);
          #--------------------------
            if Kug == len(Indices1)-1: plt.xlabel('RMSD [nm]', fontsize=22)
            plt.xticks(fontsize=20); plt.yticks([]); 
            #if Kug == 0: plt.ylabel('normalized frequency', fontsize=15); 
          #--------------------------  
            if logX: plt.xscale('log'); 
            if XLIM is not None: plt.xlim(XLIM);
            plt.locator_params(axis = 'x', nbins = 5)
  ###------- ALL OFF-Diagonals
    if len(Indices1) != 1:
        AX = plt.subplot2grid( (len(Indices1),2 if Test2 else 3), (0,1 if Test2 else 2) )
        plt.title('trajX vs trajY', fontsize=25)
        for Kai in range(0,Indices2[-1][-1]-1):
            for Kai2 in range(Kai+1,Indices2[-1][-1]):
                if Kai != Kai2:
                    temp = NP.genfromtxt('%sOffDiag_%s_%s_%s_dist_Bins%s.txt' % \
                                         (SaveDir, SaveName, TrajNameList[Kai], TrajNameList[Kai2], Bins),
                                         usecols=(1,2), skip_footer=1)
                    temp2 = NP.genfromtxt('%sOffDiag_%s_%s_%s_dist_Bins%s.txt' % \
                                         (SaveDir, SaveName, TrajNameList[Kai], TrajNameList[Kai2], Bins),
                                         usecols=(0))
                    plt.plot(temp[:,0], temp[:,1]/NP.sum(temp[:,1]))
                 #---------- Extract MIN & MAX values
                    for F in range(len(temp[:,0])):
                        if abs(temp[F,1]) > 10e-6:
                            Min = min(Min,temp2[F])
                            break
                    for F in range(len(temp[:,0])):
                        if abs(temp[-(1+F),1]) > 10e-6:
                            Max = max(Max,temp2[-(1+F)])
                            break
                 #-----------------------------------  
      #------- Plot Min/Max lines
        if XLIM is not None:
            POS = AX.get_position()
            plt.figtext((POS.x1-POS.x0)/AX.get_xlim()[1]*(Min*1.15)+POS.x0, (POS.y1-POS.y0)*0.85+POS.y0,
                    '%.4fnm' % Min, rotation=90, color='k', fontsize=20)
            plt.figtext((POS.x1-POS.x0)/AX.get_xlim()[1]*(Max*.94)+POS.x0, (POS.y1-POS.y0)*0.49+POS.y0,
                    '%.4fnm' % Max, rotation=90, color='k', fontsize=20)
            plt.axvline(Min, ls=':', color='grey', lw=2); plt.axvline(Max, ls=':', color='grey', lw=2);
      #--------------------------
        plt.xticks(fontsize=20); plt.yticks([]); 
      #--------------------------  
        if logX: plt.xscale('log'); 
        if XLIM is not None: plt.xlim(XLIM);
        plt.locator_params(axis = 'x', nbins = 5)
  ###------- ALL DIAGS in one, ALL OFF-DIAGS in one, EVERYTHING in one  
    Min = 100.0; Max = 0.0
    Diag_dist    = NP.genfromtxt('%sDiag_%s_ALL_dist_Bins%s.txt' % (SaveDir, SaveName, Bins), 
                                 usecols=(1,2), skip_footer=1)
    OffDiag_dist = NP.genfromtxt('%sOffDiag_%s_ALL_dist_Bins%s.txt' % (SaveDir, SaveName, Bins), 
                                 usecols=(1,2), skip_footer=1)
    Full_dist    = NP.genfromtxt('%sFull_%s_ALL_dist_Bins%s.txt' % (SaveDir,SaveName,Bins), 
                                 usecols=(1,2), skip_footer=1)
    temp2        = NP.genfromtxt('%sFull_%s_ALL_dist_Bins%s.txt' % (SaveDir,SaveName,Bins), 
                                 usecols=(0))
    AX = plt.subplot2grid( (len(Indices1),2 if Test2 else 3), 
                           ((0 if len(Indices1)==1 else 1), 1 if Test2 else 2), 
                          rowspan=(1 if len(Indices1)==1 else len(Indices1)-1))
    plt.plot(Diag_dist[:,0], Diag_dist[:,1]/NP.sum(Diag_dist[:,1]), 'bx-')
    plt.plot(OffDiag_dist[:,0], OffDiag_dist[:,1]/NP.sum(OffDiag_dist[:,1]), 'rx-')
    plt.plot(Full_dist[:,0], Full_dist[:,1]/NP.sum(Full_dist[:,1]), 'k+-') 
      #--------------------------  
    if logX: plt.xscale('log'); 
    if XLIM is not None: plt.xlim(XLIM);
   #---------- Extract MIN & MAX values
    for F in range(len(Full_dist[:,0])):
        if abs(Full_dist[F,1]) > 10e-6:
            Min = min(Min,temp2[F])
            break
    for F in range(len(Full_dist[:,0])):
        if abs(Full_dist[-(1+F),1]) > 10e-6:
            Max = max(Max,temp2[-(1+F)])
            break
                 #-----------------------------------  
  #------- Plot Min/Max lines
    if XLIM is not None:
        POS = AX.get_position()
        plt.figtext((POS.x1-POS.x0)/AX.get_xlim()[1]*(Min*1.15)+POS.x0, (POS.y1-POS.y0)*0.85+POS.y0,
                    '%.4fnm' % Min, rotation=90, color='k', fontsize=20)
        plt.figtext((POS.x1-POS.x0)/AX.get_xlim()[1]*(Max*.94)+POS.x0, (POS.y1-POS.y0)*0.49+POS.y0,
                '%.4fnm' % Max, rotation=90, color='k', fontsize=20)
        plt.axvline(Min, ls=':', color='grey', lw=2); plt.axvline(Max, ls=':', color='grey', lw=2);
  #------- Plot MarkerL/MarkerR lines
    if Percent < 1 and Percent > 0 and XLIM is not None:
        MarkerL, MarkerR = ReturnPercentRMSD(Percent, Full_dist, temp2)
        plt.figtext((POS.x1-POS.x0)/AX.get_xlim()[1]*(MarkerL*1.12)+POS.x0, (POS.y1-POS.y0)*0.85+POS.y0,
                    '%.4fnm' % MarkerL, rotation=90, color='r', fontsize=20)
        plt.axvline(MarkerL, ls=':', color='r', lw=2)
        plt.figtext((POS.x1-POS.x0)/AX.get_xlim()[1]*(MarkerR*0.92)+POS.x0, (POS.y1-POS.y0)*0.49+POS.y0,
                    '%.4fnm' % MarkerR, rotation=90, color='r', fontsize=20)
        plt.axvline(MarkerR, ls=':', color='r', lw=2)
  #--------------------------
    plt.xticks(fontsize=20); plt.yticks([]); plt.xlabel('RMSD [nm]', fontsize=22)
    plt.legend(['all single', 'all trajX vs Y', 'full'], fontsize=19, numpoints=1, framealpha=0.5)
    plt.locator_params(axis = 'x', nbins = 5)
 #----------------------
    if SaveDir is not None and SaveNamePdf is not None:
        Generate_Directories(SaveDir+SaveNamePdf)
        plt.savefig('%s%s' % (SaveDir, SaveNamePdf))
############################################################################################################
def determineR_using_RMSD_distributions(TrajNameList, SaveName='V3', SaveNamePdf=None,
                                        SaveDir = 'Amber14Trajs/RMSD_distributions/',
                                        RMSD_dist_Dir = 'Amber14Trajs/RMSD_files/',
                                        MatrixDir = 'Amber14Trajs/RMSD_files/BinFiles/',
                                        BinFile_precision=NP.float32,
                                        Bins=250, Percent=1, RMSD_SaveAdder=''):
    """
v12.10.16
    this function plots the RMSD distributions within a certain interval 
    containing Percent-amount of the Full RMSD distribution
    
    1. extract from all RMSD distributions the maximal value
    2. generate an adequate BINS binning for the range of RMSD values
    3. generate multiple analysis cases:
        a) only diagonals     = each single trajectories
        b) only off-diagonals = concatenate histrograms of off-diagonals due to size n*(n-1)/2
        c) all trajectories at once for one concatenated RMSD distribution

INPUT:
    TrajNameList       : {LIST}    list of Trajectory name prefixes WITHOUT ENDING, 
                                    Names of the Block RMSD matrices / distributions refer to these names,
                                    e.g. ['MD1', 'MD2', ...] <-> MD1_bin.dat, MD2_bin.dat, MD1_MD2_bin.dat, ...;
    SaveName           : {STRING}  Name for the RMSD distributions, e.g. 'V3' leads to 'Diag_V3_ALL_dist_Bins250.txt,
                                    'Diag_%s_ALL_dist_Bins%s.txt' % (SaveName, Bins);
    SaveNamePdf        : {STRING}  Savename for the PDF, e.g. 'MoleculeName+Specification.pdf';
    SaveDir            : {STRING}  <default RMSD_distributions/>, Directory, where RMSD distributions, the MaxMinValue and
                                    the PDF will be stored;
    RMSD_dist_Dir      : {STRING}  <default ''> Directory, where (possible) RMSD distributions are located (from GROMACS), e.g. 'RMSD_files/', 
                                    if it does not exist, RMSD matrices are used instead;
    MatrixDir          : {STRING}  Directory, where RMSD matrices are stored, e.g. 'RMSD_matrices/';
    BinFile_precision  : {TYPE}    FORMAT, [GROMACS/BINARY] single precision 'float32', double precision 'float64', [AMBER/ELSE] 'None';
    Bins               : {INT}     <default 250>, number of BINS  used for RMSD histrogram between 0-MaxRMSD;
    Percent            : {FLOAT}   <default 1> plotting interval, where the Percent-amount of all RMSD distributions are 
                                        located in, for Percent = 1, it plots the minimal and maximal values, e.g. 0.99; 
OUTPUT:
    plots/stores the 
        >> ALL SINGLE TRAJS | ALL X vs Y TRAJS | Concatenated Cases <<
    of the RMSD distributions for the submitted Trajectory Names
    """
    Generate_Directories(SaveDir)
#-------------------------------------
 #---- EXTRACT MAXIMAL RMSD VALUE 
#-------------------------------------    
    MaxRMSD = determineR_extract_MaxRMSD(TrajNameList, SaveName, RMSD_dist_Dir, MatrixDir, SaveDir, BinFile_precision,
                                         RMSD_SaveAdder)
#-------------------------------------
 #---- Generate RMSD_distributions
#-------------------------------------
    determineR_generate_RMSD_distributions(TrajNameList, SaveName, MatrixDir, SaveDir, BinFile_precision, Bins,
                                           RMSD_SaveAdder)
#-------------------------------------
#---- GENERATE PLOTS 
#-- using different INDICEs to plot different Groups of Trajectories
#-------------------------------------
    try:
        if 'DISPLAY' in os.environ:
            Plot_determineR_using_RMSD_distributions(TrajNameList, SaveName=SaveName, SaveNamePdf=SaveNamePdf, 
                                                     SaveDir=SaveDir, Bins=Bins, Percent=Percent)
        else:
            print 'The RMSD distributions cannot be plotted due to missing DISPLAY functionality in your python, but all distributions were generated!'
    except TypeError:
        print 'All distributions are generated, but the figures could not be generated. Try to update Python/matplotlib'
  #####################  
def ReturnPercentRMSD(Proz, Array1, Array2):
    """
v13.05.16
    supporting function to extract the limits, where the Proz-amount of the distributions are within these vertical lines
INPUT:
    Proz    :  {FLOAT}       percent, how many percent of the full area under the curve lies within the two limits [Min,Max];
    Array1  :  {NP.NDARRAY}  containing the RMSD distribution 
                                e.g. Full_dist = NP.genfromtxt('%sFull_%s_ALL_dist_Bins%s.txt' % (SaveDir,SaveName,Bins), 
                                                               usecols=(1,2), skip_footer=1);
    Array2  :  {NP.NDARRAY}  containing the BINNING EDGES, 
                                e.g. temp2     = NP.genfromtxt('%sFull_%s_ALL_dist_Bins%s.txt' % (SaveDir,SaveName,Bins), 
                                                               usecols=(0));
OUTPUT:
    return limits [Min,Max]
    """
    Min = -1; Max = -1;
    Calle = 0
    for FF in range(len(Array1[:,1])):
        Calle += Array1[FF,1]
        if Calle/NP.sum(Array1[:,1]) > (1-Proz)/2. and Min == -1:
            Min = Array2[FF]
        if Calle/NP.sum(Array1[:,1]) > 1-(1-Proz)/2. and Max == -1:
            Max = Array2[FF]
    return Min, Max
  #####################  

#######################################################
#------------------------------------------------------
#---          CLUSTERING CALCULATION
#------------------------------------------------------
#######################################################
def Generate_Clustering(MatrixDir, SaveDir, TrajNameList, TrajLengthList, Threshold, SaveName, MaxNumberLines, 
                        TimeStep=None, StartFrame=0, EndingFrame=NP.infty, PartList=None, GLOBAL=True,
                        RMSDdir=None, TrajDir=None, TopologyDir=None, TopologyName=None, Ending='.xtc',
                        Select1=None, Select2=None, AmberHome='', GromacsHome='', Program_Suffix='', ReferencePDB=None, 
                        BinFile_precision=NP.float32, RefFrame=None, RMSD_SaveAdder=''):
    """
v12.10.16
Calculates and generates the LOCAL or GLOBAL PROFILE & CENTROIDS for effective clustering for the submitted 
trajectories, 
    - using Generate_reference_for_Clustering() & Return_FullColRMSD()
    - using RMSD matrices
    - possibility, to investigate different (equal) simulation time parts
    - possibility, to submit a reference as a starting point: the structure with the lowest RMSD to this reference
                   is then the first starting centroid (ONLY FOR GROMACS OR AMBER TRAJECTORIES)
                   otherwise, the first frame of the trajectory is used as starting point
     --> it is also possible to use RefFrame, to select a frame which is then always taken as starting centroid
    - MaxNumberLines: DIRECTLY INFLICTS THE MEMORY USAGE:
        1. for GLOBAL clustering, at least one FULL row of all involved RMSD matrices has to be loaded
        2. MaxNumberLines adjust the number of these rows which are loaded at once
        3. if a full RMSD block (shape=2000,2000) can be loaded with MaxNumberLines=2000, the algorithm is much faster
        4. if the FULL RMSD matrix, containing all RMSD block matrices, can be loaded, everything can be clustered at once
    - generates: 
                |                  Threshold = 0.45                            |
                |  eff Clust     |    next Center   |     farthest Center      |
 Frame | TrajNr | PROF | effRMSD | nextC | nextRMSD | farthestC | farthestRMSD |
         - PROF:         cluster number/centroid for the specific frame
         - effRMSD:      over the course of the "average merging" different centroids, this represents the RMSD value of the
                         new centroid to the merged formerly defined centroids
         - nextC:        the closest centroid to the current frame
         - nextRMSD:     RMSD value between the current frame and the closest centroid
         - farthestC:    the farthest centroid to the current frame
         - farthestRMSD: RMSD value between the current frame and the farthest centroid
    - trajectories, which do not match <StartFrame-EndingFrame> are discarded, but TrajNr will be still consecutive,
      check the description stored in the HEADER of the stored Clustering file

INPUT:
    MatrixDir         : {STRING}    Directory, where RMSD matrices are stored, e.g. 'RMSD_matrices/';
    SaveDir           : {STRING}    Directory, where the clustering will be stored, e.g. 'Clustering/';
    TrajNameList      : {LIST}      list of Trajectory name prefixes WITHOUT ENDING, 
                                    Names of the Block RMSD matrices / distributions refer to these names,
                                    e.g. ['MD1', 'MD2', ...] <-> MD1_bin.dat, MD2_bin.dat, MD1_MD2_bin.dat, ...;
    TrajLengthList    : {INT-LIST}  number of frames for corresponding TrajName, have to match RMSD matrix shape & 
                                    length of TrajNameList, e.g. [2000, 2000, 1000, ...] 
                                    if PartList != None, TrajNameList must store the NrOfFrames for each SPLIT;
    Threshold         : {FLOAT}     Clustering-Threshold [nm], e.g. 0.2;
    SaveName          : {STRING}    PREFIX name for the clustering file, e.g. 'Mol2' leads to 'Mol2_R0.1_0-100_GLOBAL.txt', 
                            '%s_R%s_%s-%s_%s.txt' % (SaveName, Threshold, StartFrame, EndingFrame, 'GLOBAL' or 'LOCAL');
    MaxNumberLines    : {INT}       Maximal number of lines which are loaded from the FullRMSD matrix, 
                                    it inflicts directly the memory usage, if MaxNumberLines > sum(TrajLengthList), all
                                    RMSD matrices are loaded at once, recommended using length of one RMSD block,
                                    e.g. if one block RMSDmatrix.shape == (2000,2000), try to use MaxNumberLines = 2000;
    TimeStep          : {FLOAT/INT} <default None> ONLY NECESSARY IF ReferencePDB is not None! 
                                                selects the frames of the trajectory,
                                                GROMACS, -dt 't MOD dt = first [ns]', 
                                                AMBER    skip dt-th frame 'first last skip',
                                                if TimeStep = None, use ALL frames in the trajectory;
    StartFrame        : {INT}       <default 0>    starting frame of Trajectories/RMSD matrices,
                                    to select different simulation times/lengths together with 'EndingFrame';
    EndingFrame       : {INT}       <default NP.infty> ending frame of Trajectories/RMSD matrices,
                                    to select different simulation times/lengths together with 'StartingFrame';
    PartList          : {INT-LIST}  <default None> defines into how many parts the single trajectories are split, due to
                                    memory reasons,
                                        1. len(PartList) == len(TrajNameList) !!  ||   
                                        2. PartList = [1,2,3], (MD1.xtc -> MD1.xtc),
                                                               (MD2.xtc -> MD2_part1.xtc, MD2_part2.xtc),
                                                               (MD3.xtc -> MD3_part1.xtc, MD3_part2.xtc, MD3_part3.xtc), ||
                                        3. default PartList = None -> PartList = [1]*len(TrajNameList);    
    GLOBAL            : {BOOL}      <default True> if True,  a GLOBAL clustering is applied for all concatenated trajectories
                                                   if False, every trajectory is clustered separately;
    RMSDdir           : {STRING}    <default None> ONLY NECESSARY IF ReferencePDB is not None! 
                                                   directory, where to store possible RMSD curves to a submitted reference,
                                                   if is None, RMSDdir = SaveDir;
    TrajDir           : {STRING}    <default None> ONLY NECESSARY IF ReferencePDB is not None! 
                                                   Directory, where the trajectories are located if a reference is submitted,
                                                   then, RMSD curves are calculated using these, e.g. 'TrajDir/';
    TopologyDir       : {STRING}    <default None> ONLY NECESSARY IF ReferencePDB is not None! 
                                                   Directory, where the topology file is located, for GROMACS, this is
                                                   equal to the Reference(PDB), for AMBER, this is a .top file, e.g. 'TrajDir/';
    TopologyName      : {STRING}    <default None> ONLY NECESSARY IF ReferencePDB is not None! 
                                                   topology name, for GROMACS this has to contain the reference(PDB)
                                                   for AMBER, this is the input topology .top;
    Ending            : {STRING}    <default .xtc> ONLY NECESSARY IF ReferencePDB is not None! 
                                                   ending of the trajectories, only necessary, if a reference is submitted,
                                                   normally '.xtc' or '.trr', or '.pdb' or '.nc', or '.netcdf';
    Select1           : {STRING}    <default None> ONLY NECESSARY IF ReferencePDB is not None! 
                                                   FIT  selection Gromacs, e.g. 'Backbone',
                                                   CALC selection Amber  , e.g. '@N,CA,C', see Amber syntax;
    Select2           : {STRING}    <default None> ONLY NECESSARY IF ReferencePDB is not None! 
                                                   CALC selection Gromacs, e.g. 'Backbone';
    AmberHome         : {STRING}    <default ''>   ONLY NECESSARY IF ReferencePDB is not None! 
                                                   if cpptraj is not in environmental variables, define directory to 'cpptraj',
                                    e.g. /home/user/Software/amber14/bin/;
    GromacsHome       : {STRING}    <default ''>   ONLY NECESSARY IF ReferencePDB is not None! 
                                                   if g_rms | gmx_suffix rms is not in environmental variables, define directory to them,
                                    e.g. /home/user/Software/gromacs/bin/;
    Program_Suffix    : {STRING}    <default ''>   ONLY NECESSARY IF ReferencePDB is not None! 
                                                   gmx_suffix for Gromacs installation, e.g. '_467' for 'g_rms_467';
    ReferencePDB      : {STRING}    <default None> ONLY NECESSARY IF ReferencePDB is not None! 
                                                   possible reference, to which the RMSD between all trajectories are
                                            calculated, then the closest structure is the starting point for the clustering
                                                        e.g. 'Crystalstructure.pdb';
    BinFile_precision : {TYPE}      FORMAT, [GROMACS/BINARY] single precision 'float32', double precision 'float64', [AMBER/ELSE] 'None';
    RefFrame          : {INT}       <default None> possibility to submit a frame, which is the first centroid for the clustering, 0 means the first frame of the 
                                        trajectories, None means, it tries to calculate RMSD curves using GROMACS or AMBER if ReferencePDB is not None;
OUTPUT:
              
        need: 
            - RMSD matrices
            - maximal memory/size
            - length to generate blocks of full columns and maximal rows which fit to the memory/maximum
            - PossibleClusters
    """
    #----- CHECK if ReferencePDB does exist
    if ReferencePDB is not None and not os.path.exists(ReferencePDB) and not os.path.exists('%s%s' % (TrajDir,ReferencePDB)) \
                and not os.path.exists('%s%s' % (TopologyDir, ReferencePDB)):
        raise ValueError('ReferencePDB not found. Neither\n\t%s\nnor\n\t%s%s\nnor\n\t%s%s\ndo exist!' % \
                (ReferencePDB, TrajDir, ReferencePDB, TopologyDir, ReferencePDB))
    #----- DETECT Gromacs or Amber
    ## GROMACS ##
    if ReferencePDB != None and (Ending == '.xtc' or Ending == '.trr' or Ending == '.pdb'): 
        TopologyName = ReferencePDB
    ## AMBER   ##
    else: 
        pass
    #----- CHECK if clustering already exists
    if os.path.exists('%s%s_R%s_%s-%s_%s.txt' % \
                      (SaveDir,SaveName,Threshold,StartFrame,EndingFrame,'GLOBAL' if GLOBAL else 'LOCAL')) and \
       os.path.exists('%s%s_R%s_%s-%s_Centers_%s.txt' % \
                      (SaveDir,SaveName,Threshold,StartFrame,EndingFrame,'GLOBAL' if GLOBAL else 'LOCAL')):
        print '\t%s%s_R%s_%s-%s_%s.txt\nand\t%s%s_R%s_%s-%s_Centers_%s.txt\nalready exist!' % \
                      (SaveDir,SaveName,Threshold,StartFrame,EndingFrame,'GLOBAL' if GLOBAL else 'LOCAL',
                       SaveDir,SaveName,Threshold,StartFrame,EndingFrame,'GLOBAL' if GLOBAL else 'LOCAL')
        return
    #-----
    t1 = time.time()
    if PartList is None:
        PartList = [1]*len(TrajNameList)    
#####################
### v13.07.16 UPDATE: INIT FullCumTrajLenList & delete trajectories which do not match Startframe-EndingFrame
    FullCumTrajLenList = [0]
    while NP.all(FullCumTrajLenList) == False:
        TrajLenDict = Generate_TrajLenDict(len(TrajNameList), PartList, TrajLengthList, StartFrame, EndingFrame)
        FullCumTrajLenList = [max(0,min(EndingFrame,Return_TrajLen_for_trajX(elem,
                                                                                    TrajLenDict,
                                                                                    PartList, 
                                                                                    range(len(TrajNameList))))-
                                    StartFrame) for elem in range(len(TrajNameList))]
        TrajLengthList = [TrajLenDict['%s%s' % (elem1, '_part'+str(elem2) if PartList[elem1] > 1 else '')][0] \
                            for elem1 in range(len(TrajNameList)) for elem2 in range(1,PartList[elem1]+1) \
                            if FullCumTrajLenList[elem1] != 0]
        TrajNameList   = [TrajNameList[elem] for elem in range(len(TrajNameList)) if FullCumTrajLenList[elem] != 0]
        PartList       = [PartList[elem] for elem in range(len(PartList)) if FullCumTrajLenList[elem] != 0]
    FullCumTrajLenList = [sum(FullCumTrajLenList[:elem2]) for elem2 in range(0,len(TrajNameList)+1)]
    ##---- ERROR DETECT, if all trajectories DO NOT FIT into <StartFrame-EndingFrame>, stop the calculation
    if len(TrajNameList) == 0:
        print 'No trajectory fit into the simulation time [StartFrame, EndingFrame], stopping calculation!'
        return 
    #---- if EndingFrame is NP.infty OR > largest trajectory size, then use the largest trajectory size
    EndingFrame = min(EndingFrame, \
                      max([max([NP.sum([TrajLenDict['%s%s' % \
                                                    (Kai, (('_part%s' % elem) if PartList[Kai] > 1 else ''))][0] \
                                        for elem in range(1,PartList[Kai]+1)])]) \
                           for Kai in range(len(TrajNameList))]))
###
#####################
### INITIALIZATION: ClusterProfile + Centers
#   v20.01.16 UPDATE: adding lowest absolute RMSD & largest absolute RMSD comparing to centers
#   Frame | TrajNr | PROFILE | effRMSD | next Center | next RMSD | farthest Center | farthest RMSD
    ClusterProfile = NP.zeros( ( FullCumTrajLenList[-1], 2+2+2+2) ) 
    ClusterProfile[:,3] = 1000
    ClusterProfile[:,0] = range(FullCumTrajLenList[-1])
  ####
    Centers = []
    if not GLOBAL:
        for Kai in range(len(TrajNameList)):
            Centers.append([])
  ####
    Finished = False
    CentersFinished = False
  ####
    NEXT = 0
    TrajNr = 1
    LastTrajNr = 1
    AddLength = 0
  ## UPDATED v16.11.15: CRITERIA to choose reference:
      #  (a) GLOBAL = TRUE : choose only if NEXT == 0 and ONLY ONCE in the beginning
      #  (b) GLOBAL = FALSE: choose only IF NEW TRAJ is selected: if NEXT == FullCumTrajLenList but ONLY ONCE:
    if GLOBAL:
        Criteria = [0]
    else:
        Criteria = [elem for elem in FullCumTrajLenList]
    
  #########  START: extract FullColRMSD  #########
    if ReferencePDB is not None:
        RefName = ReferencePDB.split('.pdb')[0]
    while not Finished:
      ## UPDATED v16.11.15: define a value which decides, if ReferencePDB != None, INIT is done
        Critters = -1
      ## UPDATED v13.11.15: use REFERENCE STRUCTURE
        if ReferencePDB != None and Criteria.count(NEXT) > 0:
            #--- STORE Ref_RMSD.xvg to RMSDdir, if None: RMSDdir = SaveDir
            if RMSDdir is None:
                RMSDdir = SaveDir
            else:
                Generate_Directories(RMSDdir)
            if TrajDir is None or TopologyDir is None or Select1 is None:
                raise ValueError('One of the following parameters are not set, check your input:\n'+\
                                 '\tTrajDir = %s\n\tTopologyDir = %s\n\tSelect1 = %s\n' % \
                                  (TrajDir, TopologyDir, Select1))
            Critters = Criteria.pop(0)
            ## add directory to ReferencePDB
            if os.path.exists('%s%s' % (TrajDir, ReferencePDB)):
                ReferencePDB = '%s%s' % (TrajDir, ReferencePDB)
            elif os.path.exists('%s%s' % (TopologyDir, ReferencePDB)):
                ReferencePDB = '%s%s' % (TopologyDir, ReferencePDB)
            if RefFrame is None:
                if GLOBAL:
                    NEXT = Generate_reference_for_Clustering( RMSDdir, RefName, TrajDir, TrajNameList,
                                                              TopologyDir, TopologyName, Ending, TimeStep, 
                                                              Select1, Select2, StartFrame, EndingFrame, PartList,
                                                              AmberHome, GromacsHome, Program_Suffix, ReferencePDB )
                else:
                    NEXT = Generate_reference_for_Clustering( RMSDdir, RefName, TrajDir, [TrajNameList[TrajNr-1]],
                                                              TopologyDir, TopologyName, Ending, TimeStep, 
                                                              Select1, Select2, StartFrame, EndingFrame, [PartList[TrajNr-1]],
                                                              AmberHome, GromacsHome, Program_Suffix, ReferencePDB )
                    NEXT += FullCumTrajLenList[TrajNr-1]
            else:
                NEXT = RefFrame
      ## UPDATED v20.01.16: if FixMax > FullLength^2: load whole RMSD at once
        if MaxNumberLines < FullCumTrajLenList[-1] or CentersFinished == False or GLOBAL == False:  
            FullColRMSD, TrajNr, LowerEnd, UpperEnd = Return_FullColRMSD(MatrixDir, TrajNameList, NEXT, 
                                                                         MaxNumberLines, TrajLenDict, 
                                                                         FullCumTrajLenList, StartFrame, 
                                                                         EndingFrame, PartList, 
                                                                         BinFile_precision, GLOBAL, RMSD_SaveAdder)
###########################
##  CALCULATION
###########################
  ####### ClosestCenters: Local|Global, Radius, Begin|End are generated SEPARATELY #######
      ####### ALL centers are found, now assign these to NON-centers #######
        if CentersFinished:
            while LowerEnd <= NEXT and NEXT < UpperEnd:
                if GLOBAL:
                  # v20.01.16 generating CENTER with LOWEST and LARGEST RMSD
                    tempSortIndices = NP.argsort(FullColRMSD[NEXT-LowerEnd, Centers]) # v20.01.16
                    LOWEST_Center  = Centers[tempSortIndices[0]]
                  # v20.01.16 for CENTROIDS: the SndLowest center needs to be assigned, because the lowest = itself
                    if len(tempSortIndices) > 1:
                        SndLOWEST_Center = Centers[tempSortIndices[1]]
                    else:
                        SndLOWEST_Center = LOWEST_Center
                    LARGEST_Center = Centers[tempSortIndices[-1]] 
          ## PROFILE
                    ClusterProfile[NEXT, 2] = LOWEST_Center+1 -AddLength
                else:
                    AddLength = FullCumTrajLenList[TrajNr-1] 
                  # v20.01.16 generating CENTER with LOWEST and LARGEST RMSD
                    tempSortIndices = NP.argsort(FullColRMSD[NEXT-LowerEnd, \
                                                    [elem-AddLength for elem in Centers[TrajNr-1]]])
                    LOWEST_Center = Centers[TrajNr-1][tempSortIndices[0]]
                  # v20.01.16 for CENTROIDS: the SndLowest center needs to be assigned, because the lowest = itself
                    if len(tempSortIndices) > 1:
                        SndLOWEST_Center = Centers[TrajNr-1][tempSortIndices[1]]
                    else:
                        SndLOWEST_Center = LOWEST_Center
                    LARGEST_Center= Centers[TrajNr-1][tempSortIndices[-1]]
          ## PROFILE
                    ClusterProfile[NEXT, 2] = LOWEST_Center+1 -AddLength
          ## TrajNr
                ClusterProfile[NEXT, 1]       = TrajNr
                if ClusterProfile[NEXT, 3] == 1000.0: # NOT center
                    ClusterProfile[NEXT, 3] = 0.0
          ## LOWEST CENTER | LOWEST CENTER RMSD
                    ClusterProfile[NEXT, 4] = LOWEST_Center+1 -AddLength
                    ClusterProfile[NEXT, 5] = FullColRMSD[NEXT-LowerEnd, LOWEST_Center-AddLength]
                else: # CENTER
                    ClusterProfile[NEXT, 4] = SndLOWEST_Center+1 -AddLength
                    ClusterProfile[NEXT, 5] = FullColRMSD[NEXT-LowerEnd, SndLOWEST_Center-AddLength]
          ## LARGEST CENTER | LARGEST CENTER RMSD
                ClusterProfile[NEXT, 6] = LARGEST_Center+1 -AddLength
                ClusterProfile[NEXT, 7] = FullColRMSD[NEXT-LowerEnd, LARGEST_Center-AddLength]
              # v20.01.16 generate CENTER with LOWEST and LARGEST RMSD also FOR CENTROIDS
                NEXT += 1
                TrajNr = NP.searchsorted(FullCumTrajLenList, NEXT, side='right')
        elif not CentersFinished:
        ########################
        ## generate ClusterProfile for the current Frames
        ########################
      ####### INIT FIRST FRAME #######
            if (ReferencePDB is None and (NEXT == 0 or (GLOBAL == False and LastTrajNr != TrajNr))) or \
               (ReferencePDB is not None and Critters != -1):
                ClusterProfile[NEXT, 2] = NEXT+1 - (0 if GLOBAL else FullCumTrajLenList[TrajNr-1]) -AddLength
                ClusterProfile[NEXT, 3] = 0.0
                ClusterProfile[NEXT, 1] = TrajNr
                LastTrajNr = TrajNr

                if GLOBAL:
                    Centers.append(NEXT)
                else:
                    Centers[TrajNr-1].append(NEXT)
                INDICES_First = NP.argsort(FullColRMSD[NEXT-LowerEnd,:])
              ## Store RMSD/INDICES > Threshold AND current frame = new cluster
          #### IF NO NEXT PRESENT, CenterFinished - only ONE cluster!! ####
                if Threshold > FullColRMSD[NEXT-LowerEnd,INDICES_First[-1]]:
                    if GLOBAL or TrajNr == len(TrajNameList):
                        CentersFinished = True
                        NEXT = 0
                        TrajNr = NP.searchsorted(FullCumTrajLenList, NEXT, side='right')
                      ## UPDATED v25.01.16: if CenterFinished: START AT FRAME 1 to calculate CLOSEST/FARTHEST
                    G_INDICES = []; G_RMSD = []
                    if not GLOBAL and not CentersFinished:
                        NEXT = FullCumTrajLenList[TrajNr]
                    ## AddLength must be added to the LOCAL indices, because the full RMSD might have different indices
                        AddLength = FullCumTrajLenList[TrajNr]
                        TrajNr = NP.searchsorted(FullCumTrajLenList, NEXT, side='right')
                else:
                    ThreshIndex = NP.searchsorted(FullColRMSD[NEXT-LowerEnd,INDICES_First], Threshold)
            ## G_RMSD holds ALL RMSD values > Threshold, SHOULD BE SORTED in ASCENDING order
            ## G_RMSD & G_INDICES have the same indices, thus DELETE and RESORT is for both identical
                    G_RMSD = NP.copy(FullColRMSD[NEXT-LowerEnd,INDICES_First[ThreshIndex:]])
                    G_INDICES = NP.copy(INDICES_First[ThreshIndex:]+AddLength)
                    NEXT = G_INDICES[0]
                    TrajNr = NP.searchsorted(FullCumTrajLenList, NEXT, side='right')
                  ####
                    if GLOBAL:
                        Centers.append(NEXT)
                    else:
                        Centers[TrajNr-1].append(NEXT)
  ####### NEXT step ####### until FullColRMSD does NOT contain the NEXT frame anymore
            while LowerEnd <= NEXT and NEXT < UpperEnd and not CentersFinished:
                ClusterProfile[NEXT, 2] = NEXT+1 - (0 if GLOBAL else FullCumTrajLenList[TrajNr-1]) -AddLength
                ClusterProfile[NEXT, 3] = G_RMSD[0] # 0.0
                ClusterProfile[NEXT, 1] = TrajNr
                LastTrajNr = TrajNr
                ## Find ThreshIndex for NEXT
                NEXT_INDICES = NP.argsort(FullColRMSD[NEXT-LowerEnd,:])
                #---- extract ThreshIndex
                ThreshIndex = NP.searchsorted(FullColRMSD[NEXT-LowerEnd,NEXT_INDICES], Threshold)
                #---- extract INDICES smaller ThreshIndex
                Deleter = NP.concatenate(([NEXT], NEXT_INDICES[0:ThreshIndex]+AddLength))
                #---- Sort G_INDICES
                Sort_G_INDICES = NP.argsort(G_INDICES)
                #---- Extract Indices, which have to be deleted
                DEL_Indices = NP.searchsorted(G_INDICES[Sort_G_INDICES], Deleter)
                #---- Check, if DEL_Indices in G_INDICES
                DEL_Indices = [DEL_Indices[elem] for elem in range(len(Deleter)) \
                               if len(G_INDICES) > DEL_Indices[elem] \
                               and Deleter[elem] == G_INDICES[Sort_G_INDICES][DEL_Indices[elem]]]
                #---- delete Duplicates from DEL_Indices
                DEL_Indices = NP.unique(DEL_Indices)
          ## if EVERY INDEX will be deleted: ALL centers are found 
                if len(G_RMSD) > len(DEL_Indices):
                    G_INDICES = NP.delete(G_INDICES, Sort_G_INDICES[DEL_Indices])
                    G_RMSD    = NP.delete(G_RMSD,    Sort_G_INDICES[DEL_Indices])
                  ## RE-SORT 
                    G_RMSD    = NP.divide( NP.add(FullColRMSD[NEXT-LowerEnd, G_INDICES-AddLength], G_RMSD), 2 )
                    RESORT    = NP.argsort(G_RMSD)
                    G_RMSD    = G_RMSD[RESORT]
                    G_INDICES = G_INDICES[RESORT]
                  ## define NEXT
                    NEXT = G_INDICES[0]
                    TrajNr = NP.searchsorted(FullCumTrajLenList, NEXT, side='right')
                  ####
                    if GLOBAL:
                        Centers.append(NEXT)
                    else:
                        Centers[TrajNr-1].append(NEXT)
          ####### ALL centers are found = Ending criteria #######
                elif len(G_RMSD) <= len(DEL_Indices) or len(G_RMSD) == 0:
                    if GLOBAL or TrajNr == len(TrajNameList):
                        CentersFinished = True
                        NEXT = 0
                        TrajNr = NP.searchsorted(FullCumTrajLenList, NEXT, side='right')
                      ## UPDATED v25.01.16: if CenterFinished: START AT FRAME 1 to calculate CLOSEST/FARTHEST  
                    G_INDICES = []; G_RMSD = []
                    if not GLOBAL and not CentersFinished:
                        NEXT      = FullCumTrajLenList[TrajNr]
                        AddLength = FullCumTrajLenList[TrajNr]
                        TrajNr = NP.searchsorted(FullCumTrajLenList, NEXT, side='right')
  ####### ENDING criteria for BOTH METHODS #######
        if NEXT >= len(ClusterProfile[:, 0]):
            Finished = True
  ####### SAVE CLUSTER PROFILES
    t2=time.time()
    #---- generate Directories  
    Generate_Directories(SaveDir)
#---- updated_TrajLengthList
    updated_TrjLenList = []
    for trajY in range(len(TrajNameList)):
        PartY = PartList[trajY]
        if PartY == 1: ## NO SPLIT or PART1 for trajY ##
            BeginY = StartFrame;  EndY = min(EndingFrame, TrajLenDict['%s' % (trajY)][0])
            updated_TrjLenList.append(EndY - BeginY)
        else:
            for PaY in range(1,PartY+1):
                if PartY == 1:
                    BeginY = StartFrame;  EndY = min(EndingFrame, TrajLenDict['%s%s' % (trajY, PartY)][0])
                else:
                    BeginY = max(0,StartFrame-NP.sum([TrajLenDict['%s_part%s' % (trajY, elem)][0] \
                                                      for elem in range(1,PaY)]))
                    EndY   = min(EndingFrame, NP.sum([TrajLenDict['%s_part%s' % (trajY, elem)][0] \
                                                  for elem in range(1,1+PaY)]))
                    EndY   = EndY - NP.sum([TrajLenDict['%s_part%s' % (trajY, elem)][0] \
                                          for elem in range(1,PaY)])
                updated_TrjLenList.append(max(0,EndY - BeginY))
#----
    if NP.any(updated_TrjLenList):
        HEADER ='This File stores the ClusterProfile/Centers using >>ClosestCenters<< of the given TrajNameList\n'+\
                'elapsed time: {} seconds\n'.format(round(t2-t1,2))+\
                'Threshold = '+str(Threshold)+'\n'+\
                'TrajNameList = {}\n'.format(TrajNameList)+\
                'TrjLenList = {}\n'.format(TrajLengthList)+\
                'updated TrjLenList = {}\n'.format(updated_TrjLenList)+\
                'TimeStep = '+str(TimeStep)+'\n'+\
                'StartFrame = '+str(StartFrame)+'\n'+\
                'EndingFrame = '+str(EndingFrame)+'\n'+\
                'ReferencePDB = '+str(ReferencePDB)+'\n'
        HEADER ='%s*******\n%s PROFILE: Radius = %s\n*******\n' % \
                    (HEADER,'GLOBAL' if GLOBAL else 'LOCAL', Threshold) 

      ####### SAVE CLUSTER CENTERS
        if GLOBAL: ## = GLOBAL
            ## UPDATED v20.10.15: changed HEADER and ADDED values
            NP.savetxt('%s%s_R%s_%s-%s_Centers_GLOBAL.txt' % (SaveDir, SaveName, Threshold, StartFrame, EndingFrame),
                       NP.reshape(NP.concatenate( \
                            ([0], [Threshold], [len(Centers)], [int(elem+1) for elem in Centers]) ), (1,3+len(Centers))), \
                       fmt='%i %s %i   '+'%i '*len(Centers), 
                       header=''.join(['# '+elem+'\n' for elem in HEADER.split('\n')[0:-1]])+\
                              'TrajNr | Radius | Nr of Clusters | Centers')
        else:
            with open('%s%s_R%s_%s-%s_Centers_LOCAL.txt' % \
                      (SaveDir, SaveName, Threshold, StartFrame, EndingFrame), 'w') as OUTPUT:
                OUTPUT.write(''.join(['# '+elem+'\n' for elem in HEADER.split('\n')[0:-1]]))
                ## UPDATED v20.10.15: changed HEADER and ADDED values
                OUTPUT.write('# TrajNr | Radius | Nr of Clusters | Centers\n')
                Number = 1
                ## UPDATED v17.11.15: every center for different trajNr is subtracted by INDEXLENGTH[TrajNr-1]
                for Kai in range(len(Centers)):
                    OUTPUT.write(str(Number)+' '+str(Threshold)+' '+str(len(Centers[Kai]))+'   ')
                    OUTPUT.write('  '.join([str(elem+1-FullCumTrajLenList[Kai]) for elem in Centers[Kai]]))
                    OUTPUT.write('\n')
                    Number += 1
        HEADER = HEADER + """
               |                       Radius = %s                            |
               |  eff Clust     |    next Center   |     farthest Center      |
Frame | TrajNr | PROF | effRMSD | nextC | nextRMSD | farthestC | farthestRMSD |
                          """ % (Threshold)
      ####### SAVE CLUSTER PROFILES
        NP.savetxt('%s%s_R%s_%s-%s_%s.txt' % \
                   (SaveDir, SaveName, Threshold, StartFrame, EndingFrame, 'GLOBAL' if GLOBAL else 'LOCAL'), 
                   ClusterProfile, fmt='%i %i    %i %.4f    %i %.4f    %i %.4f', header=HEADER)

###############
#-------------
###############

def Generate_reference_for_Clustering(RMSDdir, RefName, TrajDir, TrajNameList, TopologyDir, TopologyName, 
                                      Ending, TimeStep, Select1, Select2=None, StartFrame=0, EndingFrame=NP.infty, 
                                      PartList=None, AmberHome='', GromacsHome='', Program_Suffix='', ReferencePDB=None):
    """ 
v12.10.16
        Helper function to generate the correct reference for the effective clustering, calculating the RMSD values of
        the submitted Xtc-files to a specific reference and choosing the INDEX with the LOWEST RMSD ABOVE the THRESHOLD
        as first frame for effective clustering
        
        (1) calculate for every trajectory the RMSD values to the Reference (PDB) for different simulation times
        (2) combine the single RMSD curves with correct indices of different trajectory names
        (3) delete single RMSD files
        
INPUT:
    RMSDdir           : {STRING}    directory, where to store single and full RMSD curves to a submitted reference,
                                        e.g. 'RMSD_curves/'
                                                   if is None, RMSDdir = SaveDir;
    RefName           : {STRING}    name affix of the RMSD curves, 'RMSD_%s_ref%s.xvg' % (TrajName, RefName)', 
                                        e.g. 'CrystalStructure';
    TrajDir           : {STRING}    Directory, where the trajectories are located to which RMSD curves are calculated,
                                        e.g. 'TrajDir/';
    TrajNameList      : {LIST}      list of Trajectory name prefixes WITHOUT ENDING, 
                                    Names of the Block RMSD matrices / distributions refer to these names,
                                    e.g. ['MD1', 'MD2', ...] <-> MD1_bin.dat, MD2_bin.dat, MD1_MD2_bin.dat, ...; 
                                    
    TopologyDir       : {STRING}    Directory, where the topology file is located, for GROMACS, this is
                                    equal to the Reference(PDB), for AMBER, this is a .top file, e.g. 'TrajDir/';
    TopologyName      : {STRING}    topology name, for GROMACS this has to contain the reference(PDB)
                                                    for AMBER, this is the input topology .top;
    Ending            : {STRING}    ending of the trajectories, normally '.xtc'/.trr'/'.pdb' or '.nc'/'.netcdf';
    TimeStep          : {FLOAT/INT} GROMACS, -dt 't MOD dt = first [ns]' 
                                    AMBER    skip dt-th frame 'first last skip'
                                    if TimeStep is one, use ALL frames in the trajectory;
    Select1           : {STRING}    FIT  selection GROMACS, e.g. 'Backbone'
                                    CALC selection AMBER  , e.g. '@N,CA,C', see Amber syntax;
    Select2           : {STRING}    <default None> CALC selection for GROMACS, e.g. 'Backbone';
    StartFrame        : {INT}       <default 0>    starting frame of Trajectories/RMSD matrices,
                                    to select different simulation times/lengths together with 'EndingFrame';
    EndingFrame       : {INT}       <default NP.infty> ending frame of Trajectories/RMSD matrices,
                                    to select different simulation times/lengths together with 'StartingFrame';
    PartList          : {INT-LIST}  <default None> defines into how many parts the single trajectories are split, due to
                                    memory reasons,
                                        1. len(PartList) == len(TrajNameList) !!  ||   
                                        2. PartList = [1,2,3], (MD1.xtc -> MD1.xtc),
                                                               (MD2.xtc -> MD2_part1.xtc, MD2_part2.xtc),
                                                               (MD3.xtc -> MD3_part1.xtc, MD3_part2.xtc, MD3_part3.xtc), ||
                                        3. default PartList = None -> PartList = [1]*len(TrajNameList);    
    AmberHome         : {STRING}    <default ''>   if cpptraj is not in environmental variables, define directory to 'cpptraj',
                                    e.g. /home/user/Software/amber14/bin/;
    GromacsHome       : {STRING}    <default ''>   if g_rms | gmx_suffix rms is not in environmental variables, define directory to them,
                                    e.g. /home/user/Software/gromacs/bin/;
    Program_Suffix    : {STRING}    <default ''>   gmx_suffix for Gromacs installation, e.g. '_467' for 'g_rms_467';
    ReferencePDB      : {STRING}    <default None>, only for AMBER, possible reference, to which the RMSD between all trajectories are
                                        calculated, then the closest structure is the starting point for the clustering, WITH possible DIRECTORY
                                                        e.g. 'Directory/Crystalstructure.pdb';
OUTPUT:
    return NEXT : {INT} correct value for NEXT frame/cluster centroid
    
    NEXT is related to the length of the trajectory, i.e. if multiple trajectories are used for LOCAL, 
    NEXT = NEXT + FullCumTrajLenList[TrajNr-1], but this is done in <Generate_Clustering()>
    """
    if len(TrajNameList) > 1:
        GLOBAL = '_GLOBAL'
    else:
        GLOBAL = '_LOCAL'
    if PartList is None:
        PartList = [1]*len(TrajNameList)
    #---- generate Directories  
    Generate_Directories(RMSDdir)
    #----------------------
    if not os.path.exists('%sFullRMSD_%s-%s_ref%s_%s-%s%s.xvg' % \
                           (RMSDdir, TrajNameList[0], TrajNameList[-1], RefName, StartFrame, EndingFrame, GLOBAL)):
      #### GENERATE single Traj/parts RMSD curves to ref
        IIndex = 0
        for TrajName in TrajNameList:
            if os.path.exists('%s%s%s' % (TrajDir, TrajName, Ending)):
                if not os.path.exists('%sRMSD_%s_ref%s.xvg' % (RMSDdir, TrajName, RefName)):
                    Generate_RMSD_Matrix(TrajDir, TopologyDir, TrajName+Ending, TopologyName, RMSDdir, RMSDdir, 
                                        'RMSD_%s_ref%s' % (TrajName, RefName), Select1, Select2, TimeStep, 
                                        AmberHome, GromacsHome, None, None, SecondTraj=None, Fit='rot+trans', 
                                        Program_Suffix=Program_Suffix, ReferencePDB=ReferencePDB, Bin=False)
            elif PartList[IIndex] > 1:
                for Parts in range(1,1+PartList[IIndex]):
                    if os.path.exists('%s%s_part%s%s' % (TrajDir, TrajName, Parts, Ending)):
                        if not os.path.exists('%sRMSD_%s_ref%s_part%s.xvg' % (RMSDdir, TrajName, RefName, Parts)):
                           Generate_RMSD_Matrix(TrajDir, TopologyDir, '%s_part%s%s' % (TrajName, Parts, Ending), 
                                                TopologyName, RMSDdir, 
                                                RMSDdir, 'RMSD_%s_ref%s_part%s' % (TrajName, RefName, Parts), 
                                                Select1, Select2, TimeStep, AmberHome, GromacsHome, None, None, SecondTraj=None, 
                                                Fit='rot+trans', Program_Suffix=Program_Suffix, ReferencePDB=ReferencePDB, 
                                                Bin=False)
                    else:
                        raise ValueError('The Trajectory\n\t%s%s_part%s%s\ndoes not exist' % \
                                         (TrajDir, TrajName, Parts, Ending))
            else:
                raise ValueError('The Trajectory\n\t%s%s%s\ndoes not exist' % (TrajDir, TrajName, Ending))
            IIndex += 1
      ##### concatenation IS necessary
        if len(TrajNameList) > 1 or PartList[0] > 1: ## concatenate different RMSD files
            IIndex = 0
            for TrajName in TrajNameList:
              ## multiple PARTS
                if PartList[IIndex] > 1 and not os.path.exists('%sRMSD_%s_ref%s.xvg' % (RMSDdir, TrajName, RefName)):
                    for Parts in range(1, 1+PartList[IIndex]):
                        if 'temp_parts' not in locals():
                            temp_parts = NP.genfromtxt('%sRMSD_%s_ref%s%s.xvg' % \
                                 (RMSDdir, TrajName, RefName, '_part%s' % Parts))
                        else:
                            temp_parts = NP.concatenate((temp_parts, NP.genfromtxt('%sRMSD_%s_ref%s%s.xvg' % \
                                 (RMSDdir, TrajName, RefName, '_part%s' % Parts))), axis=0)
                    NP.savetxt('%sRMSD_%s_ref%s.xvg' % (RMSDdir, TrajName, RefName), temp_parts, fmt='%.7f  %.7f')
                    del temp_parts
              ## multiple full Trajectory
                if len(TrajNameList) > 1:
                    if 'temp' not in locals():
                        temp = NP.genfromtxt('%sRMSD_%s_ref%s.xvg' % \
                                             (RMSDdir, TrajName, RefName))[StartFrame:EndingFrame,:]
                    else:
                        temp = NP.concatenate((temp, NP.genfromtxt('%sRMSD_%s_ref%s.xvg' % \
                                             (RMSDdir, TrajName, RefName))[StartFrame:EndingFrame,:]))
                IIndex += 1
          ### concatenate if multiple TrajNameList, ELSE load only one traj
            if len(TrajNameList) == 1:
                Ref_RMSD = NP.genfromtxt('%sRMSD_%s_ref%s.xvg' % (RMSDdir, TrajName, RefName))[StartFrame:EndingFrame]
            else:
                HEADER = 'RMSD curve to reference:\n\tTrajNameList = %s\n\tTopologyName = %s\n\tReferencePDB = %s' % \
                            (TrajNameList, TopologyName, ReferencePDB)
                NP.savetxt('%sFullRMSD_%s-%s_ref%s_%s-%s%s.xvg' % \
                           (RMSDdir, TrajNameList[0], TrajNameList[-1], RefName, StartFrame, EndingFrame, GLOBAL), 
                           temp, header=HEADER, fmt='%.7f  %.7f')
                Ref_RMSD = NP.genfromtxt('%sFullRMSD_%s-%s_ref%s_%s-%s%s.xvg' % \
                           (RMSDdir, TrajNameList[0], TrajNameList[-1], RefName, StartFrame, EndingFrame, GLOBAL))
      ##### concatenation IS NOT necessary
        else:
            Ref_RMSD = NP.genfromtxt('%sRMSD_%s_ref%s.xvg' % (RMSDdir, TrajName, RefName))[StartFrame:EndingFrame]
    else:
        Ref_RMSD = NP.genfromtxt('%sFullRMSD_%s-%s_ref%s_%s-%s%s.xvg' % \
                           (RMSDdir, TrajNameList[0], TrajNameList[-1], RefName, StartFrame, EndingFrame, GLOBAL))
    
  #### the lowest RMSD corresponds to the closest frame = NEXT
    NEXT = NP.argsort(Ref_RMSD[:,1])[0]
  #### 
    return NEXT

###############
#-------------
###############
def Return_FullColRMSD(MatrixDir, TrajNameList, CurrentRow, MaxNumberLines, TrajLenDict, FullCumTrajLenList,
                       StartFrame=0, EndingFrame=NP.infty, PartList=None, BinFile_precision=NP.float32, GLOBAL=True,
                       EventCurve=False, trajYList=None, RMSD_SaveAdder=''):
    """ 
v11.10.16
    - Helper function for <Generate_EventCurves()> AND <Generate_Clustering()> to return the full ROW of the 
      whole RMSD matrix for every submitted trajectory files
    - to select adequate cluster, it is necessary to load at least ONE FULL ROW of all involved RMSD matrices
    - with MaxNumberLines, one can select, how many ROWS are loaded, to be processed directly
    - recommended is to load at least one RMSD block (if possible due to RAM), i.e. if all RMSD matrices have 2000 rows, 
      MaxNumberLines = 2000, this loads one row of block matrices
        
INPUT:
    MatrixDir          : {STRING}    Directory, where RMSD matrices are stored, e.g. 'RMSD_matrices/';
    TrajNameList       : {LIST}      list of Trajectory name prefixes WITHOUT ENDING, 
                                    Names of the Block RMSD matrices / distributions refer to these names,
                                    e.g. ['MD1', 'MD2', ...] <-> MD1_bin.dat, MD2_bin.dat, MD1_MD2_bin.dat, ...;
    CurrentRow         : {INT}       current row of the FULL RMSD MATRIX, refers to NEXT of Generate_Clustering();
    MaxNumberLines     : {INT}       Maximal number of lines which are loaded from the FullRMSD matrix, 
                                    it inflicts directly the memory usage, if MaxNumberLines > sum(TrajLengthList), all
                                    RMSD matrices are loaded at once, recommended using length of one RMSD block,
                                    e.g. if one block RMSDmatrix.shape == (2000,2000), try to use MaxNumberLines = 2000;
    TrajLenDict        : {DICT}      Dictionary, which stores the lengths [in frames] of the trajectories, 
                                        TrajLenDict['%s%s' % (trajName, Part)] = (TrajLengthList[Index], Index);
    FullCumTrajLenList : {INT-LIST}  cumulative Trajectory lengths [in frames] starting from 0, 
                            len(FullCumTrajLenList) == len(TrajNameList), e.g. [0, 1000, 2000] for lengths [1000,1000];
    StartFrame         : {INT}       <default 0>    starting frame of Trajectories/RMSD matrices,
                                    to select different simulation times/lengths together with 'EndingFrame';
    EndingFrame        : {INT}       <default NP.infty> ending frame of Trajectories/RMSD matrices,
                                    to select different simulation times/lengths together with 'StartingFrame';
    PartList           : {INT-LIST} <default None> defines into how many parts the single trajectories are split, due to
                                    memory reasons,
                                        1. len(PartList) == len(TrajNameList) !!  ||   
                                        2. PartList = [1,2,3], (MD1.xtc -> MD1.xtc),
                                                               (MD2.xtc -> MD2_part1.xtc, MD2_part2.xtc),
                                                               (MD3.xtc -> MD3_part1.xtc, MD3_part2.xtc, MD3_part3.xtc), ||
                                        3. default PartList = None -> PartList = [1]*len(TrajNameList);    
    BinFile_precision  : {TYPE}     FORMAT, [GROMACS/BINARY] single precision 'float32', double precision 'float64', [AMBER/ELSE] 'None';
    GLOBAL             : {BOOL}     <default True> if True,  a GLOBAL clustering is applied for all concatenated trajectories
                                                   if False, every trajectory is clustered separately;
    EventCurve         : {BOOL}     <default False> if True,  FullColRMSD is returned for Generate_EventCurves()
                                                    if False, FullColRMSD is returned for Generate_Clustering();
    trajYList          : {INT-LIST} <default None>  for Generate_EventCurves(), defines the trajectory for trajY;
OUTPUT:
    if EventCurve:
        return FullColRMSD, BeginX, EndX, BeginY, EndY, LowerEnd, UpperEnd
    else:
        return FullColRMSD, CurrentTrajNr, LowerEnd, UpperEnd
    
    FullColRMSD   - {NP.ndarray} returns the FullColRMSD matrix with maximally allowed rows which fit into the RAM
    CurrentTrajNr - {INT}        returns the corresponding TrajNr
    LowerEnd      - {INT}        the FullColRMSD reaches from [LowerEnd, UpperEnd] compared to the FullRMSD matrix
    UpperEnd      - {INT}        the FullColRMSD reaches from [LowerEnd, UpperEnd] compared to the FullRMSD matrix
    """
#---- INIT PartList
    if PartList is None:
        PartList = [1]*len(TrajNameList)
#---- ADJUST MaxNumberLines
    if MaxNumberLines < FullCumTrajLenList[-1] or GLOBAL == False:
        trajXList = [NP.searchsorted(FullCumTrajLenList, CurrentRow, side='right')-1]
        if EventCurve:
            MaxNumberLines = min(MaxNumberLines, 
                                 FullCumTrajLenList[trajXList[0]+1]-CurrentRow,
                                 FullCumTrajLenList[trajXList[0]+1]-FullCumTrajLenList[trajXList[0]])
        else:
            MaxNumberLines = min(MaxNumberLines, FullCumTrajLenList[trajXList[0]+1]-FullCumTrajLenList[trajXList[0]])
    else:
        trajXList = range(len(TrajNameList))
        MaxNumberLines = FullCumTrajLenList[-1]
#---- INIT trajYList
    if not GLOBAL and trajYList is None:
        trajYList = [trajXList[0]]
    elif trajYList is None:
        trajYList = range(len(TrajNameList))
###### ###### ###### ##### #######
#---- INIT FullColRMSD
    if EventCurve:
        FullColRMSD = NP.zeros( (MaxNumberLines, 
            max(0,min(EndingFrame,FullCumTrajLenList[trajYList[-1]+1]-FullCumTrajLenList[trajYList[-1]])-StartFrame)) )
      #####  
        if max(0,min(EndingFrame,FullCumTrajLenList[trajYList[-1]+1]-FullCumTrajLenList[trajYList[-1]])-StartFrame) == 0:
            return FullColRMSD, 100, 0, 100, 0, 100, 0
      #####
    elif GLOBAL:
        FullColRMSD = NP.zeros( (MaxNumberLines, FullCumTrajLenList[-1]) )
    else:
        if len(trajXList) > 1:
            raise ValueError('Local clustering with more than one trajectory is done!\ntrajXList = %s' % trajXList)
        else:
            FullColRMSD = NP.zeros( (MaxNumberLines, 
                                     FullCumTrajLenList[trajXList[0]+1]-FullCumTrajLenList[trajXList[0]]) )
###### ###### ###### ###### ######
#---- INIT LowerRow | UpperRow, Limits for loaded RMSD_mat
    #--- use Limits to best fit around CurrentRow
    if (EventCurve and MaxNumberLines >= FullCumTrajLenList[trajXList[0]+1]-FullCumTrajLenList[trajXList[0]]) or \
       (MaxNumberLines >= FullCumTrajLenList[-1]) or \
       (not GLOBAL and MaxNumberLines >= FullCumTrajLenList[trajXList[0]+1]-FullCumTrajLenList[trajXList[0]]):
        LowerRow = 0
        UpperRow = MaxNumberLines
    else:
        LowerRow = max(0,CurrentRow-FullCumTrajLenList[trajXList[0]] - (0 if EventCurve else MaxNumberLines/2) )
        UpperRow = LowerRow+MaxNumberLines
        if not EventCurve and UpperRow > FullCumTrajLenList[trajXList[0]+1]-FullCumTrajLenList[trajXList[0]]:
            UpperRow = FullCumTrajLenList[trajXList[0]+1]-FullCumTrajLenList[trajXList[0]]
            LowerRow = UpperRow - MaxNumberLines
  #############
#--- DEFINE indices which defines the limits of FullColRMSD within the FullRMSD
#    if all RMSD matrices are concatenated, LowerEnd = 0 & UpperEnd = last
    LowerEnd = FullCumTrajLenList[trajXList[0]]+LowerRow
    UpperEnd = LowerEnd+len(FullColRMSD[:,0])
  #############
###### ###### ###### ###### ######
#---- INIT Loop for the necessary single RMSD matrices
    Looper = []
    for trajX in trajXList:      ## "Reference"-trajectory loop, to which Events are counted
        for PartX in [(('_part%s' % Part) if PartList[trajX]!=1 else '') for Part in range(1,PartList[trajX]+1)]: 
            for trajY in trajYList:  ## "Reference"-trajectory loop, to which Events are counted
                for PartY in [(('_part%s' % Part) if PartList[trajY]!=1 else '') \
                                              for Part in range(1,PartList[trajY]+1)]: 
                    if PartX != '' and EventCurve:
                        ## v22.08.16: if RMSD matrix is split: load only matching part
                        if (NP.sum([TrajLenDict['%s_part%s' % (trajX, elem)][0] \
                                   for elem in range(1, int(PartX.replace('_part','')))]) <= LowerRow and \
                            NP.sum([TrajLenDict['%s_part%s' % (trajX, elem)][0] \
                                   for elem in range(1, 1+int(PartX.replace('_part','')))]) > LowerRow) or \
                           (NP.sum([TrajLenDict['%s_part%s' % (trajX, elem)][0] \
                                   for elem in range(1, int(PartX.replace('_part','')))]) < UpperRow and \
                            NP.sum([TrajLenDict['%s_part%s' % (trajX, elem)][0] \
                                   for elem in range(1, 1+int(PartX.replace('_part','')))]) >= UpperRow) or \
                           (NP.sum([TrajLenDict['%s_part%s' % (trajX, elem)][0] \
                                   for elem in range(1, int(PartX.replace('_part','')))]) >= LowerRow and \
                            NP.sum([TrajLenDict['%s_part%s' % (trajX, elem)][0] \
                                   for elem in range(1, 1+int(PartX.replace('_part','')))]) < UpperRow):
                                Looper.append((trajX, trajY, PartX, PartY))
                                NP.sum([TrajLenDict['%s_part%s' % (trajX, elem)][0] \
                                       for elem in range(1, int(PartX.replace('_part','')))]),\
                                NP.sum([TrajLenDict['%s_part%s' % (trajX, elem)][0] \
                                       for elem in range(1, 1+int(PartX.replace('_part','')))])
                    else:
                        Looper.append((trajX, trajY, PartX, PartY))
#### #### #### #### #### 
#### LOAD RMSD matrix
#### #### #### #### #### 
    RowIndex = 0; ColIndex = 0;
    LenX = 0; LenY = 0;
    if Looper != []:
        temptrajX = Looper[0][0]; temptrajY = Looper[0][1]; tempPartX = Looper[0][2]; tempPartY = Looper[0][3]
    for (trajX, trajY, PartX, PartY) in Looper:
        #---------- if LOCAL: only diagonals must be used, if GLOBAL: all trajY are used
        if trajX == trajY or GLOBAL:
    #### INIT BeginX | EndX, for EventCurve all frames are used
            if EventCurve:
                BeginX = 0; EndX = TrajLenDict['%s%s' % (trajX, PartX)][0]
            else:
                if PartX == '' or PartX == '_part1': ## NO SPLIT or PART1 for trajX ##
                    BeginX = StartFrame;  EndX = min(EndingFrame, TrajLenDict['%s%s' % (trajX, PartX)][0])
                else:
                    BeginX = max(0,StartFrame-NP.sum([TrajLenDict['%s_part%s' % (trajX, elem)][0] \
                                                  for elem in range(1,int(PartX.split('_part')[1]))]))
                    EndX   = min(EndingFrame, NP.sum([TrajLenDict['%s_part%s' % (trajX, elem)][0] \
                                                  for elem in range(1,1+int(PartX.split('_part')[1]))]))
                    # v03.06.16: EndY must be subtracted by the LENGTHS of all SMALLER parts
                    EndX   = EndX - NP.sum([TrajLenDict['%s_part%s' % (trajX, elem)][0] \
                                                  for elem in range(1,int(PartX.split('_part')[1]))])
    #### INIT BeginY | EndY
            if PartY == '' or PartY == '_part1': ## NO SPLIT or PART1 for trajY ##
                BeginY = StartFrame;  EndY = min(EndingFrame, TrajLenDict['%s%s' % (trajY, PartY)][0])
            else:
                BeginY = max(0,StartFrame-NP.sum([TrajLenDict['%s_part%s' % (trajY, elem)][0] \
                                              for elem in range(1,int(PartY.split('_part')[1]))]))
                EndY   = min(EndingFrame, NP.sum([TrajLenDict['%s_part%s' % (trajY, elem)][0] \
                                              for elem in range(1,1+int(PartY.split('_part')[1]))]))
                # v03.06.16: EndY must be subtracted by the LENGTHS of all SMALLER parts
                EndY   = EndY - NP.sum([TrajLenDict['%s_part%s' % (trajY, elem)][0] \
                                              for elem in range(1,int(PartY.split('_part')[1]))])
    #### DEFINE RowIndex | ColIndex for FullColRMSD matrix
            if BeginX < EndX and BeginY < EndY:
              #### if trajY or PartY changes, increment ColIndex by lastly stored len(temp[0,:])
                if temptrajY != trajY or tempPartY != PartY:
                    temptrajY = trajY; tempPartY = PartY
                    ColIndex += LenY
              #### if trajX or PartX changes, increment RowIndex by lastly stored len(temp[:,0]) & ColIndex = 0
                if temptrajX != trajX or tempPartX != PartX:
                    temptrajX = trajX; tempPartX = PartX
                    RowIndex += LenX
                    ColIndex = 0
  ###########################
                if (EventCurve and MaxNumberLines >= FullCumTrajLenList[trajX+1]-FullCumTrajLenList[trajX]) or \
                   (MaxNumberLines >= FullCumTrajLenList[-1]) or \
                   (not GLOBAL and MaxNumberLines >= FullCumTrajLenList[trajX+1]-FullCumTrajLenList[trajX]):
                    LowerRow = 0
                    UpperRow = MaxNumberLines
                elif PartX == '':
                    LowerRow = max(0,CurrentRow-FullCumTrajLenList[trajX] - (0 if EventCurve else MaxNumberLines/2) )
                    UpperRow = LowerRow+MaxNumberLines
                    if not EventCurve and UpperRow > FullCumTrajLenList[trajX+1]-FullCumTrajLenList[trajX]:
                        UpperRow = FullCumTrajLenList[trajX+1]-FullCumTrajLenList[trajX]
                        LowerRow = UpperRow - MaxNumberLines
                else:
             #### LOWER BOUND ####
                    if max(0,CurrentRow-FullCumTrajLenList[trajX]-MaxNumberLines/2)+MaxNumberLines <= \
                                FullCumTrajLenList[trajX+1]-FullCumTrajLenList[trajX]:
                        LowerRow = int(max(0,CurrentRow-FullCumTrajLenList[trajX] - (0 if EventCurve else MaxNumberLines/2) -\
                               NP.sum([TrajLenDict['%s_part%s' % (trajX,elem)][2] 
                                       for elem in range(1,int(PartX.replace('_part','')))])))
                        UpperRow = int(min(LowerRow+MaxNumberLines-RowIndex, TrajLenDict['%s%s' % (trajX,PartX)][2]))
             #### UPPER BOUND ####
                    else:
                        UpperRow = TrajLenDict['%s%s' % (trajX,PartX)][2]
                        LowerRow = int(max(0, FullCumTrajLenList[trajX+1]-FullCumTrajLenList[trajX]-\
                                   NP.sum([TrajLenDict['%s_part%s' % (trajX,elem)][2] \
                                           for elem in range(1,int(PartX.replace('_part','')))]) - MaxNumberLines))
  ##########################
    #####
    ## Diagonal RMSD matrix - single traj
    #####
            if trajX == trajY and PartX == PartY and BeginX < EndX and BeginY < EndY:
                FileName = '%s%s%s_bin.dat' % (TrajNameList[trajX], PartX, RMSD_SaveAdder)
                if BinFile_precision is not None:
                    RMSD_mat = NP.fromfile('%s%s' % (MatrixDir, FileName), dtype=BinFile_precision)
                    LenX = NP.sqrt(len(RMSD_mat))
                    LenY = NP.sqrt(len(RMSD_mat))
                    if len(RMSD_mat) != LenX*LenY:
                        raise ValueError(('The RMSD matrix generated by GROMACS is not loaded correctly!\n\tCheck your >>BinFile_precision=%s<< input\n' % \
                                BinFile_precision)+\
                              'The BinFile_precision must correspond to the precision which was used to install GROMACS'+\
                              '\n\t\'float32\' corresponds to single precision\n\t\'float64\' corresponds to double precision!\n'+\
                              'This was tested for GROMACS v4.6 and v5.1, for other versions the RMSD matrices might differ and have to be preprocessed by hand\n'+\
                              'and stored in matrix-form in a txt-file using >>BinFile_precision=None<<')
                    RMSD_mat = NP.reshape(RMSD_mat, (int(LenX), int(LenY)))
                else:
                  #----- TRY TO LOAD NP.NDarray like input, like generated by AmberTools14
                    try:
                        RMSD_mat = NP.genfromtxt('%s%s' % (MatrixDir, FileName))
                    except ValueError:
                        raise ValueError('The RMSD matrix is not stored in a txt-file.\n\t'+\
                                         ('Check your RMSD matrix format and >>BinFile_precision=%s<< input\n' % BinFile_precision)+\
                                         '\'None\' corresponds to txt-file Matrix shape format\n'+\
                                         '\'float32/64\' corresponds to GROMACS binary format')
                    if len(RMSD_mat.shape) != 2:
                        raise ValueError('The RMSD matrix is not stored in Matrix form, (Rows)x(Columns)\n\tCheck your matrices stored in >>%s<<' % MatrixDir)
                   #--- if FIRST COL = Indices: strip them (AmberTools14)
                    if NP.sum(NP.absolute(RMSD_mat[:,0]-range(1,len(RMSD_mat[:,0])+1))) < 10e-6: 
                       #--- AmberTool14 uses Angstrom, thus /10 for NM
                        RMSD_mat = RMSD_mat[:,1:]/10.
                    LenX, LenY = RMSD_mat.shape
             #---- CHECK, if submitted TrajLength match the RMSD matrix length
                if LenX != TrajLenDict['%s%s' % (trajX, PartX)][0]:
                    raise ValueError('submitted TrajLenghtList does not match length of the RMSD matrix\n\t'+\
                                     '%s (%s) != %s' % \
                         (FileName, LenX, TrajLenDict['%s%s' % (trajX, PartX)][0]))
                #if  EventCurve:
                RMSD_mat = RMSD_mat[BeginX:EndX, BeginY:EndY]
                #else:
                #    RMSD_mat = RMSD_mat[BeginX:EndX, :]
                LenX, LenY = RMSD_mat.shape
    #####
    ## Off-Diagonal RMSD matrix - trajX vs trajY
    #####
            elif BeginX < EndX and BeginY < EndY:
                if os.path.exists('%s%s%s_%s%s%s_bin.dat' % \
                                  (MatrixDir, TrajNameList[trajX], PartX, TrajNameList[trajY], PartY, RMSD_SaveAdder)):
                    FileName = '%s%s_%s%s%s_bin.dat' % \
                                      (TrajNameList[trajX], PartX, TrajNameList[trajY], PartY, RMSD_SaveAdder)
                    TransPose = False
                else:
                    FileName = '%s%s_%s%s%s_bin.dat' % \
                                      (TrajNameList[trajY], PartY, TrajNameList[trajX], PartX, RMSD_SaveAdder)
                    TransPose = True
             #---- LenX is the length of the X trajectory, should be equal to the transposed if necessary   
                LenX = TrajLenDict['%s%s' % (trajX, PartX)][0]
                LenY = TrajLenDict['%s%s' % (trajY, PartY)][0]

                if BinFile_precision is not None:
                    RMSD_mat = NP.fromfile('%s%s' % (MatrixDir,FileName), dtype=BinFile_precision)
                    if TransPose:
                        RMSD_mat = NP.transpose(NP.reshape(RMSD_mat, (int(LenY),int(LenX))))
                    else:
                        RMSD_mat = NP.reshape(RMSD_mat, (int(LenX), int(LenY)))
                else:
                  #----- TRY TO LOAD NP.NDarray like input, like generated by AmberTools14
                    try:
                        RMSD_mat = NP.genfromtxt('%s%s' % (MatrixDir,FileName) )
                    except ValueError:
                        raise ValueError('The RMSD matrix is not stored in a txt-file.\n\t'+\
                                         ('Check your RMSD matrix format and >>BinFile_precision=%s<< input\n' % BinFile_precision)+\
                                         '\'None\' corresponds to txt-file Matrix shape format\n'+\
                                         '\'float32/64\' corresponds to GROMACS binary format')
                    if len(RMSD_mat.shape) != 2:
                        raise ValueError('The RMSD matrix is not stored in Matrix form, (Rows)x(Columns)\n\tCheck your matrices stored in >>%s<<' % MatrixDir)
                   #--- if FIRST COL = Indices: strip them (AmberTools14)
                    if NP.sum(NP.absolute(RMSD_mat[:,0]-range(1,len(RMSD_mat[:,0])+1))) < 10e-6: 
                       #--- AmberTool14 uses Angstrom, thus /10 for NM
                        RMSD_mat = RMSD_mat[:,1:]/10.
                    if TransPose:
                        RMSD_mat = NP.transpose(RMSD_mat)
             #---- CHECK, if submitted TrajLength match the RMSD matrix length
                if (LenX,LenY) != RMSD_mat.shape:
                    raise ValueError('submitted TrajLenghtList does not match length of the RMSD matrix\n\t'+\
                                     '%s (%s, %s) != %s' % \
                                     (FileName, LenX, LenY, RMSD_mat.shape))
                #if  EventCurve:
                RMSD_mat = RMSD_mat[BeginX:EndX, BeginY:EndY]
                #else:
                #    RMSD_mat = RMSD_mat[BeginX:EndX, :]
                LenX, LenY = RMSD_mat.shape
         #------- for Rows: use only MaxNumberLines
            if BeginX < EndX and BeginY < EndY:
                RMSD_mat = RMSD_mat[LowerRow:UpperRow,:]
                LenX, LenY = RMSD_mat.shape
    #######
    ## Store RMSD_mat to FullColRMSD
    #######
                FullColRMSD[RowIndex:(RowIndex+LenX), ColIndex:(ColIndex+LenY)] = RMSD_mat
                del RMSD_mat
    CurrentTrajNr = NP.searchsorted(FullCumTrajLenList, CurrentRow, side='right')
  ####################### adjust BeginX|EndX & BeginY|EndY for FULL RMSD matrices, which has to be returned
    if PartList[trajX] > 1:
        BeginX = 0;  EndX = NP.sum([TrajLenDict['%s_part%s' % (trajX, PayX)][0] \
                                             for PayX in range(1,1+PartList[trajX]) \
                                             if TrajLenDict['%s_part%s' % (trajX, PayX)][0] > 0])
    if PartList[trajY] > 1:
        BeginY = StartFrame;  EndY = min(EndingFrame, 
                                         NP.sum([TrajLenDict['%s_part%s' % (trajY, PayY)][0] \
                                                 for PayY in range(1,1+PartList[trajY]) \
                                                 if TrajLenDict['%s_part%s' % (trajY, PayY)][0] > 0]))
  ####################### RETURN
    if EventCurve:
        return FullColRMSD, BeginX, EndX, BeginY, EndY, LowerEnd, UpperEnd
    else:
        return FullColRMSD, CurrentTrajNr, LowerEnd, UpperEnd

###############
#-------------
###############

def Return_TrajLen_for_trajX(trajX, TrajLenDict, PartList, trajYList):
    """
v11.07.16
    - merges trajectory lengths if trajectories are split into different parts 
    - returns combined lenghts of all parts of one trajectory
    - for all trajectories, one can simply define the following array
    
    TrajectoryLengths = [max(0,min(EndingFrame,Return_TrajLen_for_trajX(elem,
                                                                                TrajLenDict,
                                                                                PartList, 
                                                                                range(len(TrajNameList))))-
                               StartFrame) for elem in range(len(TrajNameList))]
    """
    return int(NP.sum([NP.sum([NP.sum([TrajLenDict['%s%s' % (Kai, '_part%s' % elem if PartList[Kai] > 1 else '')][0] \
                                          for elem in range(1,PartList[Kai]+1)])]) \
                             for Kai in [GAGA for GAGA in trajYList if GAGA < trajX+1]])) - \
           int(NP.sum([NP.sum([NP.sum([TrajLenDict['%s%s' % (Kai, '_part%s' % elem if PartList[Kai] > 1 else '')][0] \
                                          for elem in range(1,PartList[Kai]+1)])]) \
                             for Kai in [GAGA for GAGA in trajYList if GAGA < trajX]]))

###############
#-------------
###############
def Merge_Clustering_different_Thresholds(SingleClustDir, SaveDir, SaveName, ThresholdList, StartFrame, 
                                          EndingFrame, GLOBAL):
    """
v12.10.16
- function to merge different clustering files with different cluster thresholds but same trajectories 
  and [StartFrame,EndingFrame]
- beforehand, clusterings <Generate_Clustering()> are calculated for each threshold separately (possibly on multiple machines/cores simultanously)
- these are stored for simplicity in separate files, to not limit the speed waiting on the slowest cluster threshold
- aterwards, collect single files to one agglomerated file
- the program checks, if the clusterings for the single thresholds exist, and strips values which are not present

INPUT:
    SingleClustDir : {STRING}    Directory, where the clusterings are stored with single threshold parameters, 
                                    e.g. 'Clustering/';
    SaveDir        : {STRING}    Directory, where to save the merged clusterings with multiple thresholds, 
                                    e.g. 'Clustering/';
    SaveName       : {STRING}    SaveName PREFIX, identical to Generate_Clustering(), e.g. 'Mol2' leads to 'Mol2_R0.1-0.2_0-100_GLOBAL.txt', 
                                     '%s_R%s-%s_%s-%s_%s.txt' % \
                                         (SaveName, ThresholdList[0], ThresholdList[-1], StartFrame, EndingFrame, 'GLOBAL' / 'LOCAL');
    ThresholdList  : {FLOAT-LIST} List of thresholds, which should be merged, e.g. [0.2, 0.25, 0.3, 0.35, 0.4];
    StartFrame     : {INT}       starting frame of Trajectories/RMSD matrices,
                                    to select different simulation times/lengths together with 'EndingFrame';
    EndingFrame    : {INT}       ending frame of Trajectories/RMSD matrices,
                                    to select different simulation times/lengths together with 'StartingFrame';
    GLOBAL         : {BOOL}      if True,  a GLOBAL clustering was applied for all concatenated trajectories
                                 if False, every trajectory is clustered separately;
OUTPUT:
    stores a merged clustering file containing multiple clustering thresholds
    """
    Generate_Directories(SaveDir)
    temp = ThresholdList
    ThresholdList = [elem for elem in ThresholdList if os.path.exists('%s%s_R%s_%s-%s_%s.txt' % \
                      (SingleClustDir, SaveName, elem, StartFrame, EndingFrame, 'GLOBAL' if GLOBAL else 'LOCAL')) and \
                                                 os.path.exists('%s%s_R%s_%s-%s_Centers_%s.txt' % \
                      (SingleClustDir, SaveName, elem, StartFrame, EndingFrame, 'GLOBAL' if GLOBAL else 'LOCAL'))]
    if len(ThresholdList) <= 1:
        raise ValueError('Clusterings with the submitted ThresholdList do not exist! Check, if all following files exist: \n%s\n%s' % \
           ('\n'.join(['%s%s_R%s_%s-%s_%s.txt' % \
                      (SingleClustDir, SaveName, elem, StartFrame, EndingFrame, 'GLOBAL' if GLOBAL else 'LOCAL') for elem in temp]),
            '\n'.join(['%s%s_R%s_%s-%s_Centers_%s.txt' % \
                      (SingleClustDir, SaveName, elem, StartFrame, EndingFrame, 'GLOBAL' if GLOBAL else 'LOCAL') for elem in temp])))
    del temp
    #------
    if not os.path.exists('%s%s_R%s-%s_%s-%s_%s.txt' % \
                          (SaveDir, SaveName, ThresholdList[0], ThresholdList[-1], 
                           StartFrame, EndingFrame, 'GLOBAL' if GLOBAL else 'LOCAL')) or \
       not os.path.exists('%s%s_R%s-%s_%s-%s_Centers_%s.txt' % \
                          (SaveDir, SaveName, ThresholdList[0], ThresholdList[-1], 
                           StartFrame, EndingFrame, 'GLOBAL' if GLOBAL else 'LOCAL')):
      #----- EXTRACT HEADER
        HEADER = ''
        with open('%s%s_R%s_%s-%s_%s.txt' % \
                      (SingleClustDir, SaveName, ThresholdList[0], StartFrame, EndingFrame, 'GLOBAL' if GLOBAL else 'LOCAL'),
                  'r') as INPUT:
            for line in INPUT:
                if line.split()[1] == '*******':
                    HEADER = HEADER + '# *******\n'
                    break
                elif line.split()[1] == 'Threshold':
                    HEADER = HEADER + '# ThresholdList = {}\n'.format(ThresholdList)
                else:
                    HEADER = HEADER + line
        HEADER_Add1 = ''.ljust(18)+'|                   Radius = '+str(ThresholdList[0]).ljust(6)+\
                              '                      |r='+\
                                        '|r='.join([str(elem).ljust(4) for elem in ThresholdList[1:]])+'\n'+\
                              '   Frame | TrajNr | PROF | effRMSD | nextC | nextRMSD | farthC | farthRMSD'+\
                              ' | ... '*len(ThresholdList[1:])+'\n'
      #----- INIT & STORE CENTERS
        with open('%s%s_R%s-%s_%s-%s_Centers_%s.txt' % \
                          (SaveDir, SaveName, ThresholdList[0], ThresholdList[-1], 
                           StartFrame, EndingFrame, 'GLOBAL' if GLOBAL else 'LOCAL'), 'w') as OUTPUT:
            OUTPUT.write(HEADER)
            OUTPUT.write('# TrajNr | Radius | Nr of Clusters | Centers\n')
          #----- 
            for ClustRadius in ThresholdList:
              #------ CLUSTERING  
                if 'temp' in locals():
                    temp = NP.concatenate( (temp, NP.genfromtxt('%s%s_R%s_%s-%s_%s.txt' % \
                          (SingleClustDir, SaveName, ClustRadius, StartFrame, EndingFrame, 'GLOBAL' if GLOBAL else 'LOCAL'),
                                                                usecols=(2,3,4,5,6,7))),
                                          axis=1 )
                else:
                    temp = NP.genfromtxt('%s%s_R%s_%s-%s_%s.txt' % \
                          (SingleClustDir, SaveName, ClustRadius, StartFrame, EndingFrame, 'GLOBAL' if GLOBAL else 'LOCAL'))
              #------ CENTERS
                with open('%s%s_R%s_%s-%s_Centers_%s.txt' % \
                          (SingleClustDir, SaveName, ClustRadius, StartFrame, EndingFrame, 'GLOBAL' if GLOBAL else 'LOCAL'),
                          'r') as INPUT:
                    for line in INPUT:
                        if line.split()[0] != '#':
                            OUTPUT.write(line)
      #----- STORE CLUSTERING
        NP.savetxt('%s%s_R%s-%s_%s-%s_%s.txt' % \
                          (SaveDir, SaveName, ThresholdList[0], ThresholdList[-1], 
                           StartFrame, EndingFrame, 'GLOBAL' if GLOBAL else 'LOCAL'), 
                   temp, fmt='%i %i'+'    %i %.4f  %i %.4f  %i %.4f'*len(ThresholdList),
                   header=HEADER.replace('# ','')+HEADER_Add1)
    else:
        print 'The merged Clustering for ThresholdList = {} already exist'.format(ThresholdList)+\
              '\n\t%s%s_R%s-%s_%s-%s_%s.txt\n\t%s%s_R%s-%s_%s-%s_Centers_%s.txt' % \
            (SaveDir, SaveName, ThresholdList[0], ThresholdList[-1], StartFrame, EndingFrame, 'GLOBAL' if GLOBAL else 'LOCAL',
             SaveDir, SaveName, ThresholdList[0], ThresholdList[-1], StartFrame, EndingFrame, 'GLOBAL' if GLOBAL else 'LOCAL')

###############
#-------------
###############

def Generate_Centers_GLOBAL_singles(ClusterDir, GlobalName, ThresholdList, SaveDir=None):
    """
v07.11.16
- this function generates Centers_GLOBAL_singles.txt containing
    
        TrajNr | Threshold | Nr of Clusters | Centers (1 to NrofClusters)
        
- splitting the GLOBAL (all trajectories are concatenated) clustering into different trajectories and
  extracting which clusters are occupied by which trajectory number, assigning a single trajectory clustering from the global partition
- detecting also the Size (=Nr of clusters)
- this allows to use a GLOBAL clustering and extract, how many clusters are reached by single trajectories

INPUT:
    ClusterDir    : {STRING}     Directory, where effective Clustering output is located, e.g. 'effectiveClustering/';
    GlobalName    : {STRING}     Clustering Name of GLOBAL effective clustering, which stores the clustering profile, where all trajs are concatenated
                                      e.g. 'Cluster_R5_REF_D_S1-S10_R0.2-0.7_GLOBAL.txt';
    ThresholdList : {FLOAT-LIST} Clustering ThresholdList [nm], e.g. [0.2, 0.25, 0.3, 0.35, 0.4];
    SaveDir       : {STRING}     <default None>, save directory, e.g. 'effectiveClustering/',  if None, SaveDir = ClusterDir;
OUTPUT:
    SaveDir+GlobalName.replace('GLOBAL.txt','Centers_GLOBAL_singles.txt') with:
        
        TrajNr | Threshold | Nr of Clusters | Centers
    """
    if SaveDir is None:
        SaveDir = ClusterDir
    Generate_Directories(SaveDir)
    #------
    if os.path.exists(ClusterDir+GlobalName) and not \
       os.path.exists(SaveDir+GlobalName.replace('GLOBAL.txt','Centers_GLOBAL_singles.txt')):
      ## CHECK, if a GLOBAL clustering is submitted
        if GlobalName.find('GLOBAL.txt') == -1:
            raise NameError('Please submit a GlobalName with GLOBAL clustering, with "_GLOBAL.txt"\n\tGlobalName = %s' % \
                               GlobalName)
      ## CHECK, if ThresholdList is the same like submitted
        CorrectThreshold = False
        with open(ClusterDir+GlobalName, 'r') as INPUT:
            for line in INPUT:
                if len(line.split()) > 1 and line.split()[1] == 'ThresholdList':
                    if line.split(' = ')[1] == '{}\n'.format(ThresholdList):
                        CorrectThreshold = True
                        break
                if len(ThresholdList) == 1 and len(line.split()) > 1 and line.split()[1] == 'Threshold':
                    if line.split(' = ')[1] == '{}\n'.format(ThresholdList[0]):
                        CorrectThreshold = True
                        break
        if CorrectThreshold:
            t1 = time.time()
          #---- LOAD GLOBAL CLUSTERING/PROFILE: load Time[ns] | TrajNr | PROF r1 | PROF r2 | PROF r3 | ... 
            GLOBAL_Profile = NP.genfromtxt(ClusterDir+GlobalName, 
                                           usecols=NP.concatenate(([0,1], [2+6*elem for elem in range(len(ThresholdList))]))) 
          #---- extract different Trajectory numbers
            TrajNrList = NP.unique(GLOBAL_Profile[:,1])
          #---- use the same HEADER as for the GLOBAL_Profile
            with open(SaveDir+GlobalName.replace('GLOBAL.txt','Centers_GLOBAL_singles.txt'), 'w') as OUTPUT:
                with open(ClusterDir+GlobalName, 'r') as INPUT:
                    for line in INPUT:
                        if len(line.split())>1 and line.split()[-1].find('*') != -1:
                            OUTPUT.write(line)
                            break
                        else:
                            OUTPUT.write(line)
                OUTPUT.write('# TrajNr | Threshold | Nr of Clusters | Centers\n')
              #---- CALCULATION
                for Rad in ThresholdList:
                    for TrajNr in TrajNrList:
                        Centers = NP.unique(GLOBAL_Profile[GLOBAL_Profile[:,1]==int(TrajNr)][:,ThresholdList.index(Rad)+2])
                        OUTPUT.write(str(int(TrajNr))+' '+str(Rad)+' '+str(-1 if len(Centers) == 0 else len(Centers)))
                        OUTPUT.write('   '+' '.join([str(int(elem)) for elem in Centers])+'\n')
        else:
            print 'the submitted \n\t>>Threshold = {}<< \nis NOT the same as the Clustering-Threshold'.format(ThresholdList)
    else:
        print 'Either \n\t>>%s%s<< \nalready exists or \n\t>>%s%s<< \ndoes NOT exist' % \
                (SaveDir,GlobalName.replace('GLOBAL.txt','Centers_GLOBAL_singles.txt'),ClusterDir,GlobalName)

###############
#-------------
###############
def Generate_CDE_to_File(ClusterDir, ClusterFileName, ThresholdList, Case, SaveDir=None, SaveName=None,
                         WeightDir=None, aMD_Nrs=[], sMD_Nrs=[], aMD_reweight='MF', Iterations=1, Lambda=1, Order=10):
    """
v14.10.16
- this function stores the cluster distribution entropy (CDE) using the idea of [Sawle & Ghosh JCTC 2016]
- the following parameters are stored in SaveDir + SaveName for different ThresholdList
    >> Nr of Cluster vs Time <<
    >> normalized Nr of Cluster vs Time <<
    >> Entropy vs Time <<
    >> normalized Entropy vs Time <<
- using the PROF OUTPUT from effective clustering, LOCAL or GLOBAL
- possibility, to use re-weighted frames for the calculation of p_i: frames -> weights
    - this assumes, that for the next time step, there are frames = (weight x frame) assigned to one cluster, 
      which is an approximation and have to be taken into account
    - it should be correct, not to re-weight aMD/sMD trajectories to just investigate, if the sampling expresses a trapped/undersampled behavior
      with respect on the unterlying potential, the bias does not alter this information
- PROF corresponds to the cluster profile, i.e. reports cluster number/centroid as a function of the frames
- using <Calc_CDE()>
- <GLOBAL_singles> allows to extract from one unique GLOBAL clustering the corresponding clusters for each contained 
  trajectory

INPUT:
    ClusterDir      : {STRING} Directory, where effective Clustering output is located, e.g. 'effectiveClustering/';
    ClusterFileName : {STRING} Clustering Name of effective clustering, contains the PROFILES, e.g. 'Cluster_R5_REF_D_S1-S10_R0.2-0.7_LOCAL.txt';
    ThresholdList   : {FLOAT-LIST}   Clustering ThresholdList [nm], e.g. [0.2, 0.25, 0.3, 0.35, 0.4];
    Case            : {STRING} 'LOCAL' or 'GLOBAL' or 'GLOBAL_singles', needs to correspond to the submitted ClusterFileName !!
                        "LOCAL"          - each trajectory is clustered separately ||
                        "GLOBAL"         - one global clustering for all concatenated trajectories ||
                        "GLOBAL_singles" - each trajectory is taken SEPARATELY, but the clustering was done GLOBAL! ||
                        for GLOBAL_singles, GLOBAL clustering has to be submitted in the ClusterFileName !!;
    SaveDir         : {STRING} Directory, where the CDE file is stored e.g. 'CDE/';
    SaveName        : {STRING} savename PREFIX without Ending, 
                                    e.g. 'CDE_R5_R0.2-0.7' -> 'CDE_R5_R0.2-0.7_%s.txt' % (Case);
    WeightDir       : {STRING}    <default None>, FOR RE-WEIGHTING ONLY, Directory, where the aMD/sMD weights are located, e.g. 'EventCurves/Weights/';
    aMD_Nrs         : {INT-LIST}  <default []>,   FOR RE-WEIGHTING ONLY, trajectory numbers which are generated with aMD,
                                                  numbering MUST correspond to the trajectories stored in ClusterFileName under "TrajNameList = [...]",
                                        e.g. [1,3,5] means trajNr 1,3,5 of TrajNameList are aMD trajectories | has to correspond to possible _ColY EventCurves;
    sMD_Nrs         : {INT-LIST}  <default []>,   FOR RE-WEIGHTING ONLY, trajectory numbers which are generated with scaledMD, 
                                                  numbering MUST correspond to the trajectories stored in ClusterFileName under "TrajNameList = [...]",
                                        e.g. [1,3,5] means trajNr 1,3,5 of TrajNameList are scaledMD trajectories | has to correspond to possible _ColY EventCurves;
    aMD_reweight    : {STRING}    <default MF> FOR RE-WEIGHTING ONLY, reweighting method if aMD trajs are present, default Mean-Field-Approach | 
                                    possibilities - 'MF', 'Exp', 'McL';
    Iterations      : {INT}       <default 1>     FOR RE-WEIGHTING ONLY, the number of MF iterations used if aMD (aMD_reweight=MF) or sMD trajectories are present;
    Lambda          : {FLOAT}     <default 1>     FOR RE-WEIGHTING ONLY, scaling factor for scaledMD, e.g. 0.7, 1 means no scaling;
    Order           : {INT}       <default 10>    FOR RE-WEIGHTING ONLY, Order for the MacLaurin expansion, ONLY NECESSARY IF aMD_reweight = 'McL';
OUTPUT:
    if SaveDir is not None and SaveName is not None:
        SaveDir+SaveName with:
                           |             Threshold = r1                 | r2  | ...
            Frame | TrajNr | NrClusts | normNrClusts | CDE | normCDE | ... | ...
                uses the same threshold as in ClusterFileName
    else:
        return CDE_Array
    """
    
    if os.path.exists(ClusterDir+ClusterFileName) and \
        ((SaveDir==None and SaveName==None) or not os.path.exists('%s%s_%s.txt' % (SaveDir,SaveName,Case))):
      ## generate SaveDir if not None
        if SaveName is not None:
            Generate_Directories(SaveDir)
      ## CHECK, if Case == 'LOCAL', 'GLOBAL' or 'GLOBAL_singles'
        if Case != 'LOCAL' and Case != 'GLOBAL' and Case != 'GLOBAL_singles':
            raise ValueError('Case has to be either "LOCAL" or "GLOBAL" or "GLOBAL_singles"!\n\tsubmitted Case = %s' % \
                                (Case))
      ## CHECK, if Case corresponds to ClusterFileName
        if ClusterFileName.find('%s.txt' % Case.split('_')[0]) == -1:
            raise ValueError('Case = <%s> and ClusterFileName = <%s> have to have corresponding clusterings!' % \
                                (Case, ClusterFileName))
      ## CHECK, if ThresholdList is the same like submitted
        CorrectThreshold = False
        with open(ClusterDir+ClusterFileName, 'r') as INPUT:
            for line in INPUT:
                if len(line.split()) > 1 and line.split()[1] == 'ThresholdList':
                    if line.split(' = ')[1] == '{}\n'.format(ThresholdList):
                        CorrectThreshold = True
             ## EXTRACT TrajNameList for possible re-weighting
                elif len(line.split()) > 1 and line.split()[1] == 'TrajNameList':
                    TrajNameList = [elem.replace('\'','').replace(',','') \
                                    for elem in line[line.find('[')+1:line.find(']')].split()]
             ## EXTRACT StartFrame & EndingFrame for possible re-weighting
                elif len(line.split()) > 1 and line.split()[1] == 'StartFrame':
                    StartFrame = int(line.split()[3])
                elif len(line.split()) > 1 and line.split()[1] == 'EndingFrame':
                    EndingFrame = int(line.split()[3])
                    break
            
        if CorrectThreshold:
            t1 = time.time()
          #----- GENERATE usecols to load only PROF columns from the clustering file: 
          #              Frame | TrajNr | PROF r1 | PROF r2 | ...
            USECOLS = [elem for elem in range(3*len(ThresholdList)+2)];
            for Kai in range(3*len(ThresholdList)-1):
                USECOLS[Kai+3] = USECOLS[Kai+2]+abs(Kai%3-3)
          #----- LOAD Clustering to which CDE is calculated
            PROF = NP.genfromtxt(ClusterDir+ClusterFileName, usecols=USECOLS)
          #----- GENERATE CDE ARRAY: Time[ns] | TrajNr | NrClusts r1 | normNrClusts r1 | CDE r1 | normCDE r1 || NrClustsr2
            CDE_Array = NP.zeros( (len(PROF[:,0]), 2+4*len(ThresholdList)) )
          #----- extract TrajNrList from PROF to calculate the number of clusters and go through every trajectory
            if Case == 'GLOBAL':
                TrajNrList = [0] # for GLOBAL: the whole file is used thus only one loop is necessary
                reweightList = [int(elem) for elem in NP.unique(PROF[:,1])]
            else:
                TrajNrList = [int(elem) for elem in NP.unique(PROF[:,1])]
          #----- CALCULATION: 
            ## INIT Time & TrajNr to immediately detect, which Time belong to which Trajectory
            CDE_Array[:,0] = PROF[:,0]
            CDE_Array[:,1] = PROF[:,1]

            Ranger = [0,0] ## monitors the number of rows and the position of the current trajectory number
            for TrajNr in TrajNrList:
                if Case == 'GLOBAL': ## whole sim-time is used to calculate #clust vs time | entropy vs time
                    Ar = NP.copy(PROF)
                    if sMD_Nrs != [] or (aMD_reweight == 'MF' and aMD_Nrs != []):
                        Weights = NP.ones( (len(Ar[:,0]),len(ThresholdList)) )
                    else:
                        Weights = NP.ones(len(Ar[:,0]))
                else:
                    Ar = NP.copy(PROF[PROF[:,1]==TrajNr])
                    reweightList = [TrajNr]
                    if sMD_Nrs.count(TrajNr) > 0 or (aMD_reweight == 'MF' and aMD_Nrs.count(TrajNr) > 0):
                        Weights = NP.ones( (len(Ar[:,0]),len(ThresholdList)) )
                    else:
                        Weights = NP.ones(len(Ar[:,0]))
        ###################   
        ### RE-WEIGHTING
        ###################
                rewIndex = 0
                for rewTrajNr in reweightList:
                    if aMD_Nrs.count(rewTrajNr) > 0:
                        if aMD_reweight == 'Exp':
                            if not os.path.exists('%saMD_Weight_%s.txt' % (WeightDir, TrajNameList[rewTrajNr-1])):
                                raise NameError(('You try to re-weight trajectory=%s using >Exp<-re-weighting, but\n\t%saMD_Weight_%s.txt\ndoes not exist.' % \
                                                    (TrajNameList[rewTrajNr-1], WeightDir, TrajNameList[rewTrajNr-1]))+\
                                                    'Check aMD_Nrs, WeightDir and aMD_reweight, which have to match\n\tTrajNameList = {}'.format(TrajNameList))
                            Weights[rewIndex:rewIndex+len(Ar[Ar[:,1]==rewTrajNr][:,0])] = \
                                NP.exp(NP.genfromtxt('%saMD_Weight_%s.txt' % (WeightDir, TrajNameList[rewTrajNr-1]), 
                                                     usecols=(0)))
                        elif aMD_reweight == 'McL':
                            if not os.path.exists('%saMD_Weight_%s.txt' % (WeightDir, TrajNameList[rewTrajNr-1])):
                                raise NameError(('You try to re-weight trajectory=%s using >McL<-re-weighting, but\n\t%saMD_Weight_%s.txt\ndoes not exist.' % \
                                                    (TrajNameList[rewTrajNr-1], WeightDir, TrajNameList[rewTrajNr-1]))+\
                                                    'Check aMD_Nrs, WeightDir and aMD_reweight, which have to match\n\tTrajNameList = {}'.format(TrajNameList))
                            tempWeight = NP.genfromtxt('%saMD_Weight_%s.txt' % (WeightDir, TrajNameList[rewTrajNr-1]), 
                                                       usecols=(0))
                            Weights = NP.zeros(len(tempWeight))
                            for Ord in range(0,Order+1):
                                Weights[rewIndex:rewIndex+len(Ar[Ar[:,1]==rewTrajNr][:,0])] = \
                                        NP.add(Weights[rewIndex:rewIndex+len(Ar[Ar[:,1]==rewTrajNr][:,0])], 
                                               NP.divide( NP.power( tempWeight , Ord), float(scipy.misc.factorial(Ord))))
                        elif aMD_reweight == 'MF':
                      #--- extract ThresholdList of sMD_Weights, to match the correct Threshold
                            if not os.path.exists('%saMD_Weight_MF_%s_%s-%s_Iter%s.txt' % \
                                                      (WeightDir, TrajNameList[rewTrajNr-1], StartFrame, EndingFrame, Iterations)):
                                raise NameError(('You try to re-weight trajectory=%s using >MF<, but\n\t%saMD_Weight_MF_%s_%s-%s_Iter%s.txt\ndoes not exist.' % \
                                                    (TrajNameList[rewTrajNr-1], WeightDir, TrajNameList[rewTrajNr-1], StartFrame, EndingFrame, Iterations))+\
                                            'Check aMD_Nrs, WeightDir, Iterations and aMD_reweight, which have to match\n\tTrajNameList = {}'.format(TrajNameList))
                            with open('%saMD_Weight_MF_%s_%s-%s_Iter%s.txt' % \
                                      (WeightDir, TrajNameList[rewTrajNr-1], StartFrame, EndingFrame, Iterations), 'r') as aMDin:
                                for line in aMDin:
                                    if len(line.split()) > 2 and line.split()[0] == '#' and line.split()[1] == 'ThresholdList':
                                        aMD_ThresholdList = [float(elem.replace(',','')) \
                                                             for elem in line[line.find('[')+1:line.find(']')].split()]
                                        break
                            if 'aMD_ThresholdList' not in locals():
                                raise ValueError('ThresholdList is not defined in\n\t%saMD_Weight_MF_%s_%s-%s_Iter%s.txt\nAdd the used >>ThresholdList = [...]<<.' % \
                                                    (WeightDir, TrajNameList[rewTrajNr-1], StartFrame, EndingFrame, Iterations))
                            else:
                                for TT in ThresholdList:
                                    if aMD_ThresholdList.count(TT) == 0:
                                        raise ValueError('There are no weights calculated for\n\tThreshold=%s\nin\n\t%saMD_Weight_MF_%s_%s-%s_Iter%s.txt!' % \
                                                            (TT, WeightDir, TrajNameList[rewTrajNr-1], StartFrame, EndingFrame, Iterations))
                            Indo = 0
                            for rewThresh in ThresholdList:
                                Weights[rewIndex:rewIndex+len(Ar[Ar[:,1]==rewTrajNr][:,0]), Indo] = \
                                    NP.exp(NP.genfromtxt('%saMD_Weight_MF_%s_%s-%s_Iter%s.txt' % \
                                          (WeightDir, TrajNameList[rewTrajNr-1], StartFrame, EndingFrame, Iterations), 
                                                         usecols=(aMD_ThresholdList.index(rewThresh))))
                                Indo += 1
                    elif sMD_Nrs.count(rewTrajNr) > 0:
                      #--- extract ThresholdList of sMD_Weights, to match the correct Threshold
                        if not os.path.exists('%ssMD_Weight_lambda%s_%s_%s-%s_Iter%s.txt' % \
                          (WeightDir, Lambda, TrajNameList[rewTrajNr-1], StartFrame, EndingFrame, Iterations)):
                            raise NameError(('You try to re-weight trajectory=%s using >sMD<, but\n\t%ssMD_Weight_lambda%s_%s_%s-%s_Iter%s.txt\ndoes not exist.' % \
                                                (TrajNameList[rewTrajNr-1], WeightDir, Lambda, TrajNameList[rewTrajNr-1], StartFrame, EndingFrame, Iterations))+\
                                    'Check sMD_Nrs, WeightDir, Iterations, Lambda and aMD_reweight, which have to match\n\tTrajNameList = {}'.format(TrajNameList))
                        with open('%ssMD_Weight_lambda%s_%s_%s-%s_Iter%s.txt' % \
                          (WeightDir, Lambda, TrajNameList[rewTrajNr-1], StartFrame, EndingFrame, Iterations), 'r') as sMDin:
                            for line in sMDin:
                                if len(line.split()) > 2 and line.split()[0] == '#' and line.split()[1] == 'ThresholdList':
                                    sMD_ThresholdList = [float(elem.replace(',','')) \
                                                         for elem in line[line.find('[')+1:line.find(']')].split()]
                                    break
                        
                        if 'sMD_ThresholdList' not in locals():
                            raise ValueError('ThresholdList is not defined in\n\t%ssMD_Weight_lambda%s_%s_%s-%s_Iter%s.txt\nAdd used >>ThresholdList = [...]<<.' % \
                                                (WeightDir, Lambda, TrajNameList[rewTrajNr-1], StartFrame, EndingFrame, Iterations))
                        else:
                            for TT in ThresholdList:
                                if sMD_ThresholdList.count(TT) == 0:
                                    raise ValueError('There are no weights calculated for\n\tThreshold=%s\nin\n\t%ssMD_Weight_lambda%s_%s_%s-%s_Iter%s.txt!' % \
                                                        (TT, WeightDir, Lambda, TrajNameList[rewTrajNr-1], StartFrame, EndingFrame, Iterations))
                        Indo = 0
                        for rewThresh in ThresholdList:
                            if sMD_ThresholdList.count(rewThresh) == 0:
                                raise ValueError('You want to re-weight CDE, but the corresponding\n\tweight=%s\nwith\n\tThreshold=%s is not present! Check your input!' % \
                                                 ('%ssMD_Weight_lambda%s_%s_%s-%s_Iter%s.txt' % \
                          (WeightDir, Lambda, TrajNameList[TrajNr-1], StartFrame, EndingFrame, Iterations), Threshold))
                            Weights[rewIndex:rewIndex+len(Ar[Ar[:,1]==rewTrajNr][:,0]), Indo] = \
                                NP.genfromtxt('%ssMD_Weight_lambda%s_%s_%s-%s_Iter%s.txt' % \
                          (WeightDir, Lambda, TrajNameList[rewTrajNr-1], StartFrame, EndingFrame, Iterations), 
                                             usecols=(sMD_ThresholdList.index(rewThresh)))
                    rewIndex += len(Ar[Ar[:,1]==rewTrajNr][:,0])
        ####################
              ## extract correct parts to store the actual CDE to CDE_Array array
                Ranger[0] = Ranger[1]
                Ranger[1] = Ranger[1]+len(Ar[:,0])
              ## calculate #CLUST/ENTROPY vs TIME
                ## temp: Time[ns] | TrajNr | NrClusts | normNrClusts | CDE | normCDE
                for Threshold in ThresholdList:
                    if len(Weights.shape) > 1: #(aMD_Nrs.count(TrajNr) > 0 and aMD_reweight != 'MF') or sMD_Nrs.count(TrajNr) > 0:
                        temp = Calc_CDE(Ar[:,[0,1,2+3*ThresholdList.index(Threshold)]],
                                        Weights[:,    ThresholdList.index(Threshold)])
                    else:
                        temp = Calc_CDE(Ar[:,[0,1,2+3*ThresholdList.index(Threshold)]], Weights) 
                    CDE_Array[Ranger[0]:Ranger[1],2+4*ThresholdList.index(Threshold)] = temp[:,1]
                    CDE_Array[Ranger[0]:Ranger[1],3+4*ThresholdList.index(Threshold)] = temp[:,2]
                    CDE_Array[Ranger[0]:Ranger[1],4+4*ThresholdList.index(Threshold)] = temp[:,3]
                    CDE_Array[Ranger[0]:Ranger[1],5+4*ThresholdList.index(Threshold)] = temp[:,4]
          ##-------------- 
            t2 = time.time()
          #### SAVE TO FILE
            HEAD = 'This File stores the CDE (Cluster Distribution Entropy) using the clustering >>ClosestCenters<<'+\
                   ' of the given TrajNameList\n'+\
                   '>> Nr of Cluster vs Time <<\n'+\
                   '>> normalized Nr of Cluster vs Time <<\n'+\
                   '>> Entropy vs Time <<\n'+\
                   '>> normalized Entropy vs Time <<\n'
            with open(ClusterDir+ClusterFileName, 'r') as INPUT:
                Index = 0
                for line in INPUT:
                    if line.split()[-1] == '*******':
                        break
                    elif line.split()[-1] == 'seconds':
                        HEAD = HEAD + 'elapsed time = %s seconds\n' % round(t2-t1,2)
                    elif Index != 0:
                        HEAD = HEAD + line.replace('#','')

                    Index += 1
            HEAD = HEAD + ''.ljust(18)+'|             Threshold = '+str(ThresholdList[0]).ljust(6)+'             |r='+\
                                                    '|r='.join([str(elem).ljust(4) for elem in ThresholdList[1:]])+'\n'+\
                                          '   Frame | TrajNr | NrClusts | normNrClusts | CDE | normCDE'+\
                                          ' | ... '*len(ThresholdList[1:])+'\n'
          ####
            if SaveDir is not None and SaveName is not None:
                NP.savetxt('%s%s_%s.txt' % (SaveDir,SaveName,Case), CDE_Array, header=HEAD, 
                           fmt='%i %i'+'   %i %.2f %.4f %.4f'*len(ThresholdList))
            else:
                return CDE_Array
        else:
            print 'the submitted \n\t>>Threshold = {}<< \nis NOT the same as the Clustering-Threshold'.format(ThresholdList)
    else:
        print 'Either \n\t>>%s%s_%s.txt<< \nalready exists or \n\t>>%s%s<< \ndoes NOT exist' % \
                (SaveDir,SaveName,Case,ClusterDir,ClusterFileName)

###############
#-------------
###############
def Calc_CDE(Array, Weights):
    """
v14.10.16
- this function calculates the 
    >> Nr of Cluster vs Time <<
    >> normalized Nr of Cluster vs Time <<
    >> Entropy vs Time <<
    >> normalized Entropy vs Time <<
- PROF corresponds to the cluster profile, i.e. reports cluster number as a function of the frames

INPUT:
    Array   :  {NP.NDARRAY} containing 3 columns, SimTime | TrajNr | PROF;
    Weights :  {NP.NDARRAY} containing 1 column,  Weights, whereas len(Array[:,0]) == len(Weights)
OUTPUT:
    RETURNING:  CDEarray[:,0] = Simulation Time
                CDEarray[:,1] = Nr of Cluster     = NP.sum( clust[i] )
                CDEarray[:,2] = Nr of Cluster [%] = NP.sum( clust[i] )/len(clust) (in % [0,1])
                CDEarray[:,3] = -sum_i[p_i*log(p_i)] = NP.sum( clust[i]/NP.sum(clust[i]) * NP.log(clust[i]/NP.sum(clust[i])) )
                CDEarray[:,4] = -sum_i[p_i*log(p_i)]/log(N) = CDEarray[:,3]/NP.log( NP.sum(clust[i]) )

                use only the actual number of clusters for normalization

return CDEarray -  NP.ndarray containing 3 columns: SimTime | nr of clusters | CDE
    """
  ## CHECK ARRAY & WEIGHT LENGTH
    if len(Weights) != len(Array[:,0]):
        raise ValueError('The number of frames used for the clustering does not correspond to the Weights! Check your input and simulation times!')
  ## generate Centers
    Centers = [elem for elem in NP.unique(Array[:,2])]
    TotalNrClusts = len(Centers)
    NrOfClusts = 0
  ## generate dictionary to assign immediately the Center to the correct index
    Dict = {}
    Index = 0
    for Keys in Centers:
        Dict[Keys] = Index
        Index += 1
  ## define array which counts the number of frames within the different clusters
    CenterArray = NP.zeros( (2, len(Centers)) ) # 1.row = 0 or 1 | 2.row = number of frames
  ## define RETURNING array with SimTime | nr of clusters [%] | nr of clusters | CDE | normalized CDE
    CDEarray = NP.zeros( (len(Array[:,0]),5) )
    Index = 0
    for Prof in Array[:,2]:
        CenterArray[0,Dict[Prof]] = 1
        CenterArray[1,Dict[Prof]] += Weights[Index]
                    
        CDEarray[Index,0] = Array[Index,0] # SimTime
        if NrOfClusts != TotalNrClusts:
            NrOfClusts = NP.sum(CenterArray[0,:])
      ## Nr of Clusters = sum_i ( clust[i] )
        CDEarray[Index,1] = NrOfClusts
      ## [NORMALIZED] Nr of Clusters [%] = sum_i ( clust[i] ) / len(cluster)
        CDEarray[Index,2] = NrOfClusts/TotalNrClusts # Nr of Cluster [%]
      ## p_i = clust[i]/sum_i(clust[i]) -> sum_i( clust[i]/sum_i(clust[i]) * Log( clust[i]/sum_i(clust[i]) ) )
        ## NP.sum(CenterArray[1,:]) = Nr of Frames = Index+1
        CDEarray[Index,3] = NP.sum( \
              [-( NP.float(elem)/NP.float(NP.sum(CenterArray[1,:])) * NP.log( elem/NP.float(NP.sum(CenterArray[1,:])) ) \
                               if elem != 0 else 0) \
              for elem in CenterArray[1,:]] \
                                  )
      ## CDEarray[:,4] = CDEarray[:,3] / log(N) => CDEarray[Index,3] / NP.log(CDEarray[Index,1])
        CDEarray[Index,4] = (0 if NrOfClusts == 1 else CDEarray[Index,3] / NP.log(NrOfClusts))
        Index += 1
    return CDEarray

###############
#-------------
###############
#################################################################################################
def Generate_Slope_Error(EntropyDir, EntropyName, SaveDir=None, SaveName=None,
                         SlopeTimeArray=[100,250,500], X_NormFactor=1000):
    """
v12.10.16
- this function calculates & stores the 
        >> Entropy Slope <<
        >> Entropy Error = standard error of the slope estimate <<
  for the LAST cluster | LARGEST cluster | different SlopeTimeArray
- using the cluster distribution entropy vs time & nr of clusters vs time
- storing in SaveDir + SaveName for different ThresholdList
- SlopeTimeArray [must contain 3 values] defines the number of FRAMES which are used for the LINEAR REGRESSION
- automatically extracts the <Clustering Case> and <ThresholdList> from the submitted <EntropyName>-file
- X_NormFactor defines, how many frames correspond to X-value increase of one, i.e.
    - X_NormFactor=1000 means the X-value increases in 1000 steps by 1
    - if 1000 steps mean 100ns, then over 100ns the Entropy is raised by the corresponding slope
    - thus the slopes are normalized by X_NormFactor

>>>  X-axis USEs 1/X_NormFactor to normalize the x-axis, because there are usually hundreds or thousands of frames

INPUT:
    EntropyDir      : {STRING}   Directory, where the EntropyFile from Generate_CDE_to_File() is located, 
                                    e.g. ClusterProfile/;
    EntropyName     : {STRING}   "Entropy Name" from Generate_CDE_to_File() 
                                    e.g. CDE_R5_cMD+aMD+sMD_REF_D_S1-S10_R0.2-0.7_01_LOCAL.txt;
    SaveDir         : {STRING}   save directory, e.g. 'Amber14Trajs/ClusterProfile/';
    SaveName        : {STRING}   savename PREFIX without Ending, 
                                    e.g. 'Slope_R5_R0.2-0.7' -> 'Slope_R5_R0.2-0.7_%s.txt' % (Case);
    SlopeTimeArray  : {INT-LIST} <default [100,250,500]> in FRAMES, to calculate SLOPES of the last 
                                     100/250/500 FRAMES of the trajectory;
    X_NormFactor    : {INT}      <default 1000>, X-Array normalization, 
                                        e.g. 1000 means, X-value increases in 1000 steps by 1
                                        if   1000 frames = 100ns, i.e. Entropy/Clusters increases by the slope over 100ns;
OUTPUT:
    if SaveDir is not None and SaveName is not None:
        SaveDir+SaveName with:
       >> ES = Entropy Slope | EE = Entropy Error | CS = Cluster Slope | CE = Cluster Error <<
       >> #Fr= number of frames of last cluster   | #tot = total number of clusters  
        
         |                                 Threshold = r1                                                            | r2 
         |      clustsize=1     | largest cluster plateau         | last Xns     | last Yns     | last Zns     | ...
  TrajNr | ES | EE | #Fr | #tot | ES | EE | #Frames | cluster [%] | ES|EE | CS|CE | ES|EE | CS|CE | ES|EE | CS|CE | ...
  
    else: 
        return Slope_Array
    """
  ## generate SaveDir if not None
    if SaveName is not None:
        Generate_Directories(SaveDir)
  ## DEFINE Case from the EntropyName:
    if EntropyName.find('LOCAL.txt')          != -1: Case = 'LOCAL'
    if EntropyName.find('GLOBAL.txt')         != -1: Case = 'GLOBAL'
    if EntropyName.find('GLOBAL_singles.txt') != -1: Case = 'GLOBAL_singles'
  #######    
    if os.path.exists(EntropyDir+EntropyName) and \
        ((SaveDir==None and SaveName==None) or not os.path.exists('%s%s_%s.txt' % (SaveDir, SaveName, Case))):
      ## LOAD ThresholdList from EntropyName
        CorrectThreshold = False
        with open(EntropyDir+EntropyName, 'r') as INPUT:
            for line in INPUT:
                if len(line.split()) > 1 and line.split()[1] == 'ThresholdList':
                    ThresholdList = [NP.float(elem) for elem in line.split(' = [')[1].replace(']\n','').split(',')]
                    break            
        t1 = time.time()
      #### load Entropy Array: Time[ns] | TrajNr | NrClusts | normNrClusts | CCE | normCCE | ...
        CCE_Array = NP.genfromtxt(EntropyDir+EntropyName)
      #### generate TrajNrList which fulfill Begin|End
        if Case == 'GLOBAL':
            TrajNrList = [0] # for GLOBAL: the whole file is used thus only one loop is necessary
        else:
            TrajNrList = NP.unique(CCE_Array[:,1])
      #### generate SlopeArray which stores the slopes and errors  
        SlopeArray = NP.zeros( (len(TrajNrList), 1+(4+4+4+4+4)*len(ThresholdList)) )
        SlopeArray[:,0] = TrajNrList
        RowIndex = 0
      ## go through every TrajNr
        for TrajNr in TrajNrList:
            if Case == 'GLOBAL': ## whole sim-time is used to calculate #clust vs time | entropy vs time
                Array = NP.copy(CCE_Array)
            else:
                Array = NP.copy(CCE_Array[CCE_Array[:,1]==TrajNr])
      ## go through every threshold
            ColIndex = 1
            for Rad in ThresholdList:
              #### generate Slope & Error for 5 different cases: 
               ## (i) clustsize == 1 | (ii) largest cluster plateau | (iii) last 10ns | (iv) last 25ns | (v) last 50ns
                for Case2 in range(5):
                  ### 5 different cases to generate Y-array
                    if Case2 == 0:           ## (i) clustsize == 1
                        Y1 = Array[Array[:,3+4*ThresholdList.index(Rad)] == 1][:,4+4*ThresholdList.index(Rad)]
                    elif Case2 == 1:         ## (ii) largest cluster plateau
                        Unique, Counts = NP.unique(Array[:,2+4*ThresholdList.index(Rad)], return_counts=True)
                        largestCluster = Unique[NP.argsort(Counts)[-1]]
                        Y1 = Array[Array[:,2+4*ThresholdList.index(Rad)] == largestCluster][:,4+4*ThresholdList.index(Rad)]
                    else: # Case2 == (iii), (iv), (v):
                        Y1 = Array[int(-SlopeTimeArray[Case2-2]):,4+4*ThresholdList.index(Rad)] # Entropy vs Time
                        Y2 = Array[int(-SlopeTimeArray[Case2-2]):,2+4*ThresholdList.index(Rad)] # Cluster vs Time
                  ### X-array USE 1/1000 to normalize ~1000 frames & ~1 entropy
                    X = NP.arange(0,len(Y1)*(1./X_NormFactor) - (1./X_NormFactor/2), (1./X_NormFactor))
                    #X = NP.arange(0,len(Y1)*0.001-0.0005,0.001) # range(len(Y))
                  ### generate SLOPE & ERROR
                    if len(X) > 3:
                        EntropySlope, t1, t2, t3, EntropyError = LR(X, Y1)
                        if Case2 > 1: ClusterSlope, t1, t2, t3, ClusterError = LR(X, Y2)
                    else:
                        EntropySlope, EntropyError = (100,0)
                        if Case2 > 1: ClusterSlope, ClusterError = (100,0)
                  ### STORE every values into SlopeArray[:,:]
                    if Case2 == 0:
                        SlopeArray[RowIndex, [ColIndex,ColIndex+1, ColIndex+2, ColIndex+3]] = \
                                            [EntropySlope, EntropyError, len(X), Array[-1,2+4*ThresholdList.index(Rad)]]
                        ColIndex += 4
                    elif Case2 == 1:
                        SlopeArray[RowIndex, [ColIndex,ColIndex+1, ColIndex+2, ColIndex+3]] = \
                                                [EntropySlope, EntropyError, len(X), largestCluster/float(Unique[-1])]
                        ColIndex += 4
                    else:
                        SlopeArray[RowIndex, [ColIndex,ColIndex+1, ColIndex+2, ColIndex+3]] = \
                                                [EntropySlope, EntropyError, ClusterSlope, ClusterError]
                        ColIndex += 4
          ## go to next TrajNr
            RowIndex += 1 
      ## store SlopeArray to File
        t2=time.time()
        HEADER = 'this file stores the SLOPE & standard ERROR (entropy & cluster) for different cases\n'+\
                     '\t(i) clustsize == 1\n'+\
                     '\t(ii) largest cluster plateau\n'+\
                     '\t(iii) last '+str(SlopeTimeArray[0])+' frames\n'+\
                     '\t(iv) last '+str(SlopeTimeArray[1])+' frames\n'+\
                     '\t(v) last '+str(SlopeTimeArray[2])+' frames\n'+\
                 'elapsed time = {} seconds\n'.format(round(t2-t1,2))+\
                 'SlopeTimeArray = {}\n'.format(SlopeTimeArray)
        with open(EntropyDir+EntropyName, 'r') as Input:
                for line in Input:
                    Ar = line.split()
                    if Ar[1] == 'EndingFrame':
                        HEADER = HEADER + line.replace('#','')
                        break
                    elif Ar[1] != 'This' and Ar[1] != 'elapsed' and Ar[1] != 'ThresholdList' and Ar[1] != '>>':
                        HEADER = HEADER + line.replace('#','')
        HEADER = HEADER + 'ThresholdList = {}'.format(ThresholdList)+\
    """
       >> ES = Entropy Slope | EE = Entropy Error | CS = Cluster Slope | CE = Cluster Error <<
       >> #Fr= number of frames of last cluster   | #tot = total number of clusters  
        
         |                                 Threshold = r1                                                         | r2 
         |      clustsize=1     | largest cluster plateau         | last %s last %s last %s ...
  TrajNr | ES | EE | #Fr | #tot | ES | EE | #Frames | cluster [%%] | ES|EE | CS|CE | ES|EE | CS|CE | ES|EE | CS|CE | ...
    """ % (('%s' % SlopeTimeArray[0]).ljust(9)+'|', ('%s' % SlopeTimeArray[1]).ljust(9)+'|', ('%s' % SlopeTimeArray[2]).ljust(9)+'|')
        if SaveDir is not None and SaveName is not None:
            NP.savetxt('%s%s_%s.txt' % (SaveDir, SaveName, Case), SlopeArray, header=HEADER, 
                   fmt='%i'+('   %.3f %.3f %i %i   %.3f %.3f %i %.2f   %.3f %.3f %.3f %.3f'+\
                             '   %.3f %.3f %.3f %.3f   %.3f %.3f %.3f %.3f')*len(ThresholdList))
        else:
            return SlopeArray
    else:
        print 'Either \n\t>>%s%s_%s.txt<< \nalready exists or \n\t>>%s%s<< \ndoes NOT exist' % \
                (SaveDir,SaveName,Case,  EntropyDir,EntropyName)
#######################################################
#---                PLOTTING
#######################################################
#--- ClusterProfile as a function of the simulation time
#################
def Plot_ClusterProfile(ClusterDir, ClusterName, TimeStep, Threshold, TrjLenList, Global,
                        SaveDir=None, SavePDF=None, Names=[], FigSize=[16,8]):
    """
v12.10.16
    - This function plots a clustering profile as a function of the simulation time [ns]
    - May produce huge plots depending on the number of clusters and number of involved frames, try to play with
        FigSize to adjust the visibility and figure size
    - white crosses marks the time corresponding to the cluster center
    - cluster numbers are ordered in ascending order, when they are visited during the course of the simulation
    - for a global clustering, multiple trajectories are separated by vertical lines defined by TrjLenList [ns]
    - frames from the clustering output must be changed to simulation time in [ns] by multiplicating with TimeStep in [ns]

INPUT:
    ClusterDir  : {STRING}     Directory, where effective Clustering output is stored, e.g. 'effectiveClustering/';
    ClusterName : {STRING}     Clustering Name of effective clustering containing the profiles, 
                                    e.g. 'Cluster_R5_REF_D_S1-S10_R0.2-0.7_LOCAL.txt';
    TimeStep    : {FLOAT}      time-step, which represents one frame-step, e.g. '0.1' [ns] means, 
                                one frame corresponds to 0.1ns, the frames are multiplied by this value to obtain the time;
    Threshold   : {FLOAT}      Threshold used for the clustering, MUST MATCH the thresholds in the clusterfile, e.g. '0.15' [nm];
    TrjLenList  : {FLOAT-LIST} length of the (involved) trajectories in [ns], e.g. [100, 50];
    Global      : {BOOL}       if True,  it is assumed that the clustering was done globally,
                               if False, every trajectory is handled separately;
    SaveDir     : {STRING}     Directory, where the figure is stored e.g. 'Clustering/';
    SavePDF     : {STRING}     Savename for the PDF, e.g. 'Profile_Clustering+Specifications.pdf';
    Names       : {LIST}       <default []>      a second y-axis numberes different trajectories, whereas also names for
                                                 different trajectories can be submitted, e.g. ['traj A', 'traj B'];
    FigSize     : {INT-LIST}   <default [16,8]>  size of the figure (in inches), try to adjust this array depending on 
                                                 the number of clusters and frames;
OUTPUT:
    profile stored in a PDF file
    """
    import matplotlib.pyplot as plt
    ### INIT ThresholdList       ###
    with open('%s%s' % (ClusterDir, ClusterName), 'r') as INPUT:
        for line in INPUT:
            if len(line.split()) > 1 and line.split()[1] == 'ThresholdList':
                ThresholdList = [NP.float(elem) for elem in line.split(' = [')[1].replace(']\n','').split(',')]
                break
            elif len(line.split()) > 1 and line.split()[1] == 'Threshold':
                ThresholdList = [NP.float(line.split(' = ')[1])]
                break
    ### LOAD CLUSTERING PROFILES ###
    Array = NP.genfromtxt('%s%s' % (ClusterDir, ClusterName), usecols=(0,1,2+6*ThresholdList.index(Threshold)))
    Array[:,0] = Array[:,0]*TimeStep
    ################################
    
    COLORS = ['r','b','g','k','m','c']
    
  ########  
    if Global: ## GLOBAL CLUSTERING IS ASSUMED
        Repeater = 1
    else:
        Repeater = len(NP.unique(Array[:,1]))
        TrajNrList = [int(elem) for elem in NP.unique(Array[:,1])]
    Index = 0
    while Repeater > 0:
        fig = plt.figure(figsize=FigSize)
      #----  
        ColInd = 0; Minus = 0
      #----
        if Global:
            ARR = Array
            plt.xlim([0,NP.sum(TrjLenList)]); 
        else:
            TrajNr = TrajNrList.pop(0)
            ARR = Array[Array[:,1] == TrajNr]
            Minus = -ARR[0,0]
            plt.xlim([0,TrjLenList[Index]]); 
            Index += 1
            
        UNIQUE, INDEX, COUNTS = NP.unique(ARR[:,2], return_index=True, return_counts=True)
        INDEX = [int(elem) for elem in NP.argsort(INDEX)]
        for Uni in UNIQUE[INDEX]:
            temp = ARR[ARR[:,2]==Uni]
            plt.plot(NP.add(temp[:,0],Minus),[ColInd]*len(temp[:,0]), '%s-' % COLORS[ColInd%6])
            plt.plot(NP.add(temp[:,0],Minus),[ColInd]*len(temp[:,0]), '%ss' % COLORS[ColInd%6])
            ColInd += 1
      ### ### ### ###
        plt.yticks(range(len(NP.unique(ARR[:,2]))), [int(elem) for elem in UNIQUE[INDEX]], fontsize=15);
        plt.xlabel('simulation time [ns]', fontsize=22); plt.xticks(fontsize=16)
        plt.ylabel('cluster/frame number', fontsize=22)
        plt.ylim([-1,len(NP.unique(ARR[:,2]))])
      ## MARK cluster centers - if GLOBAL is submitted, the TrajNr Array might not contain the correct indices##
        try:
            plt.plot(ARR[[int(elem-1) for elem in UNIQUE[INDEX]],0], range(len(NP.unique(ARR[:,2]))), 'wx', mew=1.5)
        except IndexError:
            pass
      ## vertical lines separating trajNr  
        if Global:
            LenList = [sum(TrjLenList[:elem]) for elem in range(len(TrjLenList))]
            for IL in LenList[1:]:
                plt.axvline(IL, lw=2, ls='-.')
      ## LEGEND
        leg = plt.legend(['%s clusters' % len(UNIQUE[INDEX])], handlelength=0, handletextpad=0, markerscale=0, 
                             loc=2, fontsize=18, framealpha=0.7); 
        for item in leg.legendHandles:
            item.set_visible(False)
      ## second y-axis ##
        ax2 = plt.twinx(); ax2.set_ylim([-1,len(NP.unique(ARR[:,2]))]); ax2.set_yticks(range(len(NP.unique(ARR[:,2])))) 
        ax2.set_yticklabels([int(elem) for elem in COUNTS[INDEX]], fontsize=15)
        ax2.set_ylabel('cluster size', fontsize=22)
      ## second x-axis ##
        if Global:
            NewXticks = [sum(TrjLenList[:elem])+TrjLenList[elem]/2 for elem in range(len(TrjLenList))]
            ax3 = plt.twiny(); ax3.set_xlim([0,NP.sum(TrjLenList)]); ax3.set_xticks(NewXticks)
            if len(Names) == 0:
                ax3.set_xticklabels([int(elem) for elem in NP.unique(Array[:,1])], fontsize=18); 
                ax3.set_xlabel('TrajNr', fontsize=22);
            else:
                ax3.set_xticklabels(Names, fontsize=18)
        else:
            NewXticks = [TrjLenList[Index-1]/2]
            ax3 = plt.twiny(); ax3.set_xlim([0,TrjLenList[Index-1]]); ax3.set_xticks(NewXticks);
            if len(Names) == 0:
                ax3.set_xticklabels([TrajNr], fontsize=18); ax3.set_xlabel('TrajNr', fontsize=22)
            else:
                ax3.set_xticklabels([Names[Index-1]], fontsize=18)      
      ## STORING ##
        if SaveDir is not None and SavePDF is not None:
            Generate_Directories(SaveDir)
            if not Global:
                plt.savefig('%s%s' % (SaveDir, SavePDF.replace('.pdf', '_Traj%s' % TrajNr).replace('.png', '_Traj%s' % TrajNr)))
            else:
                plt.savefig('%s%s' % (SaveDir, SavePDF))
            plt.close()
      ##-------##  
        Repeater -= 1

#######################################################
#--- SLOPE ERROR | PLATEAUS | Nr of Clusters
#################
def Plot_Slope_Error_Plateau_NrClust(SlopeDir, SlopeName, Threshold, Case, TimeStep, SaveDir=None, Confidence=0.95, 
                                     YMAX=50, Splitter=None, SupGrid=None, TrajExcept=[], FigText=None):
    """
v28.10.16
    This function plots/analyzes ALL information stored in Slopes.txt of
            >> Generate_Slope_Error() <<

    LAST cluster = 100% of all (found) clusters are reached
INPUT:
    SlopeDir        : {STRING}    Directory, where Slopes of 'Generate_Slope_Error()' are located, e.g. 'ClusterProfile/';
    SlopeName       : {STRING}    name of the Slopes.txt file of 'Generate_Slope_Error()',
                                    e.g. 'Slopes_R5_cMD+aMD+sMD_REF_D_S1-S10_R0.2-0.7_End_25_01_LOCAL.txt';
    Threshold       : {FLOAT}     Clustering-Threshold [nm], e.g. 0.2;
    Case            : {STRING}    'Entropy' OR 'Cluster' OR 'Plateau' OR 'NrClust', 
                                    plotting SLOPES (entropy/NrClust) OR Length of LAST cluster OR Nr of Clusters;
    TimeStep        : {FLOAT}     defines the step, 1 frame refers to TimeStep [ns] of the trajectory, 
                                    e.g. TimeStep = 0.01 means, 1 Frame = 10ps;
    SaveDir         : {STRING}    Save directory, e.g. 'ClusterProfile/';
    Confidence      : {FLOAT}     <default 0.95>, defines the confidence interval of the slopes of the linear regression;
    YMAX            : {INT}       <default 50>  , ylim([-1,YMAX]) for Case != 'Entropy';
    Splitter        : {LIST}      <default None>, defines, how many lines of SlopeName are plotted in one subplot, 
                    e.g. [(0,10), (10,20)], plotting line 0-10 | 10-20 in two subplots, extract different trajectories;
    SupGrid         : {INT-LIST}  <default None>, defines the subplotgrid, (#Rows, #Cols) = SupGrid
                                    e.g. (2,3) for 2 rows and 3 cols, ! len(Splitter) == SupGrid[0]*SupGrid[1] !;
    TrajExcept      : {INT-LIST}  <default []> defines the TrajNr's which are NOT used from the SlopeName, 
                                    e.g. [1] means the entry with TrajNr = 1 is not considered;
    FigText         : {LIST}      <default None> the inside figtext of the subplots, None = a) b) c) ... 
                                    ! len(FigText) == number of subplots !;
OUTPUT:
    Case = Entropy:
        plot the slopes of the cluster ENTROPY for LAST cluster | LAST X/Y/Z ns for all trajectory numbers
    Case = Cluster:
        plot the slopes of the #CLUSTER for LAST X/Y/Z ns for all trajectory numbers
    Case = NrClust:
        plot the NUMBER of CLUSTER for all trajectory numbers
    Case = Plateau:
        plot the LENGTH [ns] if 100% of the cluster are reached

    
extracts >ThresholdList & SlopeTimeArray out of the SlopeName

only stored if SaveDir/SaveName is not None else:
        plt.show()
    
    ### X-array USE 1/1000 to normalize ~1000 frames & ~1 entropy
    """
    import matplotlib.pyplot as plt
    C = ['m','c','r','g','b']
  ### load total number of clusters
    if os.path.exists(SlopeDir+SlopeName) and \
       (SaveDir==None or not os.path.exists('%s%s' % (Case, SlopeName.replace('.txt','.pdf')))):
        CorrectSlopeTime = False; CorrectThreshold = False;
      #---- Check if Splitter has the same dimension as SupGrid
        if Splitter is not None and SupGrid is not None and SupGrid[0]*SupGrid[1] != len(Splitter):
            Splitter = None; SupGrid = None;
      #---- LOAD ThresholdList & SlopeTimeArray from SlopeName
        with open(SlopeDir+SlopeName, 'r') as INPUT:
            for line in INPUT:
                if len(line.split()) > 1 and line.split()[1] == 'SlopeTimeArray':
                    SlopeTimeArray = [int(elem) for elem in line.split(' = [')[1].replace(']\n','').split(',')]
                if len(line.split()) > 1 and line.split()[1] == 'ThresholdList':
                    ThresholdList = [NP.float(elem) for elem in line.split(' = [')[1].replace(']\n','').split(',')]
                    break
        if ThresholdList.count(Threshold) > 0:
            t1 = time.time()
            
          #### load Entropy Array: Time[ns] | TrajNr | NrClusts | normNrClusts | CCE | normCCE | ...
           #----- define correct USECOLS for submitted Case
            if Case == 'Entropy': # LAST cluster | SlopeTimeArray[1-3]
                USECOLS = NP.concatenate(([0,3],[elem+20*ThresholdList.index(Threshold) for elem in \
                                              [9,10, 13,14, 17,18, 1,2]]))
            elif Case == 'Cluster': # SlopeTimeArray[1-3]
                USECOLS = NP.concatenate(([0,3],[elem+20*ThresholdList.index(Threshold) for elem in \
                                              [11,12, 15,16, 19,20]]))
            elif Case == 'Plateau': # LAST cluster | LARGEST cluster
                USECOLS = NP.concatenate(([0],[elem+20*ThresholdList.index(Threshold) for elem in \
                                              [3, 7, 8]]))
            elif Case == 'NrClust':
                USECOLS = NP.concatenate(([0],[elem+20*ThresholdList.index(Threshold) for elem in \
                                              [4]]))
            else:
                raise NameError('Case is not set correctly. Please use Case = %s OR %s OR %s' %\
                                ('Entropy', 'Cluster', 'Plateau'))
           #----- LOAD SlopeArray
            SlopeArray = NP.genfromtxt(SlopeDir+SlopeName, usecols=USECOLS)
            NrClust    = NP.genfromtxt(SlopeDir+SlopeName, usecols=(0,4+20*ThresholdList.index(Threshold)))
            if len(SlopeArray.shape) == 1:
                SlopeArray = NP.reshape(SlopeArray, (1, -1))
                NrClust    = NP.reshape(NrClust,    (1, -1))
            #N = len(ThresholdList)
            #criticalValue = t.ppf(1.-(1.-Confidence)/2., N-2)
            for NotTraj in TrajExcept:
                SlopeArray = SlopeArray[SlopeArray[:,0]!=NotTraj]
                NrClust    = NrClust[NrClust[:,0]!=NotTraj]
     #------------- if number of ROWS and COLS are not defined: generate maximally 30 trajNrs in one row   
            if SupGrid is None or Splitter is None:
                SupGrid  = ((len(SlopeArray[:,0])-1)/24+1, 1)
                Splitter = [0+elem*len(SlopeArray[:,0])/SupGrid[0] for elem in range(1+SupGrid[0])]
                Splitter = zip(Splitter[0:-1], Splitter[1:])
     #------------- FIGURE PLOT
            fig = plt.figure(figsize=(max(17/24.*NP.unique(Splitter)[SupGrid[1]],4.1),8/2.*SupGrid[0]))
            for Index in range(len(Splitter)):
                K,L = Splitter[Index] #[Splitter[Index], Splitter[Index+1]]
              #### subplot GRID & MARGINS  
                AX = plt.subplot2grid( SupGrid, (Index/SupGrid[1], Index%SupGrid[1]) )
                if len(SlopeArray[:,0]) > 5:
                    plt.subplots_adjust(wspace=0.01, hspace=0.4, left=0.085, right=0.99, bottom=0.10, top=0.86)
                else:
                    plt.subplots_adjust(wspace=0.01, hspace=0.4, left=0.265, right=0.89, bottom=0.16, top=0.69)

         #---------- SLOPES
                if Case == 'Entropy' or Case == 'Cluster':
                    for Cols in range(len(SlopeArray[0,2::2])):
                        XX = NP.arange(0,len(SlopeArray[K:L,0]),1)+[-0.3,-0.1,0.1,0.3][Cols]    
                        plt.errorbar(XX, [0]*len(XX), ls='', marker='s', color=C[Cols], ms=7.5)
                    Ymax = 0
                    critIndex = 0
                    for Cols in range(len(SlopeArray[0,2::2])):
                        if critIndex < len(SlopeTimeArray):
                            criticalValues = [TPPF.ppf(1.-(1.-Confidence)/2., SlopeTimeArray[critIndex]-2)]*(L-K)
                        else:
                            criticalValues = [TPPF.ppf(1.-(1.-Confidence)/2., elem-2) for elem in SlopeArray[K:L,1]]
                        XX = NP.arange(0,len(SlopeArray[K:L,0]),1)+[-0.3,-0.1,0.1,0.3][Cols]
                        plt.bar(XX-0.1, SlopeArray[K:L,Cols*2+2], yerr=SlopeArray[K:L,Cols*2+3]*criticalValues, 
                                color=C[Cols], width=0.2, error_kw=dict(lw=3, capsize=4, capthick=1, ecolor='k'))
                        Ymax = max(Ymax, NP.max(SlopeArray[:,Cols*2+1]))
                        critIndex += 1
                    if YMAX is None:
                        YMAX = Ymax
         #---------- PLATEAU
                elif Case == 'Plateau':
                    Ymax = 0
                    for Cols in range(2):
                        XX = NP.arange(0,len(SlopeArray[K:L,0]),1)+[-0.4,0.0][Cols]
                        plt.bar(XX, SlopeArray[K:L,Cols+1]*TimeStep, width=0.4, color=C[Cols])
                        Ymax = max(Ymax, NP.max(SlopeArray[:,Cols+1]*TimeStep))
                  #------- Plot cluster [%] to which belong the largest cluster
                        if Cols == 1:
                            plt.xlim([-0.5, len(SlopeArray[K:L,0])-1+0.5]);
                            POS2 = AX.get_position()
                            for xx in range(len(XX)):
                                plt.figtext((POS2.x1-POS2.x0)/(AX.get_xlim()[1]-AX.get_xlim()[0])*(XX[xx]+0.58)+POS2.x0, 
                                            (POS2.y1-POS2.y0)*0.2+POS2.y0, '%.2f' % SlopeArray[K:L,3][xx], 
                                            rotation=90, color='k', fontsize=20)
                    if YMAX is None:
                        YMAX = Ymax
         #---------- Nr of Cluster vs TrajNr                    
                else:
                    Ymax = 0
                    XX = NP.arange(0,len(SlopeArray[K:L,0]),1)-0.15
                    plt.bar(XX-0.1, SlopeArray[K:L,1], color=C[0], width=0.5)
                    Ymax = max(Ymax, NP.max(SlopeArray[:,1]))
                    YMAX = Ymax
              #### LEGEND
                if Index == 0:
                    if 17/30.*NP.unique(Splitter)[SupGrid[1]] > 4.1: NCOL = 10
                    else: NCOL = 2
                    if Case == 'Plateau':
                        plt.legend(['last cluster', 'largest cluster'], fontsize=19.5, 
                               loc=3, bbox_to_anchor=(+0., 1.25, 2., 0.092), ncol=NCOL, borderaxespad=0., numpoints=1)
                    elif Case == 'Entropy':
                        plt.legend(['last %sns' % (SlopeTimeArray[0]*TimeStep), 'last %sns' % (SlopeTimeArray[1]*TimeStep), 
                                    'last %sns' % (SlopeTimeArray[2]*TimeStep), 'last cluster'], fontsize=19.5, 
                               loc=3, bbox_to_anchor=(+0., 1.25, 2., 0.092), ncol=NCOL, borderaxespad=0., numpoints=1)
                    elif Case == 'Cluster':
                        plt.legend(['last %sns' % (SlopeTimeArray[0]*TimeStep), 'last %sns' % (SlopeTimeArray[1]*TimeStep), 
                                    'last %sns' % (SlopeTimeArray[2]*TimeStep)], fontsize=19.5, 
                               loc=3, bbox_to_anchor=(+0., 1.25, 2., 0.092), ncol=NCOL, borderaxespad=0., numpoints=1)
              #### grid of vertical lines AND 0 horizontally
                for Kai in NP.arange(0,len(SlopeArray[K:L,0])-1,1)+0.5:
                    plt.axvline(Kai, ls='--', lw=2, color='grey')
                plt.axhline(0, ls='--', lw=2, color='grey'); 
              #### X-/Y-Axis
                plt.xlim([-0.5, len(SlopeArray[K:L,0])-1+0.5]); 
                plt.xlabel('traj Nr', fontsize=(22 if Index/SupGrid[1]+1==SupGrid[0] else 0))
                plt.xticks(range(0,len(SlopeArray[K:L,0]),1), range(1,1+len(SlopeArray[K:L,0]),1), fontsize=20);
                if Case=='Entropy': 
                    plt.ylim([-1.2, 1.2]); 
                    if Index%SupGrid[1] == 0:
                        plt.yticks([-1,-0.5,0,0.5,1], fontsize=17)
                    else:
                        plt.yticks([])
                else:
                    plt.ylim([-1,YMAX]); plt.yticks(fontsize=(20 if Index%SupGrid[1] == 0 else 0))
                if Case=='Entropy' or Case=='Cluster':
                    plt.ylabel('slopes', fontsize=(22 if Index%SupGrid[1]==0 else 0))
                else:
                    plt.ylabel(('Nr of cluster' if Case!='Plateau' else 'plateau [ns]'), fontsize=(22 if Index%SupGrid[1]==0 else 0))
              #### 2nd X-axis Nr of Clusters    
                if Case == 'Entropy' or Case == 'Cluster' or Case == 'Plateau':
                    ax2 = plt.twiny(); 
                    plt.xticks(range(0,len(SlopeArray[K:L,0]),1), [int(elem) for elem in NrClust[K:L,1]], fontsize=20, color='b');
                    ax2.set_xlim([-0.5, len(SlopeArray[K:L,0])-1+0.5]); ax2.set_xlabel('Nr of clusters', color='b', fontsize=22) 
              #### FigTEXT if TrajNrs are intentionally split
                if SupGrid[1] > 1:
                    POS = AX.get_position()
                    if FigText is None:
                        FG = plt.figtext((POS.x1-POS.x0)/AX.get_xlim()[1]*(1*1.15)+POS.x0, (POS.y1-POS.y0)*0.85+POS.y0, 
                                     '%s)' % string.ascii_lowercase[Index], rotation=0, color='k', fontsize=22,
                                     transform=AX.transAxes)
                    else:
                        FG = plt.figtext((POS.x1-POS.x0)/AX.get_xlim()[1]*(1*1.15)+POS.x0, (POS.y1-POS.y0)*0.85+POS.y0, 
                                     '%s' % FigText[Index], rotation=0, color='k', fontsize=22,
                                     transform=AX.transAxes)
                    FG.set_bbox(dict(color='w', alpha=0.5))
     #------------- SAVE PLOT                
            if SaveDir is not None:
                Generate_Directories(SaveDir)
                plt.savefig('%s%s%s' % (SaveDir, Case, SlopeName.replace('.txt','.pdf')))
                plt.close()
        else:
            print 'Either the submitted \n\t>>Threshold = '+str(Threshold)+'<<\nis NOT in the Clustering-Threshold = {}'.format(ThresholdList)
            print 'OR the submitted \n\t>>ThresholdList = {}<< \nis NOT the same as the Clustering-Threshold'.format(ThresholdList)
            print 'OR the submitted \n\t>>SlopeTimeArray = {}<<\nis NOT the same as the Slopetimes'.format(SlopeTimeArray)
    else:
        print 'Either \n\t>>%s%s%s<< \nalready exists or \n\t>>%s%s<< \ndoes NOT exist' % \
                (SaveDir,Case,SlopeName.replace('.txt','.pdf'),   SlopeDir,SlopeName)
#######################################################
#--- OVERLAP vs THRESHOLD
#################
def Plot_Overlap_VS_Threshold(OverlapDir, OverlapList1, Percentile1=25, Percentile2=75, Median=False, 
                              Interpolation='linear', OverlapList2=None, XLIM1=[None,None], XLIM2=[None,None], 
                              MolName1='', MolName2='', LegendList=[None], SaveDir=None, SaveName=None):
    """
v18.11.16
This function generates the plots "Overlap vs Threshold r" for conf/dens + the corresponding integral.
- uses always ALL K reference trajectories
- possibility to submit multiple OverlapMatrices to plot for instance multiple groups together
- possibility to submit a second OverlapMatrix to compare a second molecule with its own x-axis
- possibility to submit different X-limits to "normalize" the curves for instance on a certain threshold interval
- all Groups from one OverlapList are used
- XLIM1 & XLIM2 are used also for the integral limits
- ERROR HANDLING: plots the first percentile and second percentile as error bars,
                  if the average value is outside, use the average as one limit
INPUT:
    OverlapDir    : {STRING}      Directory, where the Overlap is located, e.g. 'Overlap/';
    OverlapList   : {LIST}        List of Overlap filenames containing different cases, e.g. Pairs, different Groups,
                                    which are then plotted separately in one plot
                                    e.g. ['Overlap_ALLvsALL.txt' 'Overlap_AvsB.txt'];
    Percentile1   : {INT}         <default 25> Value of the 1st percentile [in %], e.g. 25 for the 1st quartile;
    Percentile2   : {INT}         <default 75> Value of the 2nd percentile [in %], e.g. 75 for the 3rd quartile;
    Median        : {BOOL}        <default False>  if True, the median is plotted over all K reference trajs, else the Avg;
    Interpolation : {STRING}      <default linear> Interpolation method to find the percentile values,
                                                   e.g. 'linear', 'lower', 'higher', 'midpoint', 'nearest';
    OverlapList2  : {LIST}        <default None>         second List, with same properties as OverlapList2,
                                                            e.g. a second molecule to compare;
    XLIM1         : {FLOAT-LIST}  <default [None,None]>, x-limits for the first OverlapList, plt.xlim(XLIM1);
    XLIM2         : {FLOAT-LIST}  <default [None,None]>, x-limits2 for the second axis of the 2nd OverlapList2;
    MolName1      : {STRING}      <default ''>           name to specify the first OverlapList1, e.g. 'Molecule1';
    MolName2      : {STRING}      <default ''>           name to specify the 2nd   OverlapList2, e.g. 'Molecule2';
    LegendList    : {LIST}        <default [None]>       Legend for the single elements of the OverlapList1, 
                                                            e.g. ['ALL','AvsB'];
    SaveDir       : {STRING}                             saving directory for the PDF, e.g. 'PDFs/';
    SaveName      : {STRING}                             savename, e.g. 'OverlaPvsThreshold.pdf';
OUTPUT:
    if SaveDir&SaveName is not None:
        the OverlaPvsThreshold.pdf is stored
    """
    import matplotlib.pyplot as plt
    if SaveDir is not None and SaveName is not None and os.path.exists(SaveDir+SaveName):
        print 'Figure already exists\n%s%s' % (SaveDir, SaveName)
        return
    Color = ['b', 'k', 'r', 'g', 'm', 'c', 'y', '0.5','0.8']
  
  ####################################
    OvR1 = Generate_OvR(OverlapDir, OverlapList1, Percentile1, Percentile2, Median, Interpolation)
    #--
    Integrals1 = NP.zeros( (len(OvR1[0,1:])/6, 6) ) ## densIS1, densIS1_lo, densIS1_up, confIS1, confIS1_lo, confIS1_up
    for Koi in range(0,len(OvR1[0,1:]),6):
        Integrals1[Koi/6, :] = Generate_Integrals(OvR1[:,[0,Koi+1,Koi+2,Koi+3,Koi+4,Koi+5,Koi+6]], *XLIM1)
    #-----
    if OverlapList2 is not None:
        OvR2 = Generate_OvR(OverlapDir, OverlapList2, Percentile1, Percentile2, Median, Interpolation)
        #--
        Integrals2 = NP.zeros( (len(OvR2[0,1:])/6, 6) ) ## densIS2, densIS2_lo, densIS2_up, confIS2, confIS2_lo, confIS2_up
        for Koi in range(0,len(OvR2[0,1:]),6):
            Integrals2[Koi/6, :] = Generate_Integrals(OvR2[:,[0,Koi+1,Koi+2,Koi+3,Koi+4,Koi+5,Koi+6]], *XLIM2)
    #----------------#
    #----- PLOT -----#
    #----------------#
    fig = plt.figure(figsize=(15,6.5))
    for densconfIndex in [0,1]:
  #------- conformational & density OVERLAP
        plt.subplot2grid( (1,5), (0,densconfIndex*2), colspan=2 ); 
        plt.title('%s overlap\n' % ['conformational','density'][densconfIndex], fontsize=25);
        plt.subplots_adjust(wspace=0, left=0.08, right=0.95, top=0.84, bottom=0.30)
       #-------- LEGEND
        if MolName1 != '': plt.plot(0,0, ls='-', color='r', lw=1.5, marker='s', ms=9, mfc='w'); 
        if MolName2 != '': plt.plot(0,0, ls='-', color='b', lw=1.5, marker='<', ms=9, mfc='w');
        if LegendList != [None]:
            for CC in Color:
                plt.errorbar(0,0, yerr=0, ls='',marker='s', ms=8, color=CC)
        if densconfIndex == 0:
            LEGEND = [elem for elem in [MolName1, MolName2]+LegendList if elem != '' and elem is not None]
            if LEGEND != []:
                plt.legend(LEGEND, numpoints=1, framealpha=0.5, loc=4, fontsize=18)
       #---- 1st AXIS: Plot Overlap vs Threshold V3
        Cindex = 0
        for confIndex in range([4,1][densconfIndex],len(OvR1[0,:]),6):
           #### CORRECT yerror if the value lies outside the error range, then is set to the upper/lower limit 
            YERR = [(OvR1[:,confIndex]-OvR1[:,confIndex+1]), (OvR1[:,confIndex+2]-OvR1[:,confIndex])]
            YERR[0][YERR[0]<0] = 0; YERR[1][YERR[1]<0] = 0;
            (_, caps, _) = plt.errorbar(OvR1[:,0], OvR1[:,confIndex], color='r', yerr=YERR, barsabove=True,
                                        elinewidth=0.7, capsize=7, capthick=2.5, ecolor='r', marker='s', 
                                        mfc=Color[Cindex%9], lw=1.0, ms=8, alpha=0.8); 
            for cap in caps:
                cap.set_color(Color[Cindex%9])
            Cindex += 1
        plt.xticks(fontsize=21, color='r');
        if XLIM1 != [None, None]: plt.xlim(XLIM1)
        else:                     plt.xlim([OvR1[0,0], OvR1[-1,0]])
        if densconfIndex == 0: plt.yticks(fontsize=21); plt.ylabel('overlap', fontsize=23);
        else:                  plt.yticks(fontsize=0);
        plt.xlabel('threshold r [nm]', fontsize=23);
        plt.grid(axis='y')
        plt.ylim([-0.025,1.025])
       #---- 2nd AXIS: Plot Overlap vs Threshold MET
        if OverlapList2 is not None: 
            Cindex = 0
            ax2 = plt.twiny();
            for tl in ax2.get_xticklabels(): tl.set_color('b')
            for tl in ax2.xaxis.get_majorticklabels(): tl.set_fontsize(21)
            for confIndex in range([4,1][densconfIndex],len(OvR2[0,:]),6):
                if not (NP.all(OvR2[:,confIndex]-OvR2[:,confIndex+1]>=0) and NP.all(OvR2[:,confIndex+2]-OvR2[:,confIndex]>=0)):
                    print 'PERCENTILE VALUES LEAD TO ERROR BARS OUTSIDE THE AVERAGE, confIndex=%s, OvR2, Color=%s, r1=%s, r2=%s' % \
                    (confIndex, Color[Cindex%9], OvR2[OvR2[:,confIndex]-OvR2[:,confIndex+1]<0][:,0],OvR2[OvR2[:,confIndex+2]-OvR2[:,confIndex]<0][:,0])
                YERR = [(OvR2[:,confIndex]-OvR2[:,confIndex+1]), (OvR2[:,confIndex+2]-OvR2[:,confIndex])]
                YERR[0][YERR[0]<0] = 0; YERR[1][YERR[1]<0] = 0
                (_, caps, _) = ax2.errorbar(OvR2[:,0], OvR2[:,confIndex], color='b', yerr=YERR, barsabove=True,
                                            elinewidth=0.7, capsize=7, capthick=2.5, ecolor='b', marker='<', ms=10, 
                                            mfc=Color[Cindex%9], lw=1.0, alpha=0.8); 
                for cap in caps:
                    cap.set_color(Color[Cindex%9])
                Cindex += 1
            if XLIM2 != [None, None]: ax2.set_xlim(XLIM2)
            else:                     ax2.set_xlim([OvR2[0,0], OvR2[-1,0]])
        plt.ylim([-0.025,1.025])
  #------- AREA/Integral
    plt.subplot2grid( (1,5), (0,4) )
    plt.title('averaged\n', fontsize=25)
    if LEGEND != []: 
        plt.xticks(range(1,1+len(OvR1[0,1:])/5,1), LEGEND[-len(OvR1[0,1:])/5:], fontsize=21, rotation=45)
    else:            
        plt.xticks(range(1,1+len(OvR1[0,1:])/5,1), fontsize=21)
    plt.yticks(fontsize=0)
    plt.xlim([0.5, len(OvR1[0,1:])/5+0.5]); plt.ylim([-0.025,1.025]); plt.grid(axis='y')
    IntColors = ['ro:', 'r*:']
    plt.errorbar(range(1,1+len(Integrals1[:,0]),1), Integrals1[:,0], yerr=[Integrals1[:,1],Integrals1[:,2]], color='r', marker='o', ms=8, barsabove=True, capsize=5, capthick=2.0); 
    plt.errorbar(range(1,1+len(Integrals1[:,3]),1), Integrals1[:,3], yerr=[Integrals1[:,4],Integrals1[:,5]], color='r', marker='*', ms=10, barsabove=True, capsize=5, capthick=2.0); 
    if OverlapList2 is not None:
        plt.errorbar(range(1,1+len(Integrals2[:,0]),1), Integrals2[:,0], yerr=[Integrals2[:,1],Integrals2[:,2]], color='b', marker='o', ms=8, barsabove=True, capsize=5, capthick=2.0); 
        plt.errorbar(range(1,1+len(Integrals2[:,3]),1), Integrals2[:,3], yerr=[Integrals2[:,4],Integrals2[:,5]], color='b', marker='*', ms=10, barsabove=True, capsize=5, capthick=2.0); 
   ###
    if OverlapList2 is not None:
        plt.legend(['%s (dens)' % MolName1,'%s (dens)' % MolName2,'%s (conf)' % MolName1,'%s (conf)' % MolName2], 
                   numpoints=1, ncol=4, loc='best', framealpha=0.35, fontsize=19,
                   bbox_to_anchor=(-1.056, -0.30, 1., .092))
    else:
        plt.legend(['%s (dens)' % MolName1,'%s (conf)' % MolName1], 
                   numpoints=1, ncol=2, loc='best', framealpha=0.35, fontsize=19,
                   bbox_to_anchor=(0.036, 0.11, 1., .092))
  ##########------- SAVE PDF ---------##########
    if SaveDir is not None and SaveName is not None:
        #### #### ####
        #---- generate Directories  
        Generate_Directories(SaveDir)
        plt.savefig(SaveDir+SaveName)
#--- --- --- --- --- --- --- --- --- --- ---
def Generate_OvR(OverlapDir, OverlapList, Percentile1, Percentile2, Median=False, Interpolation='linear'):
    """
v18.11.16
    - supporting function to generate 'Overlap vs Threshold'
    - Threshold | densO1 | densO1_Perc1 | densO1_Perc2 | confO1 | confO1_Perc1 | confO1_Perc2 | densO2 | ... for each element in OverlapList one densOx | confOx pair
    - generates/uses values, where all overlap values are projected onto every frame/trajectory, uses all K reference trajectories
INPUT:
    OverlapDir    : {STRING}      Directory, where the Overlap is located, e.g. 'Overlap/';
    OverlapList   : {LIST}        List of Overlap filenames containing different cases, e.g. Pairs, different Groups, ...;
    Percentile1   : {INT}         Value of the 1st percentile [in %], e.g. 25 for the 1st quartile;
    Percentile2   : {INT}         Value of the 2nd percentile [in %], e.g. 75 for the 3rd quartile;
    Median        : {BOOL}        <default False>  if True, the median is plotted over all K reference trajs, else the Avg;
    Interpolation : {STRING}      <default linear> Interpolation method to find the percentile values,
                                                   e.g. 'linear', 'lower', 'higher', 'midpoint', 'nearest';
OUTPUT:
    return OvR - {NP.NDARRAY} containing for each group
            Threshold | densO1 | densO1Perc1 | densO1Perc2 | confO1 | confO1Perc1 | confO1Perc2 | ... 
    """
    for FileName in OverlapList:
    #----- Loaded Overlap with all groups
        if os.path.exists('%s%s' % (OverlapDir, FileName)):
            tempO = NP.genfromtxt('%s%s' % (OverlapDir, FileName))
        else:
            raise NameError('\t%s%s\ndoes not exist, check your OverlapList1/2!' % (OverlapDir, FileName))
    #----- ERROR detection
        try:
            if len(NP.unique(tempO[:,1])) != len(OvR[:,0]) or NP.any(NP.unique(tempO[:,1])-OvR[:,0]):
                print len(NP.unique(tempO[:,1])), len(OvR[:,0]), NP.unique(tempO[:,1])-OvR[:,0], \
                        NP.any(NP.unique(tempO[:,1])-OvR[:,0])
                raise ValueError('The submitted Overlap files do not have the same ThresholdList!'+\
                                 '\n%s != \n%s' % (OvR[:,0], NP.unique(tempO[:,1])))
        except NameError:
            pass

    #----- generate OvR for this Overlap only: Threshold | densO1 | confO1 | densO2 | confO2 | ...
        tempOvR = NP.zeros( (len(NP.unique(tempO[:,1])), 6*len(tempO[0,2:])/4) )  
        OvR_Row = 0
        for Threshold in NP.unique(tempO[:,1]): ## 
            #tempOvR[OvR, 0] = Threshold
            for Koi in range(0,len(tempOvR[0,:]),6):
                ## densO
                if Median:
                    tempOvR[OvR_Row, Koi] = NP.median(tempO[tempO[:,1]==Threshold][:,2+4*Koi/6])
                else:
                    tempOvR[OvR_Row, Koi] = NP.average(tempO[tempO[:,1]==Threshold][:,2+4*Koi/6])
                ## densO Percentile1
                tempOvR[OvR_Row, Koi+1] = NP.percentile(tempO[tempO[:,1]==Threshold][:,2+4*Koi/6], Percentile1, 
                                                      interpolation=Interpolation)
                ## densO Percentile2
                tempOvR[OvR_Row, Koi+2] = NP.percentile(tempO[tempO[:,1]==Threshold][:,2+4*Koi/6], Percentile2, 
                                                      interpolation=Interpolation if Interpolation!='lower' else 'higher')
                ## confO
                if Median:
                    tempOvR[OvR_Row, Koi+3] = NP.median(NP.divide(tempO[tempO[:,1]==Threshold][:,4+4*Koi/6],
                                                                  tempO[tempO[:,1]==Threshold][:,5+4*Koi/6]))
                else:
                    tempOvR[OvR_Row, Koi+3] = NP.divide(NP.sum(tempO[tempO[:,1]==Threshold][:,4+4*Koi/6]),
                                                        NP.sum(tempO[tempO[:,1]==Threshold][:,5+4*Koi/6]))
                ## confO Percentile1
                tempOvR[OvR_Row, Koi+4] = NP.percentile(NP.divide(tempO[tempO[:,1]==Threshold][:,4+4*Koi/6],
                                                                  tempO[tempO[:,1]==Threshold][:,5+4*Koi/6]),
                                                        Percentile1, interpolation=Interpolation)
                ## confO Percentile2
                tempOvR[OvR_Row, Koi+5] = NP.percentile(NP.divide(tempO[tempO[:,1]==Threshold][:,4+4*Koi/6],
                                                                  tempO[tempO[:,1]==Threshold][:,5+4*Koi/6]),
                                                        Percentile2, interpolation=Interpolation if Interpolation!='lower' else 'higher')
            OvR_Row += 1
    #----- Store everything in ONE MATRIX
        if 'OvR' not in locals():
            OvR = NP.zeros( (len(NP.unique(tempO[:,1])), 1) )
            OvR[:,0] = NP.unique(tempO[:,1])
            OvR = NP.concatenate( (OvR, tempOvR), axis=1 )
        else:
            OvR = NP.concatenate( (OvR, tempOvR), axis=1 )
    #-----
    return OvR
##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### 
def Generate_Integrals(Overlap, Rlower, Rupper):
    """
v18.11.16
- supporting function to calculate the integral below the curve 'Overlap vs Threshold' 
- uses Simpson integration
- integrates EITHER from lowest Threshold to the largest Threshold, if Rlower & Rupper are None
             OR     from (the closest value to) Rlower to (the closest value to) Rupper
INPUT:
    Overlap - {NP.NDARRAY} contains 'Threshold | densO | densUp | densLo | confO | confUp | confLo' with using ALL K reference trajs 
    Rlower  - {FLOAT}      if None, Integral between lowest Threshold and largest Threshold, else start from value closest to Rlower
    Rupper  - {FLOAT}      if None, Integral between lowest Threshold and largest Threshold, else end at value closest to Rupper
OUTPUT:
    return densIS1, densErr, confIS1, confErr with asymmetric errors, with lo=Lower value, and up=Upper value
    densIS1   = SimpsonIntegral of density overlap
    densIS_lo = max(0,densIS-densIS_lo)
    densIS_up = max(0,densIS_up-densIS)
    confIS    = SimpsonIntegral of conformational overlap
    confIS_lo = max(0,confIS-confIS_lo)
    confIS_up = max(0,confIS_up-confIS)
    """
  ##### CALCULATE INTEGRAL: if Rlower/Rupper is not None, they are used for the integration
  #     due to discrete usage of R extract the CLOSEST R to Rlower/Rupper to avoid additional fitting
  # Overlap: Threshold | dens | conf
    Ilower = 0; Iupper = len(Overlap[:,0])
    if Rlower is not None and Rupper is not None:
        DIFFl = 100; DIFFu = 100; 
        for RRR in range(len(Overlap[:,0])):
            if Rlower is not None and abs(Overlap[RRR,0]-Rlower) < DIFFl:
                DIFFl = abs(Overlap[RRR,0]-Rlower)
                Ilower = RRR
            if Rupper is not None and abs(Overlap[RRR,0]-Rupper) < DIFFu:
                DIFFu = abs(Overlap[RRR,0]-Rupper)
                Iupper = RRR+1
   #----- Maximal Area approx: Rmax*1 - Rmin*1
    FULL = Overlap[Iupper-1,0]-Overlap[Ilower,0] # Rmax-Rmin
   #----- Calculate Integral
    densIS    = simps(y=Overlap[Ilower:Iupper,1], x=Overlap[Ilower:Iupper,0])/FULL
    densIS_lo = simps(y=Overlap[Ilower:Iupper,2], x=Overlap[Ilower:Iupper,0])/FULL
    densIS_up = simps(y=Overlap[Ilower:Iupper,3], x=Overlap[Ilower:Iupper,0])/FULL
    
    confIS    = simps(y=Overlap[Ilower:Iupper,4], x=Overlap[Ilower:Iupper,0])/FULL
    confIS_lo = simps(y=Overlap[Ilower:Iupper,5], x=Overlap[Ilower:Iupper,0])/FULL
    confIS_up = simps(y=Overlap[Ilower:Iupper,6], x=Overlap[Ilower:Iupper,0])/FULL
 #----- RETURN, middle value, lower value and upper value 
    return densIS, max(0,densIS-densIS_lo), max(0,densIS_up-densIS), \
           confIS, max(0,confIS-confIS_lo), max(0,confIS_up-confIS)

#######################################################
#--- HEATMAP
#################
def Plot_HeatMap_1vs1(OverlapDir, FileName, Threshold, ClusterDir=None, ClusterFileName=None, AllPrject=True, 
                      TrajExcept=[], Title='', Grid=[], CaseTitles=[], SaveDir=None, SaveName=None):
    """
v12.10.16   
This function plots the heatmap of trajectory X vs trajectory Y (whereas GroupX vs GroupY should also work).
- possibility to use AllPrject EITHER projection on both groups, OR lower triangular projection on X and upper on Y
- Grid & CaseTitles give the possibility to split the Heatmap into different regions, where Grid gives the split coords
- if a ClusterFile is submitted, the number of clusters with the same TrajNr/GroupNr is plotted below
INPUT:
    OverlapDir      : {STRING}      Directory, where the overlap files are located, e.g. 'Overlap/';
    FileName        : {STRING}      Overlap file, which contains ALL X vs Y Pairs, then a heatmap matrix is constructed;
    Threshold       : {FLOAT}       Threshold used for the overlap calculation, for which the heatmap is generated,
                                        e.g. 0.2, has to match the ThresholdList of the Overlap file;
    ClusterDir      : {STRING}      <default None>  Directory, where the clustering files are located, e.g. 'Clustering/';
    ClusterFileName : {STRING}      <default None>  Clustering file containing the centers, e.g. 'Cluster_Centers_LOCAL.txt';
    AllPrject       : {BOOL}        <default True> True - Heatmap is  symmetric = overlap is projected on both groups X & Y,
                                                    False- Heatmap is asymmetric = lower triangular projection on X, upper on Y;
    TrajExcept      : {INT-LIST}    <default []>    possibility to EXCLUDE manually trajectories by deleting the 
                                                    Rows and Columns (starting from 1 to N) and similar the Clustering,
                                            e.g. TrajExcept=[1,2] delete the first 2 trajectories;
    Title           : {STRING}      <default ''>    possibility to adjust the title of the plot, e.g. 'Molecule, r=0.11nm';
    Grid            : {INT-LIST}    <default []>    prints solid lines at Grid for ordering, 
                                            e.g. [5, 10, 15, 20] to sort 25 trajectories equally;
    CaseTitles      : {LIST}        <default []>    name the different trajectory groups defined by Grid,
                                            e.g. ['Grp1', 'Grp2', 'Grp3', 'Grp4', 'Grp5'], len(Grid)+1 == len(CaseTitles);
    SaveDir         : {STRING}      Directory, where the PDF is stored, e.g. 'HeatMaps/';
    SaveName        : {STRING}      save name, e.g. 'Molecule_HeatMap_Specifications.pdf';
OUTPUT:
    """
    import matplotlib.pyplot as plt
    if SaveDir is not None and SaveName is not None and os.path.exists(SaveDir+SaveName):
        print 'Figure already exists\n%s%s' % (SaveDir, SaveName)
        return
  #----------
    #-----
    if os.path.exists('%s%s' % (ClusterDir, ClusterFileName)):
        RowFigs = 9
        NrClusters = NP.genfromtxt('%s%s' % (ClusterDir, ClusterFileName), usecols=(0,1,2))
        #### strip clusters with -1 & strip only correct threshold & strip TrajExcept
        NrClusters = NrClusters[NrClusters[:,2]!=-1]
        NrClusters = NrClusters[NrClusters[:,1]==Threshold]
        for TrajNr in TrajExcept:
            NrClusters = NrClusters[NrClusters[:,0]!=TrajNr]
    else:
        RowFigs = 8
  ##
    fig = plt.figure(figsize=(8.5/16*12,12)) 
    for Relative in range(2): # 0 = conformational, 1 = density
 ###### subplot GRID / MARGINS
        plt.subplot2grid( (RowFigs,1), (4*Relative,0), rowspan=4);
        plt.subplots_adjust(wspace=0.55, hspace=0.05, left=0.14, right=0.99, bottom=0.055, top=0.865)
      ## HEATMAP
        Array = Generate_1vs1_Matrix(OverlapDir, FileName, Threshold, ['conformational','density'][Relative], 
                                     AllPrject, TrajExcept)
        imgplot = plt.imshow(Array, interpolation='nearest', alpha=1, cmap='bwr', origin='low', 
                             aspect='auto',vmin=0, vmax=1)
      ## X-/Y-AXIS
        if Grid != []:
            plt.yticks([elem-Grid[0] for elem in Grid+[len(Array[:,0])]],
                       [elem-Grid[0]+1 for elem in Grid+[len(Array[:,0])]],fontsize=18);
        elif len(Array[:,0]) <= 5:
            plt.yticks(range(0,len(Array[:,0]),1), range(1,len(Array[:,0])+1,1), fontsize=18)
        else:
            plt.yticks(range(0,len(Array[:,0]),5), range(1,len(Array[:,0]),5), fontsize=18)
        plt.xticks(fontsize=0); plt.ylabel('TrajNr     (%s)' % ['conformational', 'density'][Relative], fontsize=22)
      ## GRID - heatmap
        Indo = 0
        for Kai in Grid:
            plt.axvline(Kai-0.5, ls='-',color='k', lw=1.5 if Indo%2==0 else 4)
            plt.axhline(Kai-0.5, ls='-',color='k', lw=1.5 if Indo%2==0 else 4)
            Indo += 1
      ## X-Axis
        if RowFigs == 8:
            if Grid != []:
                plt.xticks([elem-Grid[0] for elem in Grid+[len(Array[:,0])]],
                           [elem-Grid[0]+1 for elem in Grid+[len(Array[:,0])]],fontsize=18);
            elif len(Array[:,0]) <= 5:
                plt.xticks(range(0,len(Array[:,0]),1), range(1,len(Array[:,0])+1,1), fontsize=18)
            else:
                plt.xticks(range(0,len(Array[:,0]),5), range(1,len(Array[:,0]),5), fontsize=18)
      ## SECOND X-Axis  
        if Relative == 0 and CaseTitles != [] and Grid != []:
            ax3 = plt.twiny(); ax3.set_xlim([1,len(Array[:,0])+1]); 
            if len(Array[:,0]) <= 5:
                ax3.set_xticks([elem+0.5 for elem in range(1,len(Array[:,0])+1,1)])
                pass
            else:
                ax3.set_xticks([1+(Grid[elem] if elem >= 0 else 0) + \
                                 ((Grid[elem+1] if elem+1 < len(Grid) else len(Array[:,0])) - \
                                  (Grid[elem] if elem >= 0 else 0))/2 for elem in range(-1,len(Grid))])
            ax3.set_xticklabels(CaseTitles, fontsize=20, color='b')
      ## COLORBAR
        if Relative == 0:
            cbaxes = fig.add_axes([.1, 0.945, 0.84, 0.02])
            cbar = plt.colorbar(imgplot, orientation='horizontal', cax=cbaxes, ticks=[0.0,0.2,0.4,0.6,0.8,1])
            cbar.ax.tick_params(labelsize=16); cbar.set_label('Overlap %s' % Title, fontsize=20)
            cbar.ax.xaxis.set_label_position('top')  
  ## Nr of Clusters vs TrajNrs
    if os.path.exists('%s%s' % (ClusterDir, ClusterFileName)):
        AX2=plt.subplot2grid( (RowFigs,1), (8,0) )
        plt.plot(range(1,len(NrClusters)+1), NrClusters[:,2], color='g', marker='s', ls=':', ms=8.5); 
        plt.ylim([0,max(NrClusters[:,2])*1.2]); plt.xlim([0.5,len(NrClusters)+0.5])
      ## GRID - Nr of Clusters 
        Indo = 0
        for Kai in Grid:
            plt.axvline(Kai+0.5, ls='-',color='k', lw=1 if Indo%2==0 else 4)
            Indo += 1
        plt.yticks(fontsize=18); plt.grid(axis='y',ls='--',color='lightgrey')
        plt.ylabel('# clusters',fontsize=20); AX2=plt.locator_params(nbins=4);
        if len(Array[:,0]) <= 5:
            plt.xticks(range(1,len(Array[:,0])+1,1), fontsize=18)
        elif Grid != []:
            plt.xticks([len(NrClusters)/6*Kai+1 for Kai in range(6)], fontsize=18); plt.xlabel('TrajNr', fontsize=20);
        else:
            plt.xticks(range(1,len(Array[:,0])+1,5), fontsize=18)
  ##########------- SAVE PDF ---------##########
    if SaveDir is not None and SaveName is not None:
        #### #### ####
        #---- generate Directories  
        Generate_Directories(SaveDir)
        plt.savefig(SaveDir+SaveName)  
        plt.close()

#----- ----- ----- ------ ------

def Generate_1vs1_Matrix(OverlapDir='Amber14Trajs/Overlap/', FileName='Overlap_R5_AllPairs_0-2000_MF+sMD.txt',
                         Threshold=0.4, Case='conformational', AllPrject=False, TrajExcept=[]):
    """
v21.11.16
This function generates HeatMaps of the OVERLAP (conformational or density) for [Group]X vs [Group]Y.
It is possible to generate a symmetric HeatMap using AllPrject=True or construct the projection on both groups 
OR             to generate an asymmetric HeatMap, where on the lower triangular is the projection on the first group
                                                  and   on the upper triangular is the projection on the snd   group.
INPUT:
    OverlapDir : {STRING}      Directory, where the overlap files are located, e.g. 'Overlap/';
    FileName   : {STRING}      Overlap file, which contains ALL X vs Y Pairs, then a heatmap matrix is constructed;
    Threshold  : {FLOAT}       Threshold used for the overlap calculation, for which the heatmap is generated,
                                        e.g. 0.2, has to match the ThresholdList of the Overlap file;
    Case       : {STRING}      <default density> 'conformational' or 'density', select if density or conformational overlap is displayed;
    AllPrject  : {BOOL}        <default False> True - Heatmap is  symmetric = overlap is projected on both groups X & Y;
                                               False- Heatmap is asymmetric = lower triangular projection on X, upper on Y;
    TrajExcept : {INT-LIST}    <default []>       possibility to EXCLUDE manually trajectories by deleting the 
                                                  Rows and Columns (starting from 1 to N) of the HeatMap
                                                    e.g. TrajExcept=[1,2] delete the first 2 trajectories resp. rows and columns;
OUTPUT:
    return HeatMap
    """
    #t1 = time.time()
    #-------
    SkipHead = 0
    with open('%s%s' % (OverlapDir, FileName), 'r') as INPUT:
        for line in INPUT:
            if len(line.split()) > 2 and line.split()[1] == 'ThresholdList':
                ThresholdList = [float(elem.replace(',','')) for elem in line[line.find('[')+1:line.find(']')].split()]
            if len(line.split()) == 0 or line[0] == '#':
                SkipHead += 1
            else:
                break
    #------- detects automatically number of Pairs in Overlap and create a Matrix
    NrOfTrajs = int( 1/2.+NP.sqrt(1/4.+2*len(NP.genfromtxt('%s%s' % (OverlapDir, FileName))[0,2:])/4) )
    HeatMap = NP.ones( (NrOfTrajs, NrOfTrajs) )
#### #### ####
# ERROR DETECTION
    GroupNrs = NP.genfromtxt('%s%s' % (OverlapDir, FileName), usecols=(0))
    if len(NP.unique(GroupNrs)) > 2:
        raise ValueError('There are more than two GroupNrs. Check your input!')
#### #### ####
# AllPrject = True
    if AllPrject:
      #--- RELATIVE        
        if Case == 'conformational':
            #--- SUM for both GroupNrs confO & TotFrames, THEN DIVIDE them
            HeatMap[NP.triu_indices(NrOfTrajs,1)] = \
                NP.divide(NP.add(NP.genfromtxt('%s%s' % (OverlapDir, FileName),
                                                  usecols=[0+4+4*elem for elem in range(NrOfTrajs*(NrOfTrajs-1)/2)],
                                                  skip_header=1+SkipHead+ThresholdList.index(Threshold)*2,
                                                  skip_footer=0+(len(ThresholdList)-ThresholdList.index(Threshold)-1)*2),
                                 NP.genfromtxt('%s%s' % (OverlapDir, FileName),
                                                  usecols=[0+4+4*elem for elem in range(NrOfTrajs*(NrOfTrajs-1)/2)],
                                                  skip_header=0+SkipHead+ThresholdList.index(Threshold)*2,
                                                  skip_footer=1+(len(ThresholdList)-ThresholdList.index(Threshold)-1)*2)),
                          NP.add(NP.genfromtxt('%s%s' % (OverlapDir, FileName),
                                                  usecols=[1+4+4*elem for elem in range(NrOfTrajs*(NrOfTrajs-1)/2)],
                                                  skip_header=1+SkipHead+ThresholdList.index(Threshold)*2,
                                                  skip_footer=0+(len(ThresholdList)-ThresholdList.index(Threshold)-1)*2),
                                 NP.genfromtxt('%s%s' % (OverlapDir, FileName),
                                                  usecols=[1+4+4*elem for elem in range(NrOfTrajs*(NrOfTrajs-1)/2)],
                                                  skip_header=0+SkipHead+ThresholdList.index(Threshold)*2,
                                                  skip_footer=1+(len(ThresholdList)-ThresholdList.index(Threshold)-1)*2)) )
      #--- ABSOLUTE
        else:
            HeatMap[NP.triu_indices(NrOfTrajs,1)] = NP.average(NP.genfromtxt('%s%s' % (OverlapDir, FileName),
                                                      usecols=[2+4*elem for elem in range(NrOfTrajs*(NrOfTrajs-1)/2)],
                                                      skip_header=0+SkipHead+ThresholdList.index(Threshold)*2,
                                                      skip_footer=0+(len(ThresholdList)-ThresholdList.index(Threshold)-1)*2), axis=0)
        #--- Make it symmetric to the Lower Triangular
        HeatMap[(NP.tril_indices(NrOfTrajs,-1))] = NP.transpose(HeatMap)[(NP.tril_indices(NrOfTrajs,-1))]    
                
#### #### ####
# SINGLES (AllPrject = False)
    else:
      #--- RELATIVE
        if Case == 'conformational':
            ## First
            HeatMap[NP.triu_indices(NrOfTrajs,1)] = NP.divide(\
                                          NP.genfromtxt('%s%s' % (OverlapDir, FileName),
                                              usecols=[0+4+4*elem for elem in range(NrOfTrajs*(NrOfTrajs-1)/2)],
                                              skip_header=1+SkipHead+ThresholdList.index(Threshold)*2,
                                              skip_footer=0+(len(ThresholdList)-ThresholdList.index(Threshold)-1)*2),
                                          NP.genfromtxt('%s%s' % (OverlapDir, FileName),
                                              usecols=[1+4+4*elem for elem in range(NrOfTrajs*(NrOfTrajs-1)/2)],
                                              skip_header=1+SkipHead+ThresholdList.index(Threshold)*2,
                                              skip_footer=0+(len(ThresholdList)-ThresholdList.index(Threshold)-1)*2) )
            HeatMap = NP.transpose(HeatMap)
            ## Second
            HeatMap[NP.triu_indices(NrOfTrajs,1)] = NP.divide(\
                                          NP.genfromtxt('%s%s' % (OverlapDir, FileName),
                                              usecols=[0+4+4*elem for elem in range(NrOfTrajs*(NrOfTrajs-1)/2)],
                                              skip_header=0+SkipHead+ThresholdList.index(Threshold)*2,
                                              skip_footer=1+(len(ThresholdList)-ThresholdList.index(Threshold)-1)*2),
                                          NP.genfromtxt('%s%s' % (OverlapDir, FileName),
                                              usecols=[1+4+4*elem for elem in range(NrOfTrajs*(NrOfTrajs-1)/2)],
                                              skip_header=0+SkipHead+ThresholdList.index(Threshold)*2,
                                              skip_footer=1+(len(ThresholdList)-ThresholdList.index(Threshold)-1)*2) )
      #--- ABSOLUTE
        else:
            ## First
            HeatMap[NP.triu_indices(NrOfTrajs,1)] = NP.genfromtxt('%s%s' % (OverlapDir, FileName),
                                                      usecols=[2+4*elem for elem in range(NrOfTrajs*(NrOfTrajs-1)/2)],
                                                      skip_header=1+SkipHead+ThresholdList.index(Threshold)*2,
                                                      skip_footer=0+(len(ThresholdList)-ThresholdList.index(Threshold)-1)*2)
            HeatMap = NP.transpose(HeatMap)
            ## Second
            HeatMap[NP.triu_indices(NrOfTrajs,1)] = \
                                        NP.genfromtxt('%s%s' % (OverlapDir, FileName),
                                                  usecols=[2+4*elem for elem in range(NrOfTrajs*(NrOfTrajs-1)/2)],
                                                  skip_header=0+SkipHead+ThresholdList.index(Threshold)*2,
                                                  skip_footer=1+(len(ThresholdList)-ThresholdList.index(Threshold)-1)*2)
    #-----------------------------------
    # DELETE ROWS/COLS of specific TrajNrs
    if TrajExcept != []:
        HeatMap = NP.delete(HeatMap, [elem-1 for elem in TrajExcept], axis=0)
        HeatMap = NP.delete(HeatMap, [elem-1 for elem in TrajExcept], axis=1)  
    return HeatMap

#######################################################
#--- DENDROGRAM
#################
def Plot_HeatMap_as_Dendro(OverlapDir, FileName, Threshold, Case='density', TrajExcept=[],
                           Labels=None, Colors=None, SaveDir=None, SaveName=None):
    """
v12.10.16
This function transforms the Heatmap_1vs1 to a hierarchically clustered dendrogram using average linkage.
- Colors is a dictionary, which has to fit the LabelNames, e.g.
    Colors = {'Label1':'g'} <-> Labels = ['Label1 1', 'Label1 2', 'Label1 3', 'Label1 4']
INPUT:
    OverlapDir : {STRING}      Directory, where the overlap files are located, e.g. 'Overlap/';
    FileName   : {STRING}      Overlap file, which contains ALL X vs Y Pairs, then a heatmap matrix is constructed;
    Threshold  : {FLOAT}       Threshold used for the overlap calculation, for which the heatmap is generated,
                                        e.g. 0.2, has to match the ThresholdList of the Overlap file;
    Case       : {STRING}      <default density> 'conformational' or 'density', select if density or conformational overlap is displayed;
    TrajExcept : {INT-LIST}    <default []>       possibility to EXCLUDE manually trajectories by deleting the 
                                                  Rows and Columns (starting from 1 to N) of the HeatMap
                                                    e.g. TrajExcept=[1,2] delete the first 2 trajectories;
    Labels     : {LIST}        <default None>     Label names for the Leaves, thus 
                                                    len(Labels) == total amount of leaves == total number of trajectories,
                                    IF COLOR IS NOT NONE, label PREFIX has to correspond to Colors, 
                                            e.g. Labels = ['Label1 1', 'Label1 2', 'Label1 3', 'Label1 4'];
    Colors     : {DICT}        <default None>     color specific label prefix, whereas 
                                                    len(Labels) == total amount of leaves == total number of trajectories,
                                    e.g. for 4 Trajectories, all leaves colored green - 
                                      Colors = {'Label1':'g'} <-> Labels = ['Label1 1', 'Label1 2', 'Label1 3', 'Label1 4'];
    SaveDir    : {STRING}                      Directory, where the PDF is stored, e.g. 'DendroGrams/';
    SaveName   : {STRING}                      save name, e.g. 'Molecule_Dendrogram_Specifications.pdf';
OUTPUT:
    """
    import matplotlib.pyplot as plt
    if SaveDir is not None and SaveName is not None and os.path.exists(SaveDir+SaveName):
        print 'Figure already exists\n%s%s' % (SaveDir, SaveName)
        return
    #---- DENDROGRAM
    Heat_mat = Generate_1vs1_Matrix(OverlapDir, FileName, Threshold, Case, True, TrajExcept)
    vals_dist = squareform(1-Heat_mat)
    Clusters = linkage(vals_dist, method='average')
    #---- PLOTTING
    fig = plt.figure(figsize=(13,4))
    ## subplot MARGINS
    plt.subplots_adjust(left=0.065, right=0.99, top=0.96, bottom=0.43)
    ## plot Dendrogram 
    ## labels=LABELS, 
    if Labels is None:
        DenDro    = dendrogram(Clusters, color_threshold=0.0, leaf_rotation=90, 
                               leaf_font_size=15, above_threshold_color='k')
    else:
        DenDro    = dendrogram(Clusters, color_threshold=0.0, labels=Labels, leaf_rotation=90, 
                               leaf_font_size=15, above_threshold_color='k')
    ## plot points to the splits
    for i, d in zip(DenDro['icoord'], DenDro['dcoord']):
        plt.plot(0.5*sum(i[1:3]), d[1], 'bo', ms=4)
    ## Y-axis & X-axis
    plt.grid(axis='y')
    plt.yticks(NP.arange(0,1.05,0.2), [1.0,0.8,0.6,0.4,0.2,0.0], fontsize=16);
    plt.ylabel('%s overlap' % Case, fontsize=19); plt.xlabel('trajectory', fontsize=19);
    plt.xticks(fontsize=16, va='top')
    ####
    if Colors is not None and Labels is not None:
        for LBL in plt.gca().get_xmajorticklabels():
            for Koi in NP.unique([len(elem) for elem in Colors.keys()]):
                if Colors.has_key(LBL.get_text()[0:Koi]):
                    LBL.set_color(Colors[LBL.get_text()[0:Koi]]) 
                    break
  ##########------- SAVE PDF ---------##########
    if SaveDir is not None and SaveName is not None:
        #### #### ####
        #---- generate Directories  
        Generate_Directories(SaveDir)
        plt.savefig(SaveDir+SaveName)  
        plt.close()

#######################################################
#--- OVERLAP vs TIME
#################
def Plot_Overlap_VS_Time(OverlapDir, OverlapList, Threshold, SimTimeList, TimeStep, Percentile1=25, Percentile2=25, 
                         Median=False, Interpolation='linear', LegendList=[], Title='', LegendNcols=1, 
                         SaveDir=None, SaveName=None, logX=False, LegendDens=True):
    """
v18.11.16
This function generates the plots 'Overlap vs simulation Time' for conformational & density overlap
- possibility to submit multiple OverlapMatrices to plot for instance multiple groups together
- each element of OverlapList MUST constain 'Start-End' which are replaced by the elements of SimTimeList, because
  Overlap files are calculated for different simulation times separately and must be merged in first place
- elements of SimTimeList define the (StartFrame, EndingFrame) tuples for the corresponding simulation time
- all Groups from one OverlapList are used, i.e. all K reference structures
- ERROR HANDLING: plots the first percentile and second percentile as error bars,
                  if the average value is outside, use the average as one limit
INPUT:
    OverlapDir    : {STRING}      Directory, where the Overlap is located, e.g. 'Overlap/';
    OverlapList   : {LIST}        List of Overlap filenames containing different cases, e.g. Pairs, different Groups,
                                  MUST CONTAIN 'Start-End' which is replaced by the corresponding element of SimTimeList
                                  e.g. ['Overlap_ALLvsALL_Start-End.txt' 'Overlap_AvsB_Start-End.txt'];
    Threshold     : {FLOAT}       Threshold used for the overlap calculation, for which the 'Overlap VS Time' is plotted,
                                        e.g. 0.2, has to match the ThresholdList of the Overlap file;
    SimTimeList   : {TUPLE-LIST}  (StartFrame,EndingFrame) tuples of the calculated simulation times, 
                                        e.g. [(0,100), (0,250), (0,500), (0,750), (0,1000), (0,1500), (0,2000)];
    TimeStep      : {FLOAT}       defines the step, 1 frame refers to TimeStep [ns], e.g. TimeStep = 0.01 means, 1 Frame = 10ps;
    Percentile1   : {INT}         <default 25> Value of the 1st percentile [in %], e.g. 25 for the 1st quartile;
    Percentile2   : {INT}         <default 75> Value of the 2nd percentile [in %], e.g. 75 for the 3rd quartile;
    Median        : {BOOL}        <default False>  if True, the median is plotted over all K reference trajs, else the Avg;
    Interpolation : {STRING}      <default linear> Interpolation method to find the percentile values,
                                                   e.g. 'linear', 'lower', 'higher', 'midpoint', 'nearest';
    LegendList    : {LIST}        <default []>  legend for the multiple overlap files in OverlapList, e.g. ['ALL', 'AvsB'];
    Title         : {STRING}      <default ''>  title specification of the plots, e.g. 'MoleculeName';
    LegendNcols   : {INT}         <default 1>   number of columns in the displayed Legend, to fit into the plot;
    SaveDir       : {STRING}      saving directory for the PDF, e.g. 'PDFs/';
    SaveName      : {STRING}      savename, e.g. 'OverlaPvsTime.pdf';
    logX          : {BOOL}        <default False> if True, the X axis is logarithmic, else not;
    LegendDens    : {BOOL}        <default True>  if True, the legend is located in the panel of the density overlap, else in the conformational overlap;
OUTPUT:
    stores the OverlapVStime.pdf
    """
    import matplotlib.pyplot as plt
    if SaveDir is not None and SaveName is not None and os.path.exists(SaveDir+SaveName):
        print 'Figure already exists\n%s%s' % (SaveDir, SaveName)
        return
    #----- ERROR DETECTION -----#
    if not NP.all([elem.find('Start-End')+1 for elem in OverlapList]):
        raise ValueError('All Overlap-files in <OverlapList> must include >Start-End< as StartFrame-EndingFrame '+\
                         'replacements! Check your input.\n%s' % OverlapList)
 #################################
     # StartTime | EndTime | densO1 | densO1_Perc1 | densO1_Perc2 | confO1 | confO1_Perc1 | confO1_Perc2 | ...
    OvT = Generate_OvT(OverlapDir, OverlapList, Threshold, SimTimeList, TimeStep,
                                   Percentile1, Percentile2, Median, Interpolation)
    if len(OvT[0,2:])/6 <=7:
        Color = ['b<', 'k>', 'r>', 'g>', 'm>', 'c>', 'y>'];
    else:
        Color = ['bs', 'ks', 'rs', 'gs', 'k<', 'r<', 'g<', 'k>', 'r>', 'g>', 
                 'b<', 'ms', 'cs', 'ys', 'b>', 'm<', 'c<', 'y<', 'm>', 'c>', 'y>'];
    #----- PLOTTING
    fig = plt.figure(figsize=(11,4))
    for RelAbs in [0,1]:  ## 0 = conformational; 1 = density
      #### suplot GRID / MARGINS
        plt.subplot2grid( (1,2), (0,RelAbs) ); 
        plt.subplots_adjust(left=0.095, right=0.99, bottom=0.20, top=0.9, wspace=0.025)
      #### PLOTTING
        plt.title('%s  %s overlap' % (Title, ['conformational','density'][RelAbs]), fontsize=25)
        if RelAbs == 1:
            for CaseIndex in range(0,len(OvT[0,2:]),6):
              #### CORRECT yerror if the value lies outside the error range, then is set to the upper/lower limit 
                YERR = [OvT[:,2+CaseIndex]-OvT[:,2+CaseIndex+1], OvT[:,2+CaseIndex+2]-OvT[:,2+CaseIndex]]
                YERR[0][YERR[0]<0] = 0; YERR[1][YERR[1]<0] = 0;
                plt.errorbar(NP.subtract(OvT[:,1],OvT[:,0]), OvT[:,2+CaseIndex], yerr=YERR, 
                             color=Color[(CaseIndex/6)%21][0], ms=7, ls='-', lw=0.5, marker=Color[(CaseIndex/6)%21][1], 
                             mfc='None', mec=Color[(CaseIndex/6)%21][0], mew=2, barsabove=False, capsize=7, capthick=1.5, 
                             alpha=0.8)
        else:
            for CaseIndex in range(0,len(OvT[0,2:]),6):
                YERR = [OvT[:,5+CaseIndex]-OvT[:,5+CaseIndex+1], OvT[:,5+CaseIndex+2]-OvT[:,5+CaseIndex]]
                YERR[0][YERR[0]<0] = 0; YERR[1][YERR[1]<0] = 0;
                plt.errorbar(NP.subtract(OvT[:,1],OvT[:,0]), OvT[:,5+CaseIndex], yerr=YERR,
                             color=Color[(CaseIndex/6)%21][0], ms=7, ls='-', lw=0.5, marker=Color[(CaseIndex/6)%21][1], 
                             mfc='None', mec=Color[(CaseIndex/6)%21][0], mew=2, alpha=0.8, barsabove=True, capsize=7, 
                             capthick=1.5);
      #### Xticks / Xlabel / Xlim
        if logX: plt.xscale('log'); 
        plt.xlim([(OvT[0,1]-OvT[0,0])-.01*(OvT[-1,1]-OvT[-1,0]), 1.05*(OvT[-1,1]-OvT[-1,0])])
        plt.xticks(fontsize=21); plt.grid()
        plt.xlabel('simulation time [ns]', fontsize=22); 
      #### Yticks / Ylabel / Ylim
        if RelAbs == 0: plt.yticks(fontsize=21); plt.ylabel('Overlap, r=%snm' % Threshold, fontsize=22);
        else:           plt.yticks(fontsize=0) 
        plt.ylim([-0.05,1.05]); 
      #### LEGEND
        if ((LegendDens and RelAbs == 1) or (not LegendDens and RelAbs==0)) and LegendList != []:
            plt.legend(LegendList, numpoints=1, ncol=LegendNcols, loc=0, framealpha=0.5, fontsize=17.5)
  ##########------- SAVE PDF ---------##########
    if SaveDir is not None and SaveName is not None:
        #### #### ####
        #---- generate Directories  
        Generate_Directories(SaveDir)
        plt.savefig(SaveDir+SaveName)
        plt.close()
#----- ----- ------ ------ ------- ----- 

##### ##### ##### ##### ##### ##### 
#----- GENERATE Overlap VS Time
def Generate_OvT(OverlapDir, OverlapList, Threshold, SimTimeList, TimeStep, Percentile1, Percentile2, Median=False, Interpolation='linear'):
    """
v18.11.16
- supporting function to merge/generate OverlaPvsTime NP.ndarray from Overlapfiles with different simulation times
- extracts the overlap for a certain threshold of different simulation time files and merge them in the ROW-dimension
- different groups are also extracted from different OverlapList elements and merged in the COLUMN-dimension
INPUT:
    OverlapDir    : {STRING}      Directory, where the Overlap is located, e.g. 'Overlap/';
    OverlapList   : {LIST}        List of Overlap filenames containing different cases, e.g. Pairs, different Groups,
                                  MUST CONTAIN 'Start-End' which is replaced by the corresponding element of SimTimeList
                                  e.g. ['Overlap_ALLvsALL_Start-End.txt' 'Overlap_AvsB_Start-End.txt'];
    Threshold     : {FLOAT}       Threshold used for the overlap calculation, for which the 'Overlap VS Time' is plotted,
                                        e.g. 0.2, has to match the ThresholdList of the Overlap file;
    SimTimeList   : {TUPLE-LIST}  (StartFrame,EndingFrame) tuples of the calculated simulation times, 
                                        e.g. [(0,100), (0,250), (0,500), (0,750), (0,1000), (0,1500), (0,2000)];
    TimeStep      : {FLOAT}       defines the step, 1 frame refers to TimeStep [ns], e.g. TimeStep = 0.01 means, 1 Frame = 10ps;
    Percentile1   : {INT}         Value of the 1st percentile [in %], e.g. 25 for the 1st quartile;
    Percentile2   : {INT}         Value of the 2nd percentile [in %], e.g. 75 for the 3rd quartile;
    Median        : {BOOL}        <default False>  if True, the median is plotted over all K reference trajs, else the Avg;
    Interpolation : {STRING}      <default linear> Interpolation method to find the percentile values,
                                                   e.g. 'linear', 'lower', 'higher', 'midpoint', 'nearest';
OUTPUT:
    return OvT {NP.NDARRAY} - StartTime | EndTime | densO1 | densO1_Perc1 | densO1_Perc2 | 
                                                    confO1 | confO1_Perc1 | confO1_Perc2 | ...
    """
###########
    ColLength = 2
    if len(OverlapList) > 1:
        for FF in OverlapList:
            for First,Snd in SimTimeList:
                if not os.path.exists('%s%s' % (OverlapDir, 
                                                FF.replace('Start', str(First)).replace('End',str(Snd)))):
                    raise NameError('The submitted\n\tOverlapFile=%s\ndoes not exist! Check your input\n%s\n%s\nwith\n%s' % \
                                    (FF.replace('Start', str(First)).replace('End',str(Snd)), OverlapDir, OverlapList, SimTimeList))
        #----- Loaded Overlap for all different simulation times
            ColLength += 6*len(NP.genfromtxt('%s%s' % (OverlapDir, 
                                FF.replace('Start', str(First)).replace('End',str(Snd))))[0,2:])/4
        OvT = NP.zeros( (len(SimTimeList), ColLength) )
        
###########
    OvT_Row = 0
    for StartFrame, EndingFrame in SimTimeList:
        if 'OvT' in locals():
            OvT[OvT_Row, 0] = StartFrame*TimeStep; OvT[OvT_Row, 1] = EndingFrame*TimeStep; 
        OvT_Col = 2
        for FileName in OverlapList:
        #----- Loaded Overlap for all different simulation times
            if not os.path.exists('%s%s' % (OverlapDir, FileName.replace('Start', str(StartFrame)).replace('End',str(EndingFrame)))):
                raise NameError('%s%s\ndoes not exist! Check your input!' % (OverlapDir, FileName.replace('Start', str(StartFrame)).replace('End',str(EndingFrame))))
            tempO = NP.genfromtxt('%s%s' % (OverlapDir, FileName.replace('Start', str(StartFrame)).replace('End',str(EndingFrame))))
            if 'OvT' not in locals():
                OvT = NP.zeros( (len(SimTimeList), 2+6*len(OverlapList)*len(tempO[0,2:])/4) ) # StartTime | EndTime | densO1 | confO1 | densO2 | confO2 | ...
                OvT[OvT_Row, 0] = StartFrame*TimeStep; OvT[OvT_Row, 1] = EndingFrame*TimeStep; 

            tempO = tempO[tempO[:,1]==Threshold]
        #-----
            if len(tempO[:,0]) == 0:
                raise ValueError('The submitted Overlap files do not include the submitted Threshold = %s!' % Threshold)
        #-----
            for Koi in range(0,6*len(tempO[0,2:])/4,6):
                ## densO
                if Median:
                    OvT[OvT_Row, OvT_Col+Koi] = NP.median(tempO[tempO[:,1]==Threshold][:,2+4*Koi/6])
                else:
                    OvT[OvT_Row, OvT_Col+Koi] = NP.average(tempO[tempO[:,1]==Threshold][:,2+4*Koi/6])
                ## densO Percentile1
                OvT[OvT_Row, OvT_Col+Koi+1] = NP.percentile(tempO[tempO[:,1]==Threshold][:,2+4*Koi/6], Percentile1,
                                                            interpolation=Interpolation)
                ## densO Percentile2
                OvT[OvT_Row, OvT_Col+Koi+2] = NP.percentile(tempO[tempO[:,1]==Threshold][:,2+4*Koi/6], Percentile2,
                                                            interpolation=Interpolation)
                ## confO
                if Median:
                    OvT[OvT_Row, OvT_Col+Koi+3] = NP.median(NP.divide(tempO[tempO[:,1]==Threshold][:,4+4*Koi/6],
                                                                      tempO[tempO[:,1]==Threshold][:,5+4*Koi/6]))
                else:
                    OvT[OvT_Row, OvT_Col+Koi+3] = NP.divide(NP.sum(tempO[tempO[:,1]==Threshold][:,4+4*Koi/6]),
                                                            NP.sum(tempO[tempO[:,1]==Threshold][:,5+4*Koi/6]))
                ## confO Percentile1
                OvT[OvT_Row, OvT_Col+Koi+4] = NP.percentile(NP.divide(tempO[tempO[:,1]==Threshold][:,4+4*Koi/6],
                                                                      tempO[tempO[:,1]==Threshold][:,5+4*Koi/6]),
                                                        Percentile1, interpolation=Interpolation)
                ## confO Percentile1
                OvT[OvT_Row, OvT_Col+Koi+5] = NP.percentile(NP.divide(tempO[tempO[:,1]==Threshold][:,4+4*Koi/6],
                                                                      tempO[tempO[:,1]==Threshold][:,5+4*Koi/6]),
                                                        Percentile2, interpolation=Interpolation)
        #-----
            OvT_Col += 6*len(tempO[0,2:])/4
        OvT_Row += 1
 #--- return OvT: StartTime | EndTime | densO1 | densO1_Perc1 | densO1_Perc2 | confO1 | confO1_Perc1 | confO1_Perc2 | ...
    return OvT

#######################################################
#--- OVERLAP vs NUMBER of CLUSTERS
#################
def Plot_Overlap_VS_Cluster(OverlapDir, OverlapList, Threshold, ClusterDir, Centers_GLOBAL_singles, Case='density', 
                          XLIM=None, YLIM=None, LEGEND=None, Percentile1=25, Percentile2=75, Median=False, 
                          Interpolation='linear', SaveDir=None, SaveName=None,
                          Symbols=['bs', 'ks', 'rs', 'gs', 'ko', 'ro', 'go', 'k<', 'r<', 'g<', 'g<', 'm<', 'c<', 'y<']):
    """
v21.11.16
    This function plots the overlap as a function of the cluster of all combined trajectories which were used for the 
    overlap calculation
    - the clustering must contain ALL trajectories with the same trajectory numbering as in the overlap files
    - Symbols follow the matplotlib plot logic, which contains 2 characters: 1st = color; 2nd = marker
INPUT:
    OverlapDir             : {STRING}     Directory, where the Overlap is located, e.g. 'Overlap/';
    OverlapList            : {LIST}       List of Overlap filenames containing different cases, 
                                          e.g. Pairs, different Groups, which are then plotted in one plot
                                          e.g. ['Overlap_ALLvsALL.txt' 'Overlap_AvsB.txt'];
    Threshold              : {FLOAT}      Threshold used for the overlap and clustering calculation, 
                                            e.g. 0.2, has to match the ThresholdList of the Overlap and Clustering file;
    ClusterDir             : {STRING}     Directory, where the clustering files are located, e.g. 'Clustering/';
    Centers_GLOBAL_singles : {STRING}     Clustering file containing the global partitioning centers, 
                                            e.g. 'Cluster_Centers_GLOBAL_singles.txt';
    Case                   : {STRING}     <default density> density or conformational for the corresponding overlap;   
    XLIM                   : {FLOAT-LIST} <default None>    defines the x-limits for the number of clusters;
    YLIM                   : {FLOAT-LIST} <default None>    defines the y-limits for the overlap;
    LegendList             : {LIST}       <default None>    Legend for the single elements of the OverlapList1, 
                                                            e.g. ['ALL','AvsB'];
    Percentile1            : {INT}        <default 25>      Value of the 1st percentile [in %], e.g. 25 for the 1st quartile;
    Percentile2            : {INT}        <default 75>      Value of the 2nd percentile [in %], e.g. 75 for the 3rd quartile;
    Median                 : {BOOL}       <default False>   if True, the median is plotted over all K reference trajs, else the Avg;
    Interpolation          : {STRING}     <default linear>  Interpolation method to find the percentile values,
                                                            e.g. 'linear', 'lower', 'higher', 'midpoint', 'nearest';
    SaveDir                : {STRING}     <default None>    saving directory for the PDF, e.g. 'PDFs/';
    SaveName               : {STRING}     <default None>    savename, e.g. 'OverlaPvsCluster.pdf';
    Symbols                : {LIST}       <default bs ks rs gs ko ro go k^ r^ g^> color and marker for the symbols,
                                                1st character = color, 2nd character = marker,
                                                b = blue, k = black, r = red, g = green, ...,
                                                s = square, o = circle, ^ = triangle, * = star, h = hexagon, ...;
OUTPUT:
    plots the pareto plot overlap (dens/conf) vs. the number of clusters which are found by all trajectories involved
    in the overlap calculation [L of O(K,L;r)]
    """
    import matplotlib.pyplot as plt
    if Case != 'density':
        Case = 'conformational'
    SkipHead = 0
    with open('%s%s' % (ClusterDir, Centers_GLOBAL_singles), 'r') as INPUT:
        for line in INPUT:
            if len(line.split()) == 0 or line[0] == '#':
                SkipHead += 1
            else:
                break
  #---- load Overlap values
    Overlap = Generate_OvR(OverlapDir, OverlapList, Percentile1, Percentile2, Median, Interpolation)
    Overlap = Overlap[Overlap[:,0]==Threshold]
  #---- calculate combined number of clusters
    NrClust = NP.zeros( len(OverlapList) )
    Index   = 0
  #---- load ThresholdList & NrOfTrajs
    with open('%s%s' % (ClusterDir, Centers_GLOBAL_singles), 'r') as INPUT:
        for line in INPUT:
            if len(line.split()) > 2 and line.split()[0] == '#' and line.split()[1] == 'ThresholdList':
                ThresholdList = [float(elem.replace(',','')) for elem in line[line.find('[')+1:line.find(']')].split()]
            elif len(line.split()) > 2 and line.split()[0] == '#' and line.split()[1] == 'Threshold':
                ThresholdList = [float(line.split()[-1])]
            elif len(line.split()) > 2 and line.split()[0] == '#' and line.split()[1] == 'TrajNameList':
                NrOfTrajs = len([elem.replace('\'','').replace(',','') \
                                for elem in line[line.find('[')+1:line.find(']')].split()])
                break
  #---- ERROR detection
    if 'ThresholdList' not in locals():
        raise ValueError('ThresholdList is not defined in\n\t%s%s\nAdd used >>ThresholdList = [...]<<.' % \
                                                (ClusterDir, Centers_GLOBAL_singles))
    if 'NrOfTrajs' not in locals():
        raise ValueError('TrajNameList is not defined in\n\t%s%s\nAdd used >>TrajNameList = [...]<<, to define the trajectories which were used.' % \
                                                (ClusterDir, Centers_GLOBAL_singles))
    if ThresholdList.count(Threshold) == 0:
        raise ValueError('The submitted\n\t>>Threshold=%s<<\nis not present in the\n\t>>ThresholdList = %s\nof\n\t%s%s' %\
                         (Threshold, ThresholdList, ClusterDir, Centers_GLOBAL_singles))
    if len(Overlap[:,0]) == 0:
        raise ValueError('The submitted\n\t>>Threshold=%s<<\nis not present in the\n\t>>ThresholdList = %s\nof all overlap files\n\t%s%s' %\
                         (Threshold, ThresholdList, OverlapDir, OverlapList))    
 #-----######
    for File in OverlapList:
      #---- extract comparelists for trajectory numbers
        with open('%s%s' % (OverlapDir, File), 'r') as INPUT:
            for line in INPUT:
                if len(line.split()) > 2 and line.split()[1] == 'CompareList':
                    CompareList = ast.literal_eval(line.split(' = ')[1].replace('\n',''))
                    break
      #---- each comparelist generates (multiple) cases
        for Single in CompareList:
            TrajNrList = NP.concatenate(Single)
            Centers = []
            for Traj in TrajNrList:
                temp = NP.genfromtxt('%s%s' % (ClusterDir, Centers_GLOBAL_singles),
                    skip_header=SkipHead+(Traj-1)+NrOfTrajs*((ThresholdList.index(Threshold))),
                    skip_footer=len(ThresholdList)*NrOfTrajs-((Traj)+NrOfTrajs*((ThresholdList.index(Threshold)))))
                Centers = NP.concatenate( (Centers, temp[3:]) )
            NrClust[Index] = len(NP.unique(Centers))
            Index += 1
#########################
    ###------ PLOT
    fig = plt.figure(figsize=(7,6))
    plt.subplot2grid( (1,1), (0,0) );
    plt.subplots_adjust(left=0.17, right=0.95, top=0.92, bottom=0.15, wspace=0.15)
  #-------
    for CaseA in range(len(OverlapList)):
        if Case=='density': Plus = 0;
        else:               Plus = 3;
        YERR = [Overlap[0,[1+Plus+6*CaseA]]-Overlap[0,[2+Plus+6*CaseA]], 
                Overlap[0,[3+Plus+6*CaseA]]-Overlap[0,[1+Plus+6*CaseA]]]
        YERR[0][YERR[0]<0] = 0; YERR[1][YERR[1]<0] = 0;
        plt.errorbar(NrClust[CaseA], Overlap[0,1+Plus+6*CaseA], yerr=YERR, color=Symbols[CaseA%len(Symbols)][0], 
                     marker=Symbols[CaseA%len(Symbols)][1], mec=Symbols[CaseA%len(Symbols)][0], 
                     mfc='none', mew=1.5, ms=8, capsize=7, capthick=1.5);
  #--------
    if YLIM is None: YLIM = ([-0.05, 1.05]);
    if XLIM is None: XLIM = [min(NrClust)-10, max(NrClust)+10]
    plt.xticks(fontsize=21); plt.xlim(XLIM); plt.yticks(fontsize=21); plt.ylim(YLIM); plt.grid(); 
    plt.xlabel('number of clusters', fontsize=22); plt.ylabel('%s overlap' % Case, fontsize=22)
    if LEGEND is not None:
        plt.legend(LEGEND, numpoints=1, fontsize=19, loc=0, ncol=1,framealpha=0.65)
    if SaveName != None and SaveDir != None:
        plt.savefig(SaveDir+SaveName)
        plt.close()

#################

def Generate_Directories(SaveDir):
    """
v12.10.16
    Helper function to generate all necessary directories

INPUT:
    SaveDir  :  {STRING}  Directory or Directories separated by '/', which should be generated if they do not exist, e.g. 'SaveDir1/SaveDir2/';
OUTPUT:
    """
    #---- generate Directories  
    if SaveDir is not None:
        SaveDir = SaveDir.replace('None','')
        for Kai in range(1,len(SaveDir.split('/'))):
            if not os.path.exists('/'.join((SaveDir).split('/')[:Kai])):
                os.mkdir('/'.join((SaveDir).split('/')[:Kai]))
              #---- ENSURE, that the directory is generated, wait for network latency  
                EmergencyCancel = 0
                while not os.path.exists('/'.join((SaveDir).split('/')[:Kai])) or EmergencyCancel > 10:
                    time.sleep(5)
                    EmergencyCancel += 1
                if EmergencyCancel > 10:
                    raise NameError('The directory >>%s<< could not be generated within 50 seconds.\nTry to generate it by hand before the calculation!' % \
                                    ('/'.join((SaveDir).split('/')[:Kai])))

#######################################################
#--- MAIN()
#################

def cmdlineparse():
    Parser = ArgumentParser(description="arguments")
    Parser.add_argument("-module", dest="module", required=True, help="module name", metavar="<module>")
    Parser.add_argument("-in", dest="input", required=True, help="module name or config file", metavar="<in>")
    Parser.add_argument("-out", dest="output", required=False, help="output name for config file", metavar="<out>")
    args=Parser.parse_args()
    return args

##### ##### #####

def GenerateIn(Module, FileName):
    """
v12.10.16
This function generates the config-files for all modules with possible examples, default values and explanations.

INPUT:
    Module    : {STRING} Name of the module, e.g. <Generate_EventCurve>
    FileName  : {STRING} save name of the config-file + possilbe directory, e.g. <Dir/EventCurves.in>
OUTPUT:
    """
    ## CHECK, if FileName contains a directory, which does not exist. Then, generate it:
    Generate_Directories(FileName)
    #---------
    MODULE = getattr(sys.modules[__name__], "%s" % Module)
    DocString = MODULE.func_doc.split('INPUT:')[1].split('OUTPUT:')[0].replace('\n','')
    #------------------------------------------------------
    LJUST = 31
    TransformDict = {'LIST': '"Name1 Name2 Name3 Name4"',
                     'INT-LIST': '"2 4 65 12 4 22"',
                     'STRING': '(non-directory) "TEXT" | (directory) "TEXT/"',
                     'FLOAT-LIST': '"0.1 0.2 0.3 0.4"',
                     'INT': '"502"',
                     'FLOAT': '"203.12"',
                     'TYPE': '"float32" or "float64"',
                     'STRING/LIST': '(STRING) "Name" | (LIST) "Name1 Name2 Name3"',
                     'LIST,TUPLE,LIST': '"[([1],[3]), ([1,2],[3,4]), ([1,2],[3])]"',
                     'FLOAT/INT': '(FLOAT) "0.1" | (INT) "3"',
                     'BOOL': '"False" or "True"',
                     'DICT': '{(Key1,Value1), (Key2,Value2)}',
                     'TUPLE-LIST': '"[(0,100), (0,500), (0,1000)]"'
                    }
    #------------------------------------------------------
    InfoText = """
    # Config-File for the module <%s()>. Ensure, that every parameter is set with the certain format
    # given as an example (WITHIN ""). All optional parameters are initialized with their default parameters.
    """ % (Module)
    #------------------------------------------------------
    with open(FileName, 'w') as OUTPUT:
        OUTPUT.write('#### #### ####%s#### #### ####\n' % InfoText)
        OUTPUT.write('##  ##  ##  ##  ##\n### DESCRIPTION: ###\n# %s\n##  ##  ##  ##  ##\n' % \
                     MODULE.func_doc.split('INPUT:')[0].replace('\n','\n# '))
        for KuH in DocString.split(';')[0:-1]:
            OUTPUT.write('#-------------------------------------------\n')
            OUTPUT.write('# %s\n' % (' '.join(KuH.split('}')[1].split())))
            Param = KuH.split(':')[0].replace(' ','')
            Type  = KuH.split('{')[1].split('}')[0]
            DEFAULT = '' if KuH.find('<default') == -1 else KuH.split('<default ')[1].split('>')[0]
            
            OUTPUT.write(('%s = "%s"' % \
                 (Param, DEFAULT)).ljust(LJUST)+\
                         (' # <%s> format example: %s\n' % (Type, TransformDict[Type])))

##### ##### #####

def ReadConfigFile(FileName):
    """
v09.09.16
    FileName  : {STRING} save name of the config-file + possilbe directory, e.g. <Dir/EventCurves.in>
    """
    INPUT = []
    
    with open(FileName, 'r') as FILE:
        for line in FILE:
            if line.split()[0][0] != '#':
                String = line.split('"')[1]
                Type   = line.split('<')[1].split('>')[0]
               #---- ERROR DETECTION
                if line.split('#')[0].find('"') == -1:
                    raise ValueError('The variables must be enclosed by quotes "", check the config file!\n%s:\n\t%s =' % \
                                     (FileName, line.split()[0]))
                elif TransformConfigFile(String, Type) == 'ERROR':
                    raise ValueError('Not all variables are assigned, check the config file!\n%s:\n\t%s = %s' % \
                                     (FileName, line.split()[0], String))
               #--------------------
                INPUT.append(TransformConfigFile(String, Type))
    return INPUT   

##### ##### #####

def TransformConfigFile(String, Type):
    """
v07.09.16    
    """
    if String == "":
        return 'ERROR'
    elif String == "None":
        return None
    elif String == '[None]':
        return [None]
    elif String == 'NP.infty':
        return NP.infty
    elif String == '[]':
        return []
    elif String == '\'\'':
        return ''
  #---------------  
    elif Type == 'LIST':
        return String.split()
    elif Type == 'INT-LIST':
        try:
            return [int(elem) for elem in String.split()]
        except ValueError:
            return ast.literal_eval(String)
    elif Type == 'FLOAT-LIST':
        try:
            return [float(elem) for elem in String.split()]
        except ValueError:
            return ast.literal_eval(String)
    elif Type == 'STRING/LIST':
        temp = String.split()
        if len(temp) == 1:
            return temp[0]
        else:
            return temp
    elif Type == 'BOOL':
        return String == 'True'
  #---------------
    elif Type == 'LIST,TUPLE,LIST' or Type == 'INT' or Type == 'FLOAT' or \
         Type == 'FLOAT/INT' or Type == 'DICT' or Type == 'TUPLE-LIST':
        return ast.literal_eval(String)
  #---------------
    else:
        return String

##### ##### #####

## v09.09.16
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print """
usage example:
    python PySamplingQuality.py -module GenerateIn           -in Generate_EventCurves -out EventData.in
    python PySamplingQuality.py -module Generate_EventCurves -in EventData.in
              """
    else:
        Args = cmdlineparse()
        Function = getattr(sys.modules[__name__], Args.module)
        if Args.module == 'GenerateIn':
            Function(Args.input, Args.output)
        else:
            INPUT = ReadConfigFile(Args.input)
            if ['Generate_EventCurves', 'determineR_using_RMSD_distributions', 'Generate_Clustering'].count(Args.module) == 1:
                INPUT.append('')
            Function(*INPUT)
