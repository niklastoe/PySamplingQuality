
from argparse import ArgumentParser
## String to literal
import ast

#####################################################################################

def Generate_Directories(SaveDir):
    """
v15.02.17
    Helper function to generate all necessary directories

INPUT:
    SaveDir  :  {STRING}  Directory or Directories separated by '/', which should be generated if they do not exist, e.g. 'SaveDir1/SaveDir2/';
OUTPUT:
    """
    #---- generate Directories  
    if SaveDir is not None:
        SaveDir = SaveDir.replace('None','')
        for Kai in xrange(1,len(SaveDir.split('/'))):
            if not os.path.exists('/'.join((SaveDir).split('/')[:Kai])) and '/'.join((SaveDir).split('/')[:Kai]) != '':
                os.mkdir('/'.join((SaveDir).split('/')[:Kai]))
              #---- ENSURE, that the directory is generated, wait for network latency  
                EmergencyCancel = 0
                while not os.path.exists('/'.join((SaveDir).split('/')[:Kai])) or EmergencyCancel > 10:
                    time.sleep(5)
                    EmergencyCancel += 1
                if EmergencyCancel > 10:
                    raise NameError('The directory >>%s<< could not be generated within 50 seconds.\nTry to generate it by hand before the calculation!' % \
                                    ('/'.join((SaveDir).split('/')[:Kai])))        

#####################################################################################

def GenerateIn(Module, FileName):
    """
v30.03.17
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
                     'LIST,LIST': '"[[1,2,3], [5]]"',
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
v30.03.17    
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
    elif Type == 'LIST,TUPLE,LIST' or Type == 'LIST,LIST' or Type == 'INT' or Type == 'FLOAT' or \
         Type == 'FLOAT/INT' or Type == 'DICT' or Type == 'TUPLE-LIST':
        return ast.literal_eval(String)
  #---------------
    else:
        return String     

##### ##### #####

def cmdlineparse():
    Parser = ArgumentParser(description="arguments")
    Parser.add_argument("-module", dest="module", required=True, help="module name", metavar="<module>")
    Parser.add_argument("-in", dest="input", required=True, help="module name or config file", metavar="<in>")
    Parser.add_argument("-out", dest="output", required=False, help="output name for config file", metavar="<out>")
    args=Parser.parse_args()
    return args
