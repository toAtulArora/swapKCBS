#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[1]:


import numpy as np
import sympy as sp
import sympy.physics.quantum as sq
import sympy.physics.quantum.qubit as sqq
import sympy.physics.quantum.gate as sqg

import cvxpy as cp

import matplotlib.pyplot as plt
import networkx as nx

sp.init_printing(use_latex='mathjax')
from IPython.display import display
from random import random
from random import seed

import pickle
import dill
dill.settings['recurse'] = True


#for pretty printing of matrices
import pandas as pd

import os


# ## idealFid

# In[2]:


class idealFid:
    
    def __init__(s):
        P1 = s.proj(1,5)
        P2 = s.proj(2,5)
        P3 = s.proj(3,5)
        P4 = s.proj(4,5)
        P5 = s.proj(5,5)
        s.P = [P1,P2,P3,P4,P5]
        s.terms = 5

    def proj(s,j,n):
        pi=np.pi
        temp1 = np.cos(pi/n)/(1+np.cos(pi/n))
        temp2 = 1-temp1
        temp3 = j*pi*(n-1)/n
        vec = np.asarray([np.sqrt(temp1),np.sqrt(temp2)*np.sin(temp3), np.sqrt(temp2)*np.cos(temp3)])
        mat = np.outer(vec,vec)
        return mat
    
    
    def idealVal(s,projList):
        op = s.P[projList[0]]
        for iProj in projList[1:]:
            op = op@s.P[iProj]
        return op[0,0]
    


# ## Extending braket functionality to include orthogonality

# In[3]:


class onStruct:
    G = nx.Graph()

class onKet(sq.Ket,onStruct):
    def _eval_innerproduct(self,bra,**hints):
        #Check if the innerproduct is with yourself, then set 1
        if sq.Dagger(bra) == self:
            return sp.Integer(1)
        #Else, if the innerproduct is with a neighbor from the graph, set 0
        elif sq.Dagger(bra) in G[self]:
            return sp.Integer(0)
        #Else, do nothing; not enough information
    
    @classmethod
    def set_on_rel(cls,givenG):
        cls.G=givenG
    
    @classmethod
    def dual_class(self):
        return onBra

    
class onBra(sq.Bra,onStruct):
    @classmethod
    def dual_class(self):
        return onKet


# In[4]:


class bKet(sq.Ket):
    
    #total number of kets (automatically increases as new instances are initialised)
    totalKets=0
    
    #this method automates the increase
    @classmethod
    def _eval_args(cls,args):
        #validate input type
        if not isinstance(args[0],int):
            raise ValueError("Integer expected in the argument, got: %r"%args[0])

        #if a ket |5> is initialised, the dimension is assumed to be 6 since we have |0>,|1>,...|5>
        if int(args[0])+1 > cls.totalKets:
            cls.totalKets=args[0]+1
        
        #pass control back to the base class
        return sq.Ket._eval_args(args) 
    
    #Could not find a way of using just one function; need to go through the two
    #based on a combination of the qubit implementation and the 
    def _represent_default_basis(s,**options):
        return s._represent_bKet(None,**options) #syntax taken from the Qubit library
    
    def _represent_bKet(s,basis,**options): #_represent_default_basis
        a=np.zeros(s.__class__.totalKets,dtype=int)
        a[s.label[0]]=1
        return sp.Matrix(a)
    
    @classmethod
    def set_dimension(cls,arg):
        cls.totalKets=arg
    
    def _eval_innerproduct(self,bra,**hints):
        #Check if the innerproduct is with yourself, then set 1
        if sq.Dagger(bra) == self:
            return sp.Integer(1)
        #Else, if the innerproduct is with a neighbor from the graph, set 0
        else:
            return sp.Integer(0)
        #Else, do nothing; not enough information
    

    @classmethod
    def dual_class(self):
        return bBra

    
class bBra(sq.Bra):
    # @classmethod
    # def _eval_args(cls,args):
    #     return args
    
    @classmethod
    def dual_class(cls):
        return bKet


# ## Extending tensor product functionality — tdsimp and tsimp

# In[5]:



def powerDrop(expr):
    if isinstance(expr,sp.Pow): #TODO: make sure the base is not too complex
        # print("PowerEncountered")
        if expr.exp>=2:
            # print("glaba")
            # display(expr.base)
            _=sq.qapply(sp.Mul(expr.base,expr.base))
            if expr.exp>2:
                return powerDrop(_*sp.Pow(expr.base,expr.exp-2))
            else:
                return _
        else:
            return expr #autoDropDim(sp.Mul(expr.base,expr.base))
    else:
        if expr.has(sp.Pow):
            #if it is a sum or a product, run this function for each part and then combine the parts; return the result
            if isinstance(expr,sp.Mul) or isinstance(expr,sp.Add) or isinstance(expr,sq.TensorProduct):
                new_args=[] #list(expr.args)
                for _ in expr.args:
                    new_args.append(powerDrop(_))
                if isinstance(expr,sp.Mul):        
                    return sp.Mul(*new_args)
                elif isinstance(expr,sp.Add):
                    return sp.Add(*new_args)  
                elif isinstance(expr,sq.TensorProduct):
                    return sq.TensorProduct(*new_args)  

            else:
                return expr
            #There would be no else here because tensor product simp would have removed that part
        else:
            return expr        
    
def autoDropDim(expr):
    #print("Expression")
    #if isinstance(expr,sp.Mul):
        #print("type:multiplier")
    #display(expr)
    
    
    if isinstance(expr,sq.TensorProduct):
        new_args=[]
        for _ in expr.args:
            #display(_)
            #print(type(_))
            if _ != sp.Integer(1):
            #if not isinstance(_,core.numbers.One):
                new_args.append(_)
        #print("TensorProduct with %d non-ones in the tensor product"%len(new_args))
        if(len(new_args)==0):
            return sp.Integer(1)
        else:
            return sq.TensorProduct(*new_args)
    else:
        if expr.has(sq.TensorProduct):
            #if it is a sum or a product, run this function for each part and then combine the parts; return the result
            if isinstance(expr,sp.Mul) or isinstance(expr,sp.Add):
                new_args=[] #list(expr.args)
                for _ in expr.args:
                    new_args.append(autoDropDim(_))
                if isinstance(expr,sp.Mul):        
                    return sp.Mul(*new_args)
                elif isinstance(expr,sp.Add):
                    return sp.Add(*new_args)  
                
            #There would be no else here because tensor product simp would have removed that part
        else:
            return expr #when the expression is just an integer or some such


        
def tsimp(e,pruneMe=True):
    res=sq.qapply(powerDrop(sq.tensorproduct.tensor_product_simp(sq.qapply(e)).doit()))
    if pruneMe:
        return prune(res)
    else:
        return res

def tdsimp(e,pruneMe=True):
    res=autoDropDim(sq.qapply(powerDrop(autoDropDim(sq.tensorproduct.tensor_product_simp(sq.qapply(e)).doit()))))
    if pruneMe:
        return prune(res)
    else:
        return res
    #return autoDropDim(sq.tensorproduct.tensor_product_simp_Mul(e).doit())
    #return autoDropDim(sq.tensorproduct.tensor_product_simp_Mul(sq.qapply(e)).doit())
    #return autoDropDim(sq.tensorproduct.tensor_product_simp(e).doit())


# ## FindCoeff was unreliable

# In[6]:


# depth=0
# depthThresh=20

def findCoeff(obj,lett):
#     global depth
    
#     print("Parent object:")
#     display(obj)
    
    if(obj==None):
#         print("coefficent is zero")
        return 0
    elif not (isinstance(obj,sp.Mul) or isinstance(obj,sp.Add)):        
        #the coefficient may be one but we couldn't see it earlier
        return obj.coeff(lett)
    else:
        #try to find the coefficient directly        
        result=obj.coeff(lett)
        #Did not work? 
        if(result==0):
            #try for each segment
            for _ in obj.args:
#                 print("child:")
#                 display(_)
                result=_.coeff(lett)
#                 print("coefficient of child:",result)
                
                #still did not work?
                if(result==0):
                    #try recursing
                    result = findCoeff(_,lett)    
                
                #found? Stop searching
                if(result!=0):
                    break
                    #return result

        #return whatever was found | worked
        return result


# ## Prune

# In[7]:


def prune(expr,thr=10,remNum=False):
    if isinstance(expr,sp.Number): 
        if remNum==False:
            if sp.Abs(expr)<10**(-thr):
                return sp.Integer(0)
            else:
                return expr
        else:
            return sp.Integer(1)
    else:
        if expr.has(sp.Number):
            #if it is a sum or a product, run this function for each part and then combine the parts; return the result
            if isinstance(expr,sp.Mul) or isinstance(expr,sp.Add) or isinstance(expr,sq.TensorProduct):
                new_args=[] #list(expr.args)
                for _ in expr.args:
                    new_args.append(prune(_,thr,remNum))
                if isinstance(expr,sp.Mul):        
                    return sp.Mul(*new_args)
                elif isinstance(expr,sp.Add):
                    return sp.Add(*new_args)  
                elif isinstance(expr,sq.TensorProduct):
                    return sq.TensorProduct(*new_args)  

            else:
                return expr
            #There would be no else here because tensor product simp would have removed that part
        else:
            return expr        

# test=(A[0]*2)
# test.has(sp.Number)
# prune(test,remNum=True)


# ## Power to mul

# In[8]:


# From: https://stackoverflow.com/questions/28117770/sympy-multiplications-of-exponential-rather-than-exponential-of-sum
# Thankfully I didn't have to sit and write this!

def pow_to_mul_(expr):
    """
    Convert integer powers in an expression to Muls, like a**2 => a*a.
    """
    pows = list(expr.atoms(sp.Pow))
    if any(not e.is_Integer for b, e in (i.as_base_exp() for i in pows)):

        raise ValueError("A power contains a non-integer exponent")
    repl = zip(pows, (sp.Mul(*[b]*e,evaluate=False) for b,e in (i.as_base_exp() for i in pows)))
    return expr.subs(repl)


# In[9]:


## Not very well written

def pow_to_mul(expr):
    if isinstance(expr,sp.Pow): 
        return pow_to_mul_(expr)
    else:
        try:
            if expr.has(sp.Pow):
                #if it is a sum or a product, run this function for each part and then combine the parts; return the result
                if isinstance(expr,sp.Mul) or isinstance(expr,sp.Add):
                    new_args=[] #list(expr.args)
                    for _ in expr.args:
                        new_args.append(pow_to_mul(_))

                    if isinstance(expr,sp.Mul):        
                        return sp.Mul(*new_args,evaluate=False)
                    elif isinstance(expr,sp.Add):
                        return sp.Add(*new_args)
                else:
                    return expr
            else:
                return expr      
        except:
            return expr


# In[10]:


def mul_to_single_mul(expr):
    if isinstance(expr,sp.Mul): 
        new_args=[]
        for _ in expr.args:
            if isinstance(_,sp.Mul):
                for __ in _.args:
                    new_args.append(__)
            elif expr.has(sp.Mul):
                new_args.append(mul_to_single_mul(_)) #If inside a mul, there is another mul, extract it
            else:
                new_args.append(_)  #if it is something entirely different, bring it along as it is
        
        return sp.Mul(*new_args,evaluate=False) #put the new arguments into a single multiply and return
    
    else:
        try:
            if expr.has(sp.Mul):
                #if it is a sum, run this function for each part and then combine the parts; return the result
                if isinstance(expr,sp.Add):
                    new_args=[] #list(expr.args)
                    for _ in expr.args:
                        new_args.append(mul_to_single_mul(_))
                    if isinstance(expr,sp.Add):
                        return sp.Add(*new_args)
                else:
                    return expr
            else:
                return expr      
        except:
            return expr


# In[11]:


def pow_to_single_mul(expr):
    return mul_to_single_mul(pow_to_mul(expr))


# ## Reverse

# In[12]:



def rev(expr):
    if isinstance(expr,sp.Mul):
        newargs=list(expr.args)
        newargs.reverse()
        return sp.Mul(*newargs)
    else:
        if expr.has(sp.Mul):
            #if it is a sum  run this function for each part and then combine the parts; return the result
            if isinstance(expr,sp.Add):
                new_args=[] #list(expr.args)
                for _ in expr.args:
                    new_args.append(rev(_))                
                return sp.Add(*new_args)
            else:
                return expr
            #There would be no else here because tensor product simp would have removed that part
        else:
            return expr     


# ## Stochastic/Pretty Printing (prints a random subset)

# In[13]:


stRandVal=1   
    
def stRand():
    global stRandVal
    stRandVal=random()

def stPrint(s):
    global stRandVal
    if(gls['printVerbose'] and stRandVal <= gls['printSparsity']):    
        print(s)
    
def stDisplay(s):
    global stRandVal
    if(gls['printVerbose'] and stRandVal <= gls['printSparsity']):    
        display(s)


# In[14]:


def prettyX(X):
    global L
    X_=np.zeros((len(L),len(L)))
    i=0
    j=0
    for l1 in L:
        for l2 in L:
            X_[i][j]=X[sTr[l1]][sTc[l2]].value
            j+=1
        j=0
        i+=1
    return X_


def prettyPrint(X):
    try:
        y=pd.DataFrame(prettyX(X))
    except:
        y=pd.DataFrame(X)
    #print(y)
    display(y)


# # Global Settings

# In[15]:


#This produces repeatable sparse prints
#uncomment to make them stochastic
seed(1)

gls = {}

gls['testType'] = 0
# 0 means no test; the real thing
# 1 is the KCBS objective; with neither observations nor the localising matrix constraints
# 2 means construct F for only the state
# 3 means ...
# 4 is feasibility testing; objective zero

#why do i have single variables in Lx?

gls['testSubType'] = "a" #"e" #"d" #"a2"
#For number 1:
#a (or any other not listed): KCBS objective
#b: adds observed constraints
#c: tests localising matrices with a translation operator based objective
#d: tests just the translation operator based objective (without localising matrices that is)
#e: KCBS from expression + twist
#a2: KCBS explicitly written with twist
#f: trace (tested with this)
#ks: Kishor Sucks! Testing the translate operator; hard coded

gls['hDepth'] = 3 #depth of the heirarchy; depends on the number of Πs in the expression for Fidelity; 3 needed

gls['locMatDepth'] = 2 #number of letters to consider for constructing the localising matrix

gls['usePs'] = True #enables the localising matrix setup; if this is disabled, hDepth would have to be around 6
#if this is enabled, hDepth can be 3 and it still runs fine

gls['cacheFolder'] = "cached/"

gls['N']=5

gls['cacheFolderN'] = gls['cacheFolder'] + "with"+ str(gls['N'])


gls['obsData'] = [np.cos(np.pi / gls['N']) / (1 + np.cos(np.pi / gls['N'])) for i in range(gls['N']) ]

gls['solverSettings'] = {'verbose':True,'solver':cp.SCS,'max_iters':16*2500,'warm_start':True}

def globalSettings(newGls):
    global gls
    
    #sanity check
    #[*gls] creates a list of keys
    #set converts a list into a set
    
    setKeysOriginal=set([*gls])
    setKeysNew=set([*newGls])
    
    if(setKeysNew<=setKeysOriginal):
        gls.update(newGls)
        gls['cacheFolderN']=gls['cacheFolder']+ "with" + str(gls['N'])
    else:
        print("Unknown setting:",setKeysNew-setKeysOriginal)        
        print("Current settings are:",gls)


# In[65]:


#globalSettings({'testTypea':7})
# print(gls)


# In[ ]:





# # Ideal KCBS

# ## The class

# In[16]:



class cls_iKCBS:    
    
    def __init__(s,n,verbose=False,thr=10,test=False,forceCalc=False):
        print("The testType is", gls['testType'])
        s.thr=thr
        if verbose==True:
            print("Creating projectors etc")

        cls_iKCBS.n=n
        #j in 0,1,2,3,4
        
        s.ke = [bKet(j) for j in range(3)]    
        s.br = [sq.Dagger(s.ke[j]) for j in range(3)]
        
        s.Πs = []
                
        ### N'd here
        #evaluate the projectors for the ideal case
        for j in range(n):
            s.Πs.append(sp.N(s.eval(j,n)))

        if(test and False):
            print("Testing Πs")
            for i in range(n):
                for k in range(n):
                    print("Inner product of ",i," and ",k," projectors is")
                    temp=0
                    for j in range(3):
                        temp+=sq.qapply(s.br[j]*s.Πs[i]*s.Πs[k]*s.ke[j])
                    print(temp)
                    del temp
                    
   
        #calculate the matrix representation
        s.matΠs = [sq.represent(_) for _ in s.Πs] #I leave this as well to avoid breaking things; TODO: try to remove it
        
        ### N'd here
        #the first two vectors are taken to be k'
        s.kp = [sp.N(s.eval(0,n,True)),sp.N(s.eval(1,n,True))]
        
        #the third is constructed as k0 x k1 (cross product) and then added to the list
        s.kp.append(sp.N(s.cross(s.kp)))
        
        
        #evaluates the conjugate for each element in the list
        s.bp=[sq.Dagger(ikp) for ikp in s.kp]                
        
        if(test):
            #Testing orthogonality
            print("Testing orthogonality of the braP and ketP objects")
            for braP in s.bp:
                for ketP in s.kp:
                    print(sq.qapply(braP*ketP))#sp.N(sq.qapply(braP*ketP)))
        
    
        #the Translation operator
        s.T = s.kp[0]*s.bp[2] + s.kp[1]*s.bp[0] + s.kp[2]*s.bp[1] 
        s.Ti = s.kp[2]*s.bp[0] + s.kp[0]*s.bp[1] + s.kp[1]*s.bp[2] #TODO: Automate this
        
        if(test):
            print("Testing the basic Translation operator")
            for braP in s.bp:
                for ketP in s.kp:
                    #print(braP)
                    print(sq.qapply(braP*s.T*ketP))#sp.N(sq.qapply(braP*ketP)))
                    #print(sp.N(sq.qapply(sp.N(braP)*sp.N(s.T)*sp.N(ketP))))
            display(sq.represent(s.T))

        
        #s.mT = s.mkp[0]*s.mbp[2] + s.mkp[1]*s.mbp[0] + s.mkp[2]*s.mbp[1] 
        #s.mTi = s.mkp[2]*s.mbp[0] + s.mkp[0]*s.mbp[1] + s.mkp[1]*s.mbp[2] #TODO: Automate this
        
        s.matT=sq.represent(s.T)
        
        # print("matT:")
        # print(s.matT)
        
        #Now the abstract ones
        
        #creates n abstract projectors (basically just a bunch of non-commuting symbols)
        s._Πs = sp.symbols('Π_0:%d'%n,commutative=False)
        
        #We use this to create a localising matrix
        #We must imposes conj(_P) * Translation >> 0
        s._P = sp.symbols('P',commutative=False) 
        s._Pd = sp.symbols('Pd',commutative=False) #we define them separately because sq.dagger doesn't work too well with symbols which are not based on kets etc
        #s._Pd = sq.dagger(sp)
        
        s.daggerDict = {}
        s.sqDict = {}
        for i in range(n):
            s.daggerDict[sq.Dagger(s._Πs[i])]=s._Πs[i]
            s.sqDict[(s._Πs[i])*(s._Πs[i])]=s._Πs[i]
        
        s.daggerDict[sq.Dagger(s._Pd)]=s._P
        s.daggerDict[sq.Dagger(s._P)]=s._Pd
        s.sqDict[s._P*s._Pd]=sp.Integer(1)
        s.sqDict[s._Pd*s._P]=sp.Integer(1)
        
        #not very happy because it should come from the graph
        s.gDict = {}
        for i in range(n):
            s.gDict[s._Πs[i]*s._Πs[(i+1)%n]]=sp.Integer(0)
            s.gDict[s._Πs[(i+1)%n]*s._Πs[i]]=sp.Integer(0)
        
        if verbose==True:
            print("Solving a linear system to find the coefficients for expressing the translation operator as a sum of projectors")

        s._T=s.eval_lin_c() #NB: it computes the coefficients into s.c    
        
        ##Debugging
        s.numTDict={}
        for i in range(n):
            s.numTDict[s._Πs[i]]=s.Πs[i]
        if(test):
            print("Converting the numerical solution into a usable form")
            #display(s._T)
            numT = s._T.subs(s.numTDict)
            #display(numT)
            print("Testing the numerically found Translation operator.")
            # for braP in s.bp:
            #     for ketP in s.kp:
            #         print(sq.qapply(braP*numT*ketP)) #sp.N(sq.qapply(braP*ketP)))
            numMatT=sq.represent(numT)
            display(sp.Transpose(numMatT) * numMatT)
            
            display(sq.represent(numT))
            

        if verbose==True:
            print("Solving the cross Π")
        
        # print("The cross is")
        # print(sq.represent(s.kp[2]*s.bp[2]))
        # print("|0'⟩")
        # print(sq.represent(s.kp[0]))
        # print("|1'⟩")
        # print(sq.represent(s.kp[1]))
        # print("|2'⟩")
        # print(sq.represent(s.kp[2]))
        
        # for i in range(3):
        #     for j in range(3):
        #         print("innerproduct:", i,j)
        #         print(sq.represent(s.bp[i]*s.kp[j]))
        # for i in range(5):
        #     print("ideal vec",i)
        #     print(sq.represent(sp.N(s.eval(i,n,True))))
        
        #s._Πcross = s.eval_Π_cross(RHS=sq.represent(s.kp[2]*s.bp[2])) #s.eval_lin_c(RHS=sq.represent(s.kp[2]*s.bp[2]))
        
        s._Πcross = s.eval_lin_c(RHS=sq.represent(s.kp[2]*s.bp[2]))
            
        if verbose==True:
            print("Evaluating F")

        if(test):
            s.Πcross = s.eval_lin_c(RHS=sq.represent(s.kp[2]*s.bp[2]),retIde=True)
            s.test_swapGate()
            
        if(not test):
            
            filename=gls['cacheFolderN']+"Fsaved4_loc"+str(gls['usePs'])
            if(gls['testType']==2):
                filename+="_stateOnly"
            
            try:
                y=dill.load(open(filename,"rb"))
                # if(gls['testType']==2):
                #     y=dill.load(open("Fsaved3_stateOnly", "rb"))
                # else:
                #     y=dill.load(open("Fsaved3", "rb"))
                s.F=y
                print("Loaded from file")
            except:
                y=None
            
            if y==None or forceCalc==True:
                print("Evaluating F")
                s.eval_state_hw()
                s.Fs = [s.eval_state_F()]    
                if(gls['testType']!=2):
                    for i in range(n):
                        print("iteration: ",i)
                        s.Fs.append(s.Fs[-1]+5.0*s.eval_state_F(s.Πs[i],s._Πs[i]))
                #s.F = s.Fs[-1] #last element, the most recent calculation
                s.F = prune(s.Fs[-1])
                # if(gls['testType']==2):
                #     dill.dump(s.F, open("Fsaved3_stateOnly", "wb"))
                # else:
                #     dill.dump(s.F, open("Fsaved3", "wb"))
                dill.dump(s.F,open(filename,"wb"))
                print("Saved to disk")
        
        print("done")
        
    def eval_idealVal(s,expr):
        value = 0
        try:
            value = sq.represent(expr.subs(s.numTDict))[0,0] #sq.represent(s.br[0])@sq.represent(expr.subs(s.numTDict))@sq.represent(s.ke[0])[0][0]
        except:
            value = sq.represent(expr.subs(s.numTDict)) #if a number was the expression
        
        return value
        
    def test_swapGate(s):
        s.TP = sq.TensorProduct 
        s.T = s.kp[0]*s.bp[2] + s.kp[1]*s.bp[0] + s.kp[2]*s.bp[1] 
        s.Ti = s.kp[2]*s.bp[0] + s.kp[0]*s.bp[1] + s.kp[1]*s.bp[2] #TODO: Automate this
        
        Id_ = s.kp[0]*s.bp[0] + s.kp[1]*s.bp[1] + s.kp[2]*s.bp[2]
        
        dontDie=prune(sp.N(sq.qapply(s.T*s.T)))
        
        _U_=sq.represent(s.TP(Id_,s.kp[0]*s.bp[0])) + sq.represent(s.TP(s.T,s.kp[1]*s.bp[1]))         + sq.represent(s.TP(dontDie,s.kp[2]*s.bp[2]))
        display(_U_)
        
        _T_=sq.represent(s.TP(Id_,s.kp[0]*s.bp[0] + s.kp[2]*s.bp[1] + s.kp[1]*s.bp[2]))
        display(_T_)        
        
        #_V_= sq.represent(s.TP(s.kp[0]*s.bp[0],Id_) + s.TP(s.kp[1]*s.bp[1],s.Ti) + s.TP(s.kp[2]*s.bp[2],s.Ti*sp.N(s.Ti)))
        _V_ = sq.represent(s.TP(s.Πs[0],Id_) + s.TP(s.Πs[1],s.Ti) + s.TP(s.Πcross,s.Ti*sp.N(s.Ti)))
        display(_V_)
        
        print("comparing the projectors with the ket based projectors")
        
        display(sq.represent(s.kp[0]*s.bp[0]))
        display(sq.represent(s.Πs[0]))

        display(sq.represent(s.kp[1]*s.bp[1]))
        display(sq.represent(s.Πs[1]))

        display(sq.represent(s.kp[2]*s.bp[2]))
        display(sq.represent(s.Πcross))
        
        print("did this work?")
        
        
        SWAP = _T_*_U_*_V_*_U_
        display(SWAP)
        
        testVec=sq.represent(s.TP(s.ke[0],s.ke[1]))
        print("Multiplying this")
        display(testVec)
        display(SWAP*testVec)
        
    def eval_state_hw(s):
        s.TP = sq.TensorProduct
        #Perhaps this should be P and not Pd
        
        if(gls['usePs']):
            #s._U_=tsimp(s.TP(sp.Integer(1),prune(sp.N(s.kp[0]*s.bp[0]))) + s.TP(s._Pd,prune(sp.N(s.kp[1]*s.bp[1]))) + s.TP(s._Pd * s._Pd,prune(sp.N(s.kp[2]*s.bp[2]))))
            s._U_=tsimp(s.TP(sp.Integer(1),prune(sp.N(s.kp[0]*s.bp[0]))) +                         s.TP(s._P,prune(sp.N(s.kp[1]*s.bp[1]))) +                         s.TP(s._P * s._P,prune(sp.N(s.kp[2]*s.bp[2]))))
        else:            
            dontDie=prune(sp.N(sq.qapply(s._T*s._T)))
            s._U_=tsimp(s.TP(sp.Integer(1),prune(sp.N(s.kp[0]*s.bp[0]))) +                         s.TP(s._T,prune(sp.N(s.kp[1]*s.bp[1]))) +                         s.TP(dontDie,prune(sp.N(s.kp[2]*s.bp[2]))))
            
        s._T_=s.TP(sp.Integer(1),prune(sp.N(s.kp[0]*s.bp[0])) + prune(sp.N(s.kp[2]*s.bp[1])) + prune(sp.N(s.kp[1]*s.bp[2])))
        s._V_=prune(s.TP(s._Πs[0],sp.Integer(1)) + s.TP(s._Πs[1],prune(sp.N(s.Ti))) + s.TP(prune(s._Πcross),prune(sp.N(s.Ti)*sp.N(s.Ti))))
        
    
    def eval_state_F(s,proj=sp.Integer(1),proj_=sp.Integer(1)):
        s.TP = sq.TensorProduct #just to make naming easy
            

        print("Evaluated U,T,V; staying alive")

        _kin_ = s.TP(sp.Integer(1),sp.N(sq.qapply(proj*s.ke[0])))
        _bin_ = s.TP(sp.Integer(1),sp.N(sq.qapply(s.br[0]*proj)))
        
        
        _kout_ = s.TP(sp.Integer(1)*proj_,sp.N(s.kp[0])) #s.TP(sp.Integer(1),sp.N(s.kp[0]))
        _bout_ = s.TP(sp.Integer(1)*proj_,sp.N(s.bp[0])) #s.TP(sp.Integer(1),sp.N(s.bp[0]))
                                                                                             

        print("evaluation stage:")
        F0=tsimp(_bin_ * s._T_).subs(s.gDict)
        print("1")
        F1=tsimp(F0 * s._U_).subs(s.gDict) #*_V_
        print("2")
        # #s.F2=tsimp(F1 * _V_)
        # #G00 = tsimp(F1*_V_)
        # #print("2.1")
        # #display(_U_)
        # G00 = tsimp(_U_*_kout_).subs(s.gDict)
        # print("2.1")
        # G01 = tsimp(_V_*G00).subs(s.gDict)
        # print("2.2")
        # #G0 = tsimp(_V_*_U_)
        # t1=sq.qapply(F1*G01).subs(s.gDict)
        # print("2.3")
        
        G0=tsimp(s._V_*_kout_).subs(s.gDict)
        print("3")
        
        #display(F1)
        #display(G0)
        t1=sq.qapply(F1*G0).subs(s.gDict)
        
        t2=t1.subs(s.daggerDict).subs(s.sqDict) #not sure why the s.daggerDict step is needed;
        
        H0 = tsimp(t2) #tsimp(F1*G0)
        print("4")
        #s.F1 = s._bout_ * _T_ * _U_ * _V_ * s._kout_
        #s.F2 = s.F1 * 
        s.F = tdsimp(H0)
        print("5")
        
        #return tdsimp(sp.expand(s.F * sq.Dagger(s.F)))
        return sp.expand((sq.Dagger(s.F) * s.F).subs(s.daggerDict)).subs(s.sqDict).subs(s.sqDict).subs(s.gDict)
        if verbose==True:
            print("Dinner out is a go.")
        
    def cross(s,vecs):
        ca=[sq.qapply(sq.Dagger(s.ke[i])*vecs[0]).doit() for i in range(3)]
        cb=[sq.qapply(sq.Dagger(s.ke[i])*vecs[1]).doit() for i in range(3)]
        
        res = (ca[2-1]*cb[3-1] - ca[3-1]*cb[2-1])*s.ke[0] +              (ca[3-1]*cb[1-1]-ca[1-1]*cb[3-1])*s.ke[1] +              (ca[1-1]*cb[2-1] - ca[2-1]*cb[1-1])*s.ke[2]
        
        return res

    
    #This I may not even be using
    #I had made eval_lin_c more capable
    def eval_Π_cross(s,RHS,varCount=None):
        cls=s.__class__
        if varCount==None:
            varCount=cls_iKCBS.n*2
        if varCount <= cls.n:
            raise ValueError("number of variables should be at least n")
    
        #a=sp.symbols('a0:%d'%varCount)
        varCount=9
        a=sp.symbols('a0:%d'%9)
        ap=[]
        #s.b=sp.symbols('b0:%d'%varCount)
        b=[] #family of solutions
        s.c=[] #particular solution with the free variables set to one
        
        #Setting up of constraints
        #If the system is not able to find a solution, consider increasing
        #the number of variables by taking different sets of products or products of three operators
        
        TfΠ = [a[0]*s.Πs[0] + a[1]*s.Πs[1] + a[2]*s.Πs[2] + a[3]*s.Πs[3] + a[4]*s.Πs[4] +                      a[5]*s.Πs[4]*s.Πs[2] +                      a[6]*s.Πs[4]*s.Πs[1] +                      a[7]*s.Πs[1]*s.Πs[3] +                      a[8]*s.Πs[2]*s.Πs[0] ]
        
        _TfΠ = [a[0]*s._Πs[0] + a[1]*s._Πs[1] + a[2]*s._Πs[2] + a[3]*s._Πs[3] + a[4]*s._Πs[4] +                      a[5]*s._Πs[4]*s._Πs[2] +                      a[6]*s._Πs[4]*s._Πs[1] +                      a[7]*s._Πs[1]*s._Πs[3] +                      a[8]*s._Πs[2]*s._Πs[0] ]        
        
#         TfΠ=[a[0]*s.Πs[0]]
#         _TfΠ=[a[0]*s._Πs[0]]
#         ap.append(a[0])
#         for i in range(1 +cls.n):
#             #if (not (i==2 or i==3)):
#             TfΠ.append(TfΠ[-1] + a[i]*s.Πs[i])
#             _TfΠ.append(_TfΠ[-1] + a[i]*s._Πs[i])
#             ap.append(a[i])            

#         if cls.n == 5:
#             TfΠ.append(TfΠ[-1] + a[5]*s.Πs[4]*s.Πs[2] \
#                     + a[6]*s.Πs[4]*s.Πs[1] \
#                     + a[7]*s.Πs[1]*s.Πs[3] \
#                     + a[8]*s.Πs[2]*s.Πs[0])
#             _TfΠ.append(_TfΠ[-1] + a[5]*s._Πs[4]*s._Πs[2] \
#                     + a[6]*s._Πs[4]*s._Πs[1] \
#                     + a[7]*s._Πs[1]*s._Πs[3] \
#                     + a[8]*s._Πs[2]*s._Πs[0])
#             ap+=[a[5],a[6],a[7],a[8]]

        LHS=sq.represent(TfΠ[-1])
        #display(_TfΠ)
        # LHS=a[0]*s.matΠs[0]
        # for i in range(1,cls.n):
        #     LHS=LHS+a[i]*s.matΠs[i]
        # for i in range(cls.n,varCount):
        #     LHS=LHS+a[i]*s.matΠs[i%cls.n]*s.matΠs[(i+2)%cls.n]
            
        #RHS=s.matT
        
        if cls.n!=5:
            soln=sp.solve(sp.N(LHS-RHS),a,dict=True)
        else:
            soln=sp.solve(sp.N(LHS-RHS),a[:9],dict=True)
            #soln=sp.solve(sp.N(LHS-RHS),ap,dict=True)
        
        print(soln)
        #print("type:",type(soln[0]))#, " and shape:", soln.shape())
        for key,value in soln[0].items():
            if np.abs(value)<10**(-s.thr):
                soln[0][key]=sp.Integer(0)
        
        print(soln)
        '''
            #soln=sp.solve([a[0]-1,a[1]-2,a[2]-3,a[3]-4,a[4]-5,a[5]-1,a[6]-1,a[7]-1,a[8]-1,a[9]-1],a,dict=True)

            # sols = solve([t1 + t2 + t3, eq1, eq2], [t1, t2, t3], dict=True)
            # sols[0][t1] # This is t1 in the first solution        
        '''

        #Dictionary to assign one/zero to free variables
        dRem={}
        
        #IMPORTANT: 
        #In sympy, symbols/expressions are unmutable (you can't change them ever);
        #You can save a substituted expression into a new expression (and use a symbol to denote it)
        #To wit: when you substitute, a new expression is produced

        for i in range(varCount):
            #the variables which were evaluated to an expression, assign them to b[i]
            try:
                b.append(soln[0][a[i]])      
            #if the variable was free, assign b[i] to be one
            #and create a dictionary to substitute these free variables
            except:
                b.append(sp.Integer(1))
                dRem[a[i]]=1
        
        #in the solution saved into b[i], substitute for the free variables using the dictionary
        #save the result into the variable c
        s.c = [_.subs(dRem) for _ in b]

        #substitute the solution into the coefficients in Tfπ_p, the operator T as a sum of projectors (and its products)
        dFin={}
        for i in range(varCount):
            dFin[a[i]]=s.c[i]                
        
        return _TfΠ[-1].subs(dFin)            
        
    
    #evaluates the matrices and vectors
    def eval(s,j,n,ve=False):
        N=sp.Integer(n)
        J=sp.Integer(j+1)
        #print(N,j)
        one=sp.Integer(1)
        #print(one)
        α1=sp.cos((sp.pi)/N)/(one+sp.cos(sp.pi/N)) #verified
        #a1=sp.cos(sp.pi/N)
        #
        #print(α1)
        
        α2=one-α1 #verified
        α3=J * sp.pi * (N-one)/N #verified
        
        vec = sp.sqrt(α1)*s.ke[0] + sp.sqrt(α2)*sp.sin(α3)*s.ke[1] + sp.sqrt(α2)*sp.cos(α3)*s.ke[2]
        
        projector = vec*sq.Dagger(vec)
           
        #matrixprojector = sp.Matrix([0,0])
        #return projector
        #display(sq.represent(projector))
        #display(sq.represent(vec))
        
        if ve==False:
            return projector
        else:
            return vec
    
    
    #I use this for both the projector
    #and for the cross product
    
    def eval_lin_c(s,varCount=None,RHS=None,retIde=False):
        cls=s.__class__
        if varCount==None:
            varCount=cls_iKCBS.n*2
        if varCount <= cls.n:
            raise ValueError("number of variables should be at least n")
    
        a=sp.symbols('a0:%d'%varCount)
        #s.b=sp.symbols('b0:%d'%varCount)
        b=[] #family of solutions
        s.c=[] #particular solution with the free variables set to one
        
        #Setting up of constraints
        #If the system is not able to find a solution, consider increasing
        #the number of variables by taking different sets of products or products of three operators
        TfΠ=[a[0]*s.Πs[0]]
        _TfΠ=[a[0]*s._Πs[0]]
        
        
        #if n is 5, then we use a special form (the general method doesn't find good solutions for some reason)
        if cls.n == 5:            
            for i in range(1,5):
                TfΠ.append(TfΠ[-1] + a[i]*s.Πs[i])
                _TfΠ.append(_TfΠ[-1] + a[i]*s._Πs[i])
            
            TfΠ.append(TfΠ[-1] + a[5]*s.Πs[4]*s.Πs[2]                     + a[6]*s.Πs[4]*s.Πs[1]                     + a[7]*s.Πs[1]*s.Πs[3]                     + a[8]*s.Πs[2]*s.Πs[0])
            _TfΠ.append(_TfΠ[-1] + a[5]*s._Πs[4]*s._Πs[2]                     + a[6]*s._Πs[4]*s._Πs[1]                     + a[7]*s._Πs[1]*s._Πs[3]                     + a[8]*s._Πs[2]*s._Πs[0])

            if RHS == None:
                RHS=s.matT
            
            LHS=sq.represent(TfΠ[-1])
            
            soln=sp.solve(sp.N(LHS-RHS),a[:9],dict=True)
        #if n is 7, then the form we use depends on whether we want the cross product solved or the translation operator solved
        #if it is the translation operator, then RHS is none
        elif cls.n == 7:
            for i in range(1,5):
                TfΠ.append(TfΠ[-1] + a[i]*s.Πs[i])
                _TfΠ.append(_TfΠ[-1] + a[i]*s._Πs[i])
            
            
            #x1*P1 + x2*P2 + x3*P3 + x4*P4 + x5*P5 + x6*(P5@P3) + x7*(P5@P2) + x8*(P2@P4) + x9*(P3@P1)
            #x5*(p4*p2) + x6*(p4*p1) + x7*(p1*p3) + x8(p2*p0)
            
            #Translation Op | the form is the same as that for n=5
            if RHS == None:
                TfΠ.append(TfΠ[-1] + a[5]*s.Πs[4]*s.Πs[2]                         + a[6]*s.Πs[4]*s.Πs[1]                         + a[7]*s.Πs[1]*s.Πs[3]                         + a[8]*s.Πs[2]*s.Πs[0])
                _TfΠ.append(_TfΠ[-1] + a[5]*s._Πs[4]*s._Πs[2]                         + a[6]*s._Πs[4]*s._Πs[1]                         + a[7]*s._Πs[1]*s._Πs[3]                         + a[8]*s._Πs[2]*s._Πs[0])

                RHS=s.matT
                print("Solving the Translation operator's solution hardcoded")

                LHS=sq.represent(TfΠ[-1])

                soln=sp.solve(sp.N(LHS-RHS),a[:9],dict=True)
                
                
            #For the Cross Operator | the form is now different    
            #term1 = x1*P1 + x2*P2 +x3*P3 + x4*P4 + x5*P5 + x6*P6
            #x0*p0 + x1*p1 + x2*p2 + x3*p3 + x4*p4 + x5*p5
            # term2 = x8*(P1@P3) + x9*(P3@P1) + x10*(P2@P4)
            #x7*p0*p2 + x8*p2*p0 + x9*p1*p3

            else:
                TfΠ.append(TfΠ[-1] + a[5]*s.Πs[5]                         +  a[7]*s.Πs[0]*s.Πs[2]                         +  a[8]*s.Πs[2]*s.Πs[0]                         +  a[9]*s.Πs[1]*s.Πs[3])
                _TfΠ.append(_TfΠ[-1] + a[5]*s._Πs[5]                         +  a[7]*s._Πs[0]*s._Πs[2]                         +  a[8]*s._Πs[2]*s._Πs[0]                         +  a[9]*s._Πs[1]*s._Πs[3])            

                LHS=sq.represent(TfΠ[-1])

                soln=sp.solve(sp.N(LHS-RHS),a[:10],dict=True)

        else:
            #in general we include a[0]*Π[0] + a[1]*Π[1] + .. a[n-1]*Π[n-1]
            for i in range(1,cls.n):
                TfΠ.append(TfΠ[-1] + a[i]*s.Πs[i])
                _TfΠ.append(_TfΠ[-1] + a[i]*s._Πs[i])
            
            for i in range(cls.n,varCount):
                TfΠ.append(TfΠ[-1] + a[i]*s.Πs[i%cls.n]*s.Πs[(i+2)%cls.n])
                _TfΠ.append(_TfΠ[-1] + a[i]*s._Πs[i%cls.n]*s._Πs[(i+2)%cls.n])
                
            if RHS == None:
                RHS=s.matT
            
            LHS=sq.represent(TfΠ[-1])
            
            soln=sp.solve(sp.N(LHS-RHS),a,dict=True)
            
        
            
            
#         LHS=sq.represent(TfΠ[-1])
#         #display(_TfΠ)
#         # LHS=a[0]*s.matΠs[0]
#         # for i in range(1,cls.n):
#         #     LHS=LHS+a[i]*s.matΠs[i]
#         # for i in range(cls.n,varCount):
#         #     LHS=LHS+a[i]*s.matΠs[i%cls.n]*s.matΠs[(i+2)%cls.n]
#         if RHS == None:
#             RHS=s.matT
        
#         if cls.n==5:
#             soln=sp.solve(sp.N(LHS-RHS),a[:9],dict=True)
#         else:
#             soln=sp.solve(sp.N(LHS-RHS),a,dict=True)
            
        
        
        
        
        print(soln)
        #print("type:",type(soln[0]))#, " and shape:", soln.shape())
        for key,value in soln[0].items():
            prune(value)
            soln[0][key]=value
            # print(value)
            # if np.abs(value)<10**(-s.thr):
            #     soln[0][key]=sp.Integer(0)
        
        print(soln)
        '''
            #soln=sp.solve([a[0]-1,a[1]-2,a[2]-3,a[3]-4,a[4]-5,a[5]-1,a[6]-1,a[7]-1,a[8]-1,a[9]-1],a,dict=True)

            # sols = solve([t1 + t2 + t3, eq1, eq2], [t1, t2, t3], dict=True)
            # sols[0][t1] # This is t1 in the first solution        
        '''

        #Dictionary to assign one/zero to free variables
        dRem={}
        
        #IMPORTANT: 
        #In sympy, symbols/expressions are unmutable (you can't change them ever);
        #You can save a substituted expression into a new expression (and use a symbol to denote it)
        #To wit: when you substitute, a new expression is produced

        for i in range(varCount):
            #the variables which were evaluated to an expression, assign them to b[i]
            try:
                b.append(soln[0][a[i]])      
            #if the variable was free, assign b[i] to be one
            #and create a dictionary to substitute these free variables
            except:
                b.append(sp.Integer(1))
                dRem[a[i]]=1
        
        # print("Substituting free variables for one")
        # print(b)
        
        #in the solution saved into b[i], substitute for the free variables using the dictionary
        #save the result into the variable c
        s.c = [_.subs(dRem) for _ in b]

        #substitute the solution into the coefficients in Tfπ_p, the operator T as a sum of projectors (and its products)
        dFin={}
        for i in range(varCount):
            dFin[a[i]]=s.c[i]                
        
        #soln[0].subs(dFin)
        print("final substitution")
        print(dFin)
        
        if retIde==False:
            return _TfΠ[-1].subs(dFin)
        else:
            return TfΠ[-1].subs(dFin)
            
    def eval_c(s,varCount=None):
        cls=s.__class__
        if varCount==None:
            varCount=cls_iKCBS.n*2
        if varCount <= cls.n:
            raise ValueError("number of variables should be at least n")
    
        a=sp.symbols('a0:%d'%varCount)
        #s.b=sp.symbols('b0:%d'%varCount)
        b=[] #family of solutions
        s.c=[] #particular solution with the free variables set to one
        
        #Setting up of constraints
        #If the system is not able to find a solution, consider increasing
        #the number of variables by taking different sets of products or products of three operators
        TfΠ=a[0]*s.Πs[0]
        _TfΠ=a[0]*s._Πs[0]
        for i in range(1,cls.n):
            TfΠ=TfΠ + a[i]*s.Πs[i]
            _TfΠ=_TfΠ + a[i]*s._Πs[i]
        for i in range(cls.n,varCount):
            TfΠ=TfΠ + a[i]*s.Πs[i%cls.n]*s.Πs[(i+2)%cls.n]
            _TfΠ=_TfΠ + a[i]*s._Πs[i%cls.n]*s._Πs[(i+2)%cls.n]
            
            
        LHS=sq.represent(TfΠ)
        
        # LHS=a[0]*s.matΠs[0]
        # for i in range(1,cls.n):
        #     LHS=LHS+a[i]*s.matΠs[i]
        # for i in range(cls.n,varCount):
        #     LHS=LHS+a[i]*s.matΠs[i%cls.n]*s.matΠs[(i+2)%cls.n]
            
        RHS=s.matT
        
        soln=sp.solve(sp.N(LHS-RHS),a,dict=True)

        '''
            #soln=sp.solve([a[0]-1,a[1]-2,a[2]-3,a[3]-4,a[4]-5,a[5]-1,a[6]-1,a[7]-1,a[8]-1,a[9]-1],a,dict=True)

            # sols = solve([t1 + t2 + t3, eq1, eq2], [t1, t2, t3], dict=True)
            # sols[0][t1] # This is t1 in the first solution        
        '''

        #Dictionary to assign one/zero to free variables
        dRem={}
        
        #IMPORTANT: 
        #In sympy, symbols/expressions are unmutable (you can't change them ever);
        #You can save a substituted expression into a new expression (and use a symbol to denote it)
        #To wit: when you substitute, a new expression is produced

        for i in range(varCount):
            #the variables which were evaluated to an expression, assign them to b[i]
            try:
                b.append(soln[0][a[i]])      
            #if the variable was free, assign b[i] to be one
            #and create a dictionary to substitute these free variables
            except:
                b.append(sp.Integer(1))
                dRem[a[i]]=1
        
        #in the solution saved into b[i], substitute for the free variables using the dictionary
        #save the result into the variable c
        s.c = [_.subs(dRem) for _ in b]

        #substitute the solution into the coefficients in Tfπ_p, the operator T as a sum of projectors (and its products)
        dFin={}
        for i in range(varCount):
            dFin[a[i]]=s.c[i]                
        
        return _TfΠ.subs(dFin)
        
    #This doesn't work as expected
    def partial_trace(s,M):
        
        s.br_ = [ sq.TensorProduct(sq.IdentityOperator(),_) for _ in s.br]
        s.ke_ = [ sq.TensorProduct(sq.IdentityOperator(),_) for _ in s.ke]
        
        res=0*s.br_[0]*M*s.ke_[0]        #to get the type right!
        for i in range(3):
            for j in range(3):
                res=res+s.br_[i]*M*s.ke_[j]
        #return sq.qapply(res).doit()
        return tsimp(res)
        


# ## Instantiation Interface

# In[17]:


# iKCBS = None

def init_iKCBS(verbose=True):
    global iKCBS
    iKCBS=cls_iKCBS(gls['N'],verbose=verbose)


# ## Testing

# In[105]:


#init_iKCBS()


# # SDP Part (SymPy meets CvxPy)

# ## Preparation

# In[18]:


def init_SC_0():
    global N,I,A,B
    
    N=gls['N']

    print(N)
    I=[sp.Integer(1)]
    A=list(sp.symbols('Π_0:%d'%N,commutative=False))
    B=[sp.symbols('P',commutative=False),sp.symbols('Pd',commutative=False)]


# display(I)
# display(A)
#print(B)


# In[19]:


def init_SC_1():
    print("Fetching the objective")
    global objective 
    objective = iKCBS.F
    print(B)


# In[20]:


L1 = []

def init_SC_2():
    print("Constructing dictionaries")
    global N,I,A,B,sqDicts,bConjDicts,L1
    
    if(gls['usePs']):
        L1=A+B
    else:
        L1=A

    #These will help simplify A^3 to A when needed
    #lDepth = 2
    ##PUT THIS IN GLOBAL (after understanding what is happening)
    
    gls['lDepth']=2
    
    sqDicts={}
    bConjDicts={}

    for iA in A:
        sqDicts[iA**5]=iA
        sqDicts[iA**4]=iA
        sqDicts[iA**3]=iA
        sqDicts[iA**2]=iA

    sqDicts[B[0]*B[1]]=sp.Integer(1)
    sqDicts[B[1]*B[0]]=sp.Integer(1)

    #This I use to conjugate
    bConjDicts={B[0]:B[1],B[1]:B[0]}

#I also define the conj function
def conj(expr):
    return rev(expr).subs(bConjDicts,simultaneous=True)


# In[21]:


def init_SC_3():
    global N,I,A,B,L1,G,gDicts
    print("Creating the graph (assumed cyclic for now)")
    #print(N,L1)
    
    G = nx.Graph()

    G = nx.Graph()
    G.add_nodes_from(L1)
    G.add_edges_from([[L1[i],L1[(i+1)%(N)]] for i in range(N)])
    

    #create a dictionary to simplify expressions later which sets projector products to zero
    #global gDicts
    gDicts={}
    for l1 in L1:
        for l2 in L1:
            if l1 in G[l2]:
                gDicts.update({l1*l2:sp.Integer(0)})


    # get_ipython().run_line_magic('matplotlib', 'inline')
    nx.draw(G, with_labels=True)


# In[22]:


def init_SC_4():
    global N,I,A,B,L1,G,gDicts,objective_
    print("Simplifying the objective using the graph")
    #Simplify objective using the orthogonality relations of the graph
    #display(prune(objective).subs(gDicts))  
    if(gls['testType']==1):
        #quick and dirty
        #because it won't be used
        objective_ = sp.expand(objective)
    else:
        #treat it properly
        objective_ = sp.expand(prune(objective).subs(gDicts)).subs(gDicts).subs(sqDicts)

        #display(objective_)


# In[23]:


def init_SC_5():
    global N,I,A,B,L1,G,gDicts,objective_,Lx,L,L_
    
    Lx=[]
    force=False
    fileName=gls['cacheFolderN']+"alphabetSave_testType_"+str(gls['testType'])+"_hDepth_"+str(gls['hDepth'])+"_lenL1_"+str(len(L1))

    try:
        print("Loading from file")
        loadedThis=dill.load(open(fileName, "rb"))
        [Lx,L,L_]=loadedThis
        print("Done")
    except:
        print("Failed to load")
        loadedThis=None


    if (loadedThis==None or force==True):
        Lx=[]
        print("Evaluating")
        #This generates words of length n with each position holding any of the y letters
        def loop_rec(y, n=None, val=None):
            if n >= 1:
                for x in range(y):
                    let=L1[x]
                    if val == None:
                        newVal=let
                    else:
                        newVal=let*val
                    loop_rec(y, n - 1, newVal)
            else:
                #TODO: Find a better way of removing powers
                #remember sqDicts also simplifies UdU to identity
                val_=val.subs(gDicts).subs(sqDicts).subs(sqDicts).subs(gDicts)
                if not val_ in Lx and not val_ in L1 and val_ != sp.Integer(0) and val_ != sp.Integer(1) :
                    #print(val_)
                    #print(val_ in L1)
                    Lx.append(val_)

        for i in range(2,gls['hDepth']+1):    
            #loop_rec(N,i)
            loop_rec(len(L1),i)


        L = I + L1 + Lx
        L_ = L1 + Lx

        dill.dump([Lx,L,L_], open(fileName, "wb"))
        print("Saved to disk")


    print(L1)
    if(len(Lx)<30):
        display(Lx)
    else:
        print(Lx)    
    


# In[24]:


def init_SC_6():
    global N,I,A,B,L1,G,gDicts,objective_,Lx,L,L_,sTi,sTr,sTc
    
    #symbol to index dictionary
    sTi={}

    #The former should be obseleted gradually

    sTr={}
    sTc={}

    for i in range(len(L)):
        sTc[L[i]]=i #for the column, the usual thing should work just as well
        #print(L[i],conj(L[i]))
        sTr[conj(L[i])]=i #for the row it should be conjugated

        sTi[L[i]]=i

    #print(sTi)
    
def init_SC_7():
    global N,I,A,B,L1,G,gDicts,objective_,Lx,L,L_,sTi,sTr,sTc,Lloc
    # Defining a class of words with number of letters bounded

    ## Also initialise the variables for the localising matrix
    ## Else things don't quite work
    Lloc=[]
    for lett in L:
        #print(lett)
        if (isinstance(lett,sp.Number)):
            Lloc.append(lett)
        elif (len(pow_to_single_mul(lett).args)<=gls['locMatDepth']):
            Lloc.append(lett)
    


# In[25]:




def lettersForExpr(termInSum,silent=False):
    global I
    lI = I[0]
    #global N,I,A,B,L1,G,gDicts,objective_,Lx,L,L_,sTi,sTr,sTc
    
    termInSum_ = pow_to_single_mul(termInSum) #sp.expand(pow_to_mul(termInSum))
    #print(termInSum_)
    #print(termInSum_.args)
    if(len(termInSum_.args)==0):
        return[termInSum_,lI,1.0]    
    elif(len(termInSum_.args[1:])<= gls['hDepth']  ): #if we can handle this using just the letters we produced
        lett = termInSum_.args[1]
        for lettInTerm in termInSum_.args[2:]:
            lett *= lettInTerm
        coeff = sp.N(termInSum_.args[0]) #In case we got symbolic square roots

        lett=lett.subs(sqDicts)            
        return[lett,lI,coeff]
        #eX = X[sTi[lett]][sTi[lI]]
        #y += coeff*eX

        #display(termInSum_)
        #display(lett) #These can be directly handled

    elif(len(termInSum_.args[1:])<= 2*gls['hDepth'] ): #we can handle at most twice the letters            
        half=int(gls['hDepth']) +1
        #print(len(termInSum_.args[1:]),2*hDepth)
        #print(half)
        #print(termInSum_)
        lett1 = termInSum_.args[1]
        for lettInTerm in termInSum_.args[2:half]:
            lett1 *= lettInTerm
        lett2 = termInSum_.args[half]
        for lettInTerm in termInSum_.args[half+1:]:
            lett2 *= lettInTerm
        # display(termInSum_)
        # display(lett1) 
        # display(lett2)
        coeff = sp.N(termInSum_.args[0]) #In case we got symbolic square roots
        lett1=lett1.subs(sqDicts)
        lett2=lett2.subs(sqDicts)
        return [lett1,lett2,coeff]
        #eX = X[sTi[lett1]][sTi[lett2]]
        #y += coeff*eX
    else:
        if(silent==False):
            print("objective not fully captured")           
        return ["objective not fully captured"]
        #flag="Objective not fully captured"
        #raise Warning("Increase hDepth: It looks like your objective is not captured by the current depth of the SDP heirarchy")
    


# In[26]:


def init_SympyCvxpy():
    init_SC_0()
    init_SC_1()
    init_SC_2()
    init_SC_3()
    init_SC_4()
    init_SC_5()
    init_SC_6()
    init_SC_7()
    


# ## Constraints from QM and setting up SDP variables

# In[27]:



def init_QMconstraintsToSDP():
    global N,I,A,B,L1,G,gDicts,objective_,Lx,L,L_,sTi,sTr,sTc,Lloc,X,X1,constraints,locConstraints
    
    ForceCalc=False
    verbose=False
    partialDisplay=False

    hint=False #This will probably start working in a new version of cvxpy

    fileName=gls['cacheFolderN']+"lSave_4_testType_"+str(gls['testType'])+"_hDepth_"+str(gls['hDepth'])+"_lenL_"+str(len(L))+"_locDepth"+str(gls['locMatDepth'])+"_usePs_"+str(gls['usePs'])

    flags=["issues"]


    ConstraintsThreshold = 10000 #Maximum number of constraints
    useThreshold=False

    # X = cp.Variable((len(L),len(L)),symmetric=True)    
    # X1 = cp.Variable((len(Lloc),len(Lloc)),symmetric=True)       #Localising matrix

    ###################
    #Settings for the localising matrix

    gls['printVerbose']=True
    gls['printSparsity'] = 0.01
    seed(1) #to have repeatable "random" prints

    try:
        print("Loading from file",fileName)
        loadedThis=dill.load(open(fileName, "rb"))
        #Just trying to not save the variables; instead defining them each time
        [X,X1,constraints,locConstraints]=loadedThis
        print("Done")
        print("SDP Size",X[0])
        print("Constraints:", len(constraints))
        print("Localising Constraints:", len(locConstraints))
    except:
        loadedThis=None




    if loadedThis==None or ForceCalc==True:
        print("Evaluating constraints")


        #L is the alphabet, to wit: it is the list of letters
        #
        X = cp.Variable((len(L),len(L)),symmetric=True)    

        #>> is for matrix inequality
        constraints = [X >> 0,X[sTi[lI],sTi[lI]]==1]


        print(Lloc)
        # if locMatDepth=2
        # π1,π2,π0π2,..,π3π0

        #M = []
        X1 = cp.Variable((len(Lloc),len(Lloc)),symmetric=True)       #Localising matrix



    #     print("Adding constraints from the graph")
    #     #Constraints from the graph (these come from single letters)        
    #     for l1 in L1:
    #         for l2 in L1:
    #             eX = X[sTi[l1]][sTi[l2]] #picks the corresponding element from the variable matrix of the SDP library
    #             if l1 in G[l2]:
    #                 constraints += [eX == 0]

    #     print("done")




        print("Adding constraints from Quantum Mechanics")

        tempCount=0
        dictTerms = {}
        halfLoop = 0
        #lettersUsed = []
        for l1 in L:
            print("Row of ",l1)
            if(tempCount>ConstraintsThreshold):
                break

            #for l2 in L[halfLoop:]:
            for l2 in L:
                #Constraints from symmetry
                #constraints += [ X[sTr[l1]][sTc[l2]] == X[sTc[l2]][sTr[l1]] ]

                term = l1*l2


                tS=[term]
                #display(term)
                for i in range(gls['lDepth']+1):
                    tS.append(tS[-1].subs(sqDicts))
                simTerm=tS[-1]

                #Don't enable this until cvxpy has been upgraded to handle giving a guess solution
                if(hint==True):                
                    X[sTr[l1]][sTc[l2]].value=iKCBS.eval_idealVal(simTerm)

                #display(simTerm)
                #Orthogonality from the graph
                tempVal=simTerm.subs(gDicts)
                if tempVal==sp.Integer(0):
                    #print("I set ",l1,l2," to zero")
    #                 display(l1)
    #                 display(l2)
    #                 display(simTerm.subs(gDicts))
                    constraints += [ X[sTr[l1]][sTc[l2]] == 0 ]
                elif tempVal == sp.Integer(1):
                    constraints += [ X[sTr[l1]][sTc[l2]] == 1 ]   
                    print(term," set to ",1)
                elif not(simTerm in dictTerms):
                    dictTerms.update({simTerm:[l1,l2]})            
                else:
    #                 l1_=dictTerms[simTerm][0]
    #                 l2_=dictTerms[simTerm][1]
                    [l1_,l2_]=dictTerms[simTerm]
                    #I am not sure why the or part of this statement is reducing constraints

                    if (not (l1==l1_ and l2==l2_)): #and not (l1==l2_ and l2==l1_)):                

                        #print()
                        #if ((l1==l2_ and l2==l1_)):
    #                         print("GOT ONE!")
    #                         print("The ls")
    #                         display(l1)
    #                         display(l2)

    #                         print("The l_s")
    #                         display(l1_)
    #                         display(l2_)

                        constraints += [ X[sTr[l1]][sTc[l2]] == X[sTr[l1_]][sTc[l2_]] ]

                        #(ignore)
                        #Hermitian conjugate should be the same
                        #easiest way to see this is that the matrix indices, M_ij=M_ji 
                        #The conjugate part is automatically taken care of in sTc and sTr


                        #constraints += [ X[sTr[l2]][sTc[l1]] == X[sTr[l2_]][sTc[l1_]] ]

                        tempCount+=1

                        if(tempCount>ConstraintsThreshold and useThreshold==True):
                            break

                        if(tempCount<10 or                           (100<tempCount and tempCount<110) or                           (1000<tempCount and tempCount<1010) or verbose):

                            #print("Adding a new constraint")
                            print(l1,"*",l2,"=",l1_,"*",l2_)
                            try:
                                v1=iKCBS.eval_idealVal(l1*l2)
                                v2=iKCBS.eval_idealVal(l1_*l2_)                        
                                if(np.abs(v1 - v2)>1e-9): #1e-9):
                                    print("Issue?")
                                    print( v1, v2, np.abs(v1-v2) )
                            except:
                                pass

                            #print("should be equal to",l1_,",",l2_)
                            if(partialDisplay):
                                print("l1")
                                display(l1)
                                print("l2")
                                display(l2)
                                print("l1_")
                                display(l1_)
                                print("l2_")
                                display(l2_)

                del tS            
            halfLoop+=1

        print("done")

        print("Constraints", len(constraints))
        print("SDP", X[0])

    ###############Localising Matrix Constraints (had to be done together else the saving doesn't seem to work)

        if(gls['usePs']==True):
        #if(hDepth>=3 and len(L1)>5):

            print("Localising matrix constraints...")

            #Construct a restricted set from L
            locConstraints=[X1 >> 0] #X1 is PSD

            i=0
            j=0
            for lett1 in Lloc:
                #goes up to two letters            
                print("on row:",lett1)
                row = []
                for lett2 in Lloc:   
                    stRand()
                    #I need to conjugate because I am constructing the
                    #matrix in this order

                    expr = sp.expand(conj(lett1)*B[1]*iKCBS._T*lett2).subs(gDicts).subs(sqDicts).subs(gDicts).subs(sqDicts)
                    # print("the expression")

                    stPrint("Current Expression")
                    stDisplay(expr)
                    #res = None
                    res = None
                    allTerms=True
                    if expr != sp.Integer(0):
                        #if there are many terms in the sum, sum over them
                        if(isinstance(expr,sp.Add)):
                            for _ in expr.args:
                                #print("Going in")
                                res_ = lettersForExpr(_,silent=True)
                                stPrint("Current Word:")
                                stPrint(_)
                                stPrint("Extraction:")
                                stPrint(res_)
                                try:                                
                                    __ = res_[2]*X[sTr[res_[0]]][sTc[res_[1]]]   
                                    if(res is None):
                                        res = __
                                    else:
                                        res += __ #this will contain something like 1.13*Π_1*Π_2 + 1.42*Π_1*Π_2
                                    stPrint("Added to sum.")
                                except:
                                    stPrint("Couldn't key.")
                                    allTerms=False
                                    break #this should break the loop because one of the terms couldn't be keyed
                                    #pass
                                    #the constraint belongs to a level of heirarchy not being considered
                        #If it is just one term, compute directly
                        else:
                            res_ = lettersForExpr(expr,silent=True)
                            #print("printing res ")
                            #print(res_)

                            try:
                                res = res_[2]*X[sTr[res_[0]]][sTc[res_[1]]]      
    #                             __ = res_[2]*X[sTr[res_[0]]][sTc[res_[1]]]      
    #                             if(res is None):
    #                                 res = __
    #                             else:
    #                                 res += __ #this will contain something like 1.13*Π_1*Π_2 + 1.42*Π_1*Π_2
                            except:
                                allTerms=False
                                pass
                                #constraint is in a higher level of the heirarchy;
                    #I don't think I even need to check if res is None            
                    if ( (not res is None) and (allTerms==True) ):                    
                        locConstraints+=[X1[i][j]==res]
                        stPrint("Constraint Added.")
                    else:
                        stPrint("Constraint not added.")
                        # if(i!=j):
                        #     locConstraints+=[X1[j][i]==res]
                        #row.append(res) #this appends that calculation into the row
                    # else:
                    #     var = cp.Variable()
                    #     row.append(var) #(cp.Constant(0))
                    #     newCpVars.append(var)
                    j+=1
                i+=1
                j=0

        else:                           #if usePs is off, save nothing #if (locConstraints is None):
            locConstraints = []
        dill.dump([X,X1,constraints,locConstraints], open(fileName, "wb"))
        print("Saved to disk")


    try:
        del fileName
        del ForceCalc
    except:
        print("Cleaned.")


# ## Objective; Symbolic to SDP

# In[28]:



def init_symObjectiveToSDP():
    global N,I,A,B,L1,G,gDicts,objective_,Lx,L,L_,sTi,sTr,sTc,Lloc,X,X1,constraints,locConstraints,y
    #Objective function; symbolic to SDP constraints

    # objective function evaluated (loop over product of all letters)
    y = 0

    
    if(gls['testType']==1):

        if(gls['testSubType']=="b"):
            print("Feasibilty test; objective=0")
        elif(gls['testSubType']=="c"):
            print("TEST: Simple overlap testing the translate operator using localising matrices")
            #I want the expression to be Π_0 P Π_2 Pd
            #because P Π_2 Pd = Π_0 in the ideal case
            # B[0] = P; B[1] = Pd
            # A[i] = Π_i
            # expr = A[0] * B[0] * A[2] * B[1]
            y = X[sTr[A[1]*B[0]]][sTc[A[0]*B[1]]]    

        elif(gls['testSubType']=="ks"):
            print("TEST: Kishor Sucks?")
            #<T> hard coded
            y=-0.272019649514071*X[sTr[A[0]]][sTc[lI]] -                                         0.485868271756646*X[sTr[A[1]]][sTc[lI]] +                                         3.33019067678556*X[sTr[A[1]*A[3]]][sTc[lI]] -                                         2.05817102727149*X[sTr[A[2]]][sTc[lI]] +                                         2.77032771530716*X[sTr[A[2]*A[0]]][sTc[lI]] +                                         2.77032771530715*X[sTr[A[4]]][sTc[lI]] -                                         4.82849874257864*X[sTr[A[4]*A[1]]][sTc[lI]] -                                         1.15229372655726*X[sTr[A[4]*A[2]]][sTc[lI]]

            ## −0.272019649514071Π0 − 0.485868271756646Π1 + 3.33019067678556Π1Π3
            ## −2.05817102727149Π2 + 2.77032771530716Π2Π0 + 2.77032771530715Π4
            ## −4.82849874257864Π4Π1 − 1.15229372655726Π4Π2        

        elif(gls['testSubType']=="d" or gls['testSubType']=="e"):
            # res=None
            if(gls['testSubType']=="d"):
                print("TEST: Translate without localising matrices")
                print("Translation operator")
                display(iKCBS._T)
                print("Conjugated translation operator")
                display(conj(iKCBS._T))            
                #expr = iKCBS._T #
                expr=A[1]*A[3]
                #expr = ((A[1] * iKCBS._T * A[0] * (conj(iKCBS._T))).expand().subs(gDicts).subs(sqDicts).subs(gDicts).subs(sqDicts)).expand()
                #expr = ((A[1] * iKCBS._T * A[0]).expand().subs(gDicts).subs(sqDicts).subs(gDicts).subs(sqDicts)).expand()
                #expr = ((iKCBS._T * (conj(iKCBS._T))).expand().subs(gDicts).subs(sqDicts).subs(gDicts).subs(sqDicts)).expand()
                #expr = A[1]

            elif(gls['testSubType']=="e"):
                print("TEST: KCBS objective with extra things")
                expr= A[0] + A[1] + A[2] + A[3] + A[4] + 0.2*A[0]*A[2]*A[0] + 0.3*A[0]*A[2]*A[0]


            #expr = pow_to_single_mul( ( A[0] * iKCBS._T * A[2] * (conj(iKCBS._T)).expand()).expand() )
            elif(gls['testSubType']=="f"):
                print("trace of T*conj(T)")
                expr = ((iKCBS._T * (conj(iKCBS._T))).expand().subs(gDicts).subs(sqDicts).subs(gDicts).subs(sqDicts)).expand()


            display(expr)
            print("going in",expr.args)
            for _ in expr.args:
                #print("Going in")            
                display(_)
                res_ = lettersForExpr(_)
                print(res_)
                __ = res_[2]*X[sTr[res_[0]]][sTc[res_[1]]] 
                if(y is 0):
                    y = __
                else:
                    y += __

        else:
            print("TEST: Constructing the KCBS expression instead of the Fidelity")
            len(L1)
            #print(L1[:1])
            for term in L1[:5]:
                display(term)            
                y += X[sTr[lI]][sTc[term]]
            #y += 3*X[sTi[lI]][sTi[L1[2]]]            
            if(gls['testSubType']=="a2"):
                print("adding extra things")
                y+=0.2*X[sTr[A[0]*A[2]]][sTc[A[0]]] + 0.3*X[sTr[A[0] * A[2]]][sTc[A[0]]]


    else:

        print("Constructing the SDP objective from the symbolic objective") 
        for termInSum in objective_.args:   #e.g. termInSum could look like a multiplication object with arguments (0.13,Π_4,Π_3) and so on    
            res = lettersForExpr(termInSum) #returns, e.g. [1.23,Π1,Π1] 
            #we need to access
            y += res[2] * X[sTr[res[0]]][sTc[res[1]]]
            del res


    print("Done")


# ## Observed Constraints

# In[29]:


#######################Constraints from the observations
######
#####
####
###
##
#
def init_observedConstraints():
    global N,I,A,B,L1,G,gDicts,objective_,Lx,L,L_,sTi,sTr,sTc,Lloc,X,X1,constraints,locConstraints,y,obsConstraints,data
    
    obsConstraints=[]
    print("Observed values; Adding constraints")
    #If the parameters are not defined, define them
    #If they are defined, just update them
    try:
        if(data is None):
            data=cp.Parameter(N)
    except:
        data=cp.Parameter(N)

    #a=1/np.sqrt(5) - 0.00001 #0.000001 #0.03
    # a=1/np.sqrt(5)
    # a=0.44
    # data.value = [a,a,a,a,a]

    # data.value = [0.446, 0.452, 0.446, 0.441, 0.450] #Experimental
    if(len(gls['obsData']) == N):
        data.value = gls['obsData']

        j=0

        for l1 in L1[:N]:
            obsConstraints += [ X[sTr[l1],sTc[l1]] == data[j] ]
            j+=1    
        del j
        print("Done.")        
    else:
        print("No observed constraints added (it may be that the sizes didn't match)")

    
    #print(obsConstraints)


# # SDP, solving

# ## Construct problem for CVXPY

# In[30]:



def init_SDP():
    global N,I,A,B,L1,G,gDicts,objective_,Lx,L,L_,sTi,sTr,sTc,Lloc,X,X1,constraints,locConstraints,y,obsConstraints,data
    global probMax,probMin

    #Possible solvers 
    #MOSEK needs a license; will set it up later 

    #SCS 
    #CVXOPT 
    #MOSEK 

    verbose=True 
    currentConstraints = [] 


    if(gls['testType']==0 or gls['testType']==2):    
        if(gls['usePs']==True):
            currentConstraints += locConstraints
        currentConstraints += constraints+obsConstraints 
        #currentConstraints+=obsConstraints

    elif(gls['testType']==1):
        currentConstraints = constraints
        if(gls['testSubType']=="b" or gls['testSubType']=="c" or gls['testSubType']=="d"        or gls['testSubType']=="f" or gls['testSubType']=="a2" or gls['testSubType']=="e" or gls['testSubType']=="ks"):
            #y=0
            currentConstraints += obsConstraints
        if(gls['testSubType']=="c"): # or testSubType=="d"):
            currentConstraints += locConstraints
            if(gls['usePs']==False):
                print("WARNING: Ps are off while the test needs Ps!")


    probMax = cp.Problem(cp.Maximize(y),currentConstraints)
    probMin = cp.Problem(cp.Maximize(-y),currentConstraints)

    # probMax = cp.Problem(cp.Maximize(0),currentConstraints)
    # probMin = cp.Problem(cp.Maximize(-0),currentConstraints)


    # 
    # ## Solve

# In[31]:


def solve_SDP():
    global probMax,probMin,X
    
    try: 
        print("Initialising...")
        print("max:",probMax.solve(**gls['solverSettings']))#(verbose=True,solver=cp.MOSEK)) #,warm_start=True))#,eps=1e-5)) #,eps=1e-4))#,max_iters=100000))
        print("sqrt(5)",np.sqrt(5))
        if(gls['solverSettings']['verbose']):
            prettyPrint(X)        
    except:
        print("max: Couldn't solve")


    try:
        print("Initialising...")
        print("min:",-probMin.solve(**gls['solverSettings']))#(verbose=True,solver=cp.SCS,max_iters=16*2500,warm_start=True)) #))#,eps=1e-5)) #,eps=1e-4))#,max_iters=100000))
        print("sqrt(5)",np.sqrt(5))
        if(gls['solverSettings']['verbose']):
            prettyPrint(X)
    except:
        print("min: Couldn't solve")


# # Aggregated Init and Solve

# ## Definitions

# In[32]:


def printStep(i):
    global totSteps
    
    print("STEP ",i,"/",totSteps," in progress...")

def printStepDone(i):
    global totSteps
    
    print("STEP ",i,"/",totSteps," done.\n\n")

    
def init():
    global totSteps
    
    totSteps = 5
    
    printStep(1)
    print("Initialising Ideal KCBS calculations and performing symbolic computations")
    init_iKCBS()
    printStepDone(1)
    
    printStep(2)
    print("Preparing: Sympy meets Cvxpy")
    init_SympyCvxpy()
    printStepDone(2)
    
    printStep(3)
    print("Adding constraints from QM to the SDP variables")
    init_QMconstraintsToSDP()
    printStepDone(3)
    
    printStep(4)
    print("Creating the objective in terms of the SDP variables")
    init_symObjectiveToSDP()
    printStepDone(4)
    
    printStep(5)
    print("Adding observed constraints")
    init_observedConstraints()
    printStepDone(5)

    
def solve(obsData=None):
    if(obsData is not None):
        gls['obsData']=obsData
        
    global totSteps
    totSteps=3
    
    printStep(1)
    print("Updating observed constraints")
    init_observedConstraints()
    printStepDone(1)
    
    printStep(2)
    print("Readying the solver")
    init_SDP()
    printStepDone(2)
    
    printStep(3)
    print("Solving now")
    solve_SDP()
    printStepDone(3)


# ## Aggregated Test
# 
# Uncomment to enable

# In[ ]:


globalSettings({'cacheFolder':'cached_2/','N':7}) #,'solverSettings':{'verbose':True,'solver':cp.MOSEK}})
init()
solve()


# In[ ]:


# 7.23 expected

