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
#from IPython.display import display


import pickle
import dill
dill.settings['recurse'] = True


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
    


# ## Extending the braket library to include orthogonality

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


# ### Debugging | representation example

# In[ ]:


class SzUpKet(sq.Ket):
    def _represent_SzUpKet(s,basis,**options):
        return sp.Matrix([1,0])
    
#     def _represent_SzOp(s,basis,**options):
#         return sp.Matrix([1,0])

class SzOp(sq.Operator):
    pass

sz = SzOp('Sz')
up = SzUpKet('up')
sq.represent(up,basis=SzUpKet) #,basis=sz)


# ### Testing | Dimension increasing?

# In[ ]:


# class bKet(sq.Ket):
    
#     #total number of kets (automatically increases as new instances are initialised)
#     totalKets=0
    
#     #this method automates the increase
#     @classmethod
#     def _eval_args(cls,args):
#         #validate input type
#         if not isinstance(args[0],int):
#             raise ValueError("Integer expected in the argument, got: %r"%args[0])

#         #if a ket |5> is initialised, the dimension is assumed to be 6 since we have |0>,|1>,...|5>
#         if int(args[0])+1 > cls.totalKets:
#             cls.totalKets=args[0]+1
        
#         #pass control back to the base class
#         return sq.Ket._eval_args(args) 
    
#     #Could not find a way of using just one function; need to go through the two
#     #based on a combination of the qubit implementation and the 
#     def _represent_default_basis(s,**options):
#         return s._represent_bKet(None,**options) #syntax taken from the Qubit library
    
#     def _represent_bKet(s,basis,**options): #_represent_default_basis
#         a=np.zeros(s.__class__.totalKets,dtype=int)
#         a[s.label[0]]=1
#         return sp.Matrix(a)
    
#     @classmethod
#     def set_dimension(cls,arg):
#         cls.totalKets=arg
    
#     def _eval_innerproduct(self,bra,**hints):
#         #Check if the innerproduct is with yourself, then set 1
#         if sq.Dagger(bra) == self:
#             return sp.Integer(1)
#         #Else, if the innerproduct is with a neighbor from the graph, set 0
#         else:
#             return sp.Integer(0)
#         #Else, do nothing; not enough information
    

#     @classmethod
#     def dual_class(self):
#         return bBra

    
# class bBra(sq.Bra):
#     # @classmethod
#     # def _eval_args(cls,args):
#     #     return args
    
#     @classmethod
#     def dual_class(cls):
#         return bKet


# In[ ]:


a=bKet(0)

b=a*sq.Dagger(a)

print(bKet.totalKets)
sq.represent(a)

c=bKet(3);
sq.represent(a)
sq.represent(b)
sq.represent(c)


# In[ ]:


a=bKet(0); a


# In[ ]:


sq.represent(a)


# In[ ]:


b=bKet(1); b


# In[ ]:


sq.represent(b)


# In[ ]:


sq.represent(a)


# In[ ]:


c=bKet(3); c


# In[ ]:


sq.represent(a)


# ## Extending the tensor product functionality—tsimp and tdsimp

# In[5]:



def powerDrop(expr):
    if isinstance(expr,sp.Pow): #TODO: make sure the base is not too complex
        # print("PowerEncountered")
        if expr.exp>=2:
            # print("glaba")
            # #display(expr.base)
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
    ##display(expr)
    
    
    if isinstance(expr,sq.TensorProduct):
        new_args=[]
        for _ in expr.args:
            ##display(_)
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


# ## Find Coeff was not reliable

# In[6]:


# depth=0
# depthThresh=20

def findCoeff(obj,lett):
#     global depth
    
#     print("Parent object:")
#     #display(obj)
    
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
#                 #display(_)
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


# ## Power to Mul

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


# In[11]:


def pow_to_single_mul(expr):
    return mul_to_single_mul(pow_to_mul(expr))


# # Ideal KCBS calculations

# In[12]:



class cls_iKCBS:    
    
    def __init__(s,n,verbose=False,thr=10,test=False,forceCalc=False):
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
            #display(sq.represent(s.T))

        
        #s.mT = s.mkp[0]*s.mbp[2] + s.mkp[1]*s.mbp[0] + s.mkp[2]*s.mbp[1] 
        #s.mTi = s.mkp[2]*s.mbp[0] + s.mkp[0]*s.mbp[1] + s.mkp[1]*s.mbp[2] #TODO: Automate this
        
        s.matT=sq.represent(s.T)
        
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
        if(test):
            print("Converting the numerical solution into a usable form")
            numTDict={}
            for i in range(n):
                numTDict[s._Πs[i]]=s.Πs[i]
            ##display(s._T)
            numT = s._T.subs(numTDict)
            ##display(numT)
            print("Testing the numerically found Translation operator.")
            # for braP in s.bp:
            #     for ketP in s.kp:
            #         print(sq.qapply(braP*numT*ketP)) #sp.N(sq.qapply(braP*ketP)))
            numMatT=sq.represent(numT)
            #display(sp.Transpose(numMatT) * numMatT)
            
            #display(sq.represent(numT))
            

        if verbose==True:
            print("Solving the cross Π")
        
        #s._Πcross = s.eval_Π_cross(RHS=sq.represent(s.kp[2]*s.bp[2])) #s.eval_lin_c(RHS=sq.represent(s.kp[2]*s.bp[2]))
        s._Πcross = s.eval_lin_c(RHS=sq.represent(s.kp[2]*s.bp[2]))
            
        if verbose==True:
            print("Evaluating the first term of F")

        if(test):
            s.Πcross = s.eval_lin_c(RHS=sq.represent(s.kp[2]*s.bp[2]),retIde=True)
            s.test_swapGate()
            
        if(not test):
            
            try:
                y=dill.load(open("Fsaved2", "rb"))
                s.F=y
                print("Loaded from file")
            except:
                y=None
            
            if y==None or forceCalc==True:
                print("Evaluating F")
                s.eval_state_hw()
                s.Fs = [s.eval_state_F()]                        
                for i in range(n):
                    print("iteration: ",i)
                    s.Fs.append(s.Fs[-1]+5.0*s.eval_state_F(s.Πs[i],s._Πs[i]))
                s.F = s.Fs[-1] #last element, the most recent calculation
                
                dill.dump(s.F, open("Fsaved2", "wb"))
                print("Saved to disk")
        
        print("done")
        
    def test_swapGate(s):
        s.TP = sq.TensorProduct 
        s.T = s.kp[0]*s.bp[2] + s.kp[1]*s.bp[0] + s.kp[2]*s.bp[1] 
        s.Ti = s.kp[2]*s.bp[0] + s.kp[0]*s.bp[1] + s.kp[1]*s.bp[2] #TODO: Automate this
        
        Id_ = s.kp[0]*s.bp[0] + s.kp[1]*s.bp[1] + s.kp[2]*s.bp[2]
        
        dontDie=prune(sp.N(sq.qapply(s.T*s.T)))
        
        _U_=sq.represent(s.TP(Id_,s.kp[0]*s.bp[0])) + sq.represent(s.TP(s.T,s.kp[1]*s.bp[1]))         + sq.represent(s.TP(dontDie,s.kp[2]*s.bp[2]))
        #display(_U_)
        
        _T_=sq.represent(s.TP(Id_,s.kp[0]*s.bp[0] + s.kp[2]*s.bp[1] + s.kp[1]*s.bp[2]))
        #display(_T_)        
        
        #_V_= sq.represent(s.TP(s.kp[0]*s.bp[0],Id_) + s.TP(s.kp[1]*s.bp[1],s.Ti) + s.TP(s.kp[2]*s.bp[2],s.Ti*sp.N(s.Ti)))
        _V_ = sq.represent(s.TP(s.Πs[0],Id_) + s.TP(s.Πs[1],s.Ti) + s.TP(s.Πcross,s.Ti*sp.N(s.Ti)))
        #display(_V_)
        
        print("comparing the projectors with the ket based projectors")
        
        #display(sq.represent(s.kp[0]*s.bp[0]))
        #display(sq.represent(s.Πs[0]))

        #display(sq.represent(s.kp[1]*s.bp[1]))
        #display(sq.represent(s.Πs[1]))

        #display(sq.represent(s.kp[2]*s.bp[2]))
        #display(sq.represent(s.Πcross))
        
        print("did this work?")
        
        
        SWAP = _T_*_U_*_V_*_U_
        #display(SWAP)
        
        testVec=sq.represent(s.TP(s.ke[0],s.ke[1]))
        print("Multiplying this")
        #display(testVec)
        #display(SWAP*testVec)
        
    def eval_state_hw(s):
        s.TP = sq.TensorProduct
        #dontDie=prune(sp.N(sq.qapply(s._T*s._T)))
        
        #s._U_=tsimp(s.TP(sp.Integer(1),prune(sp.N(s.kp[0]*s.bp[0]))) + s.TP(s._T,prune(sp.N(s.kp[1]*s.bp[1]))) + s.TP(dontDie,prune(sp.N(s.kp[2]*s.bp[2]))))
        s._U_=tsimp(s.TP(sp.Integer(1),prune(sp.N(s.kp[0]*s.bp[0]))) + s.TP(s._Pd,prune(sp.N(s.kp[1]*s.bp[1]))) + s.TP(s._Pd * s._Pd,prune(sp.N(s.kp[2]*s.bp[2]))))
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
        # ##display(_U_)
        # G00 = tsimp(_U_*_kout_).subs(s.gDict)
        # print("2.1")
        # G01 = tsimp(_V_*G00).subs(s.gDict)
        # print("2.2")
        # #G0 = tsimp(_V_*_U_)
        # t1=sq.qapply(F1*G01).subs(s.gDict)
        # print("2.3")
        
        G0=tsimp(s._V_*_kout_).subs(s.gDict)
        print("3")
        
        ##display(F1)
        ##display(G0)
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
        ##display(_TfΠ)
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
        J=sp.Integer(j)
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
        ##display(sq.represent(projector))
        ##display(sq.represent(vec))
        
        if ve==False:
            return projector
        else:
            return vec
        
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
        
        for i in range(1,cls.n):
            TfΠ.append(TfΠ[-1] + a[i]*s.Πs[i])
            _TfΠ.append(_TfΠ[-1] + a[i]*s._Πs[i])

        if cls.n != 5:            
            for i in range(cls.n,varCount):
                TfΠ.append(TfΠ[-1] + a[i]*s.Πs[i%cls.n]*s.Πs[(i+2)%cls.n])
                _TfΠ.append(_TfΠ[-1] + a[i]*s._Πs[i%cls.n]*s._Πs[(i+2)%cls.n])
        else:
            TfΠ.append(TfΠ[-1] + a[5]*s.Πs[4]*s.Πs[2]                     + a[6]*s.Πs[4]*s.Πs[1]                     + a[7]*s.Πs[1]*s.Πs[3]                     + a[8]*s.Πs[2]*s.Πs[0])
            _TfΠ.append(_TfΠ[-1] + a[5]*s._Πs[4]*s._Πs[2]                     + a[6]*s._Πs[4]*s._Πs[1]                     + a[7]*s._Πs[1]*s._Πs[3]                     + a[8]*s._Πs[2]*s._Πs[0])
            
            
        LHS=sq.represent(TfΠ[-1])
        ##display(_TfΠ)
        # LHS=a[0]*s.matΠs[0]
        # for i in range(1,cls.n):
        #     LHS=LHS+a[i]*s.matΠs[i]
        # for i in range(cls.n,varCount):
        #     LHS=LHS+a[i]*s.matΠs[i%cls.n]*s.matΠs[(i+2)%cls.n]
        if RHS == None:
            RHS=s.matT
        
        if cls.n!=5:
            soln=sp.solve(sp.N(LHS-RHS),a,dict=True)
        else:
            soln=sp.solve(sp.N(LHS-RHS),a[:9],dict=True)
        
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
        


# In[13]:


iKCBS = cls_iKCBS(5,verbose=True) #,test=False,forceCalc=True)


# In[14]:


##display(sp.simplify(iKCBS.F.subs(iKCBS.sqDict)))
# #display(iKCBS.F.subs(iKCBS.sqDict))
#The orthogonal relations are imposed later using the graph


# # SDP part (SymPy meets CvxPy)

# In[15]:


N=5

#Ψ=[sp.symbols('Ψ')]
I=[sp.Integer(1)]
A=list(sp.symbols('Π_0:%d'%N,commutative=False))
B=[sp.symbols('P',commutative=False),sp.symbols('Pd',commutative=False)]

#A=list(sp.symbols('Π_0:%d'%N,commutative=False))

# #display(I)
# #display(A)
print(B)


# In[16]:


objective = iKCBS.F

# objective1=iKCBS.F

# #objective = A[0] + A[1] + A[2] + A[3] + A[4]
# objective2 = A[0]*A[0] + 1e-13*A[1] + 1e-13*A[2] + 1e-13*A[3] + 1e-13*A[4]

# objective=objective
##display(objective);


# In[17]:


hDepth = 3 #depth of the heirarchy; depends on the number of Πs in the expression for Fidelity

L1=A + B #appends the lists
#print(L1)


##display(Lx)

#These will help simplify A^3 to A when needed
lDepth = 2
sqDicts={} #{A[i]**2:A[i] for i in range(len(L1))}
for iA in A:
    sqDicts[iA**5]=iA
    sqDicts[iA**4]=iA
    sqDicts[iA**3]=iA
    sqDicts[iA**2]=iA

sqDicts[B[0]*B[1]]=sp.Integer(1)
sqDicts[B[1]*B[0]]=sp.Integer(1)




# L2=[]
# for let1 in L1:
#     for let2 in L1:
#         L2.append(let1*let2)




# L = I + L1 + L2
# L_ = L1 + L2
#print(L)


# In[18]:


G = nx.Graph()

#node2 in G[node1] #yes if they are connected by an edge

G = nx.Graph()
G.add_nodes_from(L1)
G.add_edges_from([[L1[i],L1[(i+1)%(N)]] for i in range(N)])

#create a dictionary to simplify expressions later which sets projector products to zero
gDicts={}
for l1 in L1:
    for l2 in L1:
        if l1 in G[l2]:
            gDicts.update({l1*l2:sp.Integer(0)})


get_ipython().run_line_magic('matplotlib', 'inline')
nx.draw(G, with_labels=True)


# In[19]:


#Simplify objective using the orthogonality relations of the graph
##display(prune(objective).subs(gDicts))  
objective_ = sp.expand(prune(objective).subs(gDicts)).subs(gDicts).subs(sqDicts)
# #display(objective_)


# In[20]:


Lx=[]

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
        val_=val.subs(gDicts).subs(sqDicts).subs(sqDicts).subs(gDicts)
        if not val_ in Lx and val_ != 0 :
            Lx.append(val_)

for i in range(2,hDepth+1):    
    #loop_rec(N,i)
    loop_rec(len(L1),i)

    
L = I + L1 + Lx
L_ = L1 + Lx

print(L1)
# #display(Lx)


# In[21]:


#symbol to index dictionary
sTi={}

for i in range(len(L)):
    sTi[L[i]]=i

#print(sTi)


# In[22]:


print(Lx)


# ## Sympy says hello to Cvxpy

# In[23]:


#A function for directly extracting terms to construct indices

lI = I[0]


def lettersForExpr(termInSum):
    termInSum_ = pow_to_single_mul(termInSum) #sp.expand(pow_to_mul(termInSum))
    print(termInSum_)
    print(termInSum_.args)
    if(len(termInSum_.args[1:])<= hDepth  ): #if we can handle this using just the letters we produced
        lett = termInSum_.args[1]
        for lettInTerm in termInSum_.args[2:]:
            lett *= lettInTerm
        coeff = sp.N(termInSum_.args[0]) #In case we got symbolic square roots

        lett=lett.subs(sqDicts)            
        return[lett,lI,coeff]
        #eX = X[sTi[lett]][sTi[lI]]
        #y += coeff*eX

        ##display(termInSum_)
        ##display(lett) #These can be directly handled

    elif(len(termInSum_.args[1:])<= 2*hDepth ): #we can handle at most twice the letters            
        half=int(hDepth) +1
        #print(len(termInSum_.args[1:]),2*hDepth)
        #print(half)
        #print(termInSum_)
        lett1 = termInSum_.args[1]
        for lettInTerm in termInSum_.args[2:half]:
            lett1 *= lettInTerm
        lett2 = termInSum_.args[half]
        for lettInTerm in termInSum_.args[half+1:]:
            lett2 *= lettInTerm
        # #display(termInSum_)
        # #display(lett1) 
        # #display(lett2)
        coeff = sp.N(termInSum_.args[0]) #In case we got symbolic square roots
        lett1=lett1.subs(sqDicts)
        lett2=lett2.subs(sqDicts)
        return [lett1,lett2,coeff]
        #eX = X[sTi[lett1]][sTi[lett2]]
        #y += coeff*eX
    else:
        print("objective not fully captured")           
        return ["objective not fully captured"]
        #flag="Objective not fully captured"
        #raise Warning("Increase hDepth: It looks like your objective is not captured by the current depth of the SDP heirarchy")


# In[32]:


ForceCalc=False
fileName="lSave_3A"

flags=["issues"]

###
#objective_ = sp.expand(5.0*(0.1234*A[0]*A[2]*A[0] + 0.1234*A[1]*A[3]*A[1]*A[3]*A[1]*A[3]*A[1]))

#objective_

##display(objective_)

ConstraintsThreshold = 10000 #Maximum number of constraints
useThreshold=False

try:
    print("Loading from file")
    loadedThis=dill.load(open(fileName, "rb"))
    [X,y,constraints]=loadedThis
    print("Done")
    print("SDP Size",X[0])
    print("Constrainst:", len(constraints))
except:
    loadedThis=None

 
    
    
if loadedThis==None or ForceCalc==True:
    print("Evaluating constraints")


    #L is the alphabet, to wit: it is the list of letters
    #
    X = cp.Variable((len(L),len(L)),symmetric=True)    

    #>> is for matrix inequality
    constraints = [X >> 0,X[sTi[lI],sTi[lI]]==1]

    #objective
    #y = objective.subs({A[0]:X[sTi[A[0]]],A[1]:X[sTi[A[1]]]})
    #y = objective.subs({A[0]:sTi[A[0]],A[1]:sTi[A[1]]})


    # objective function evaluated (loop over product of all letters)
    y = 0

#     print("Constructing the SDP objective from the symbolic objective")
#     termsUsed = []
#     for l1 in L:
#         print("row of ",l1)
#         for l2 in L:
#             if l1 != lI or l2 != lI: #this is because if we ask for the coefficient of 1, things get messed up
#                 term = l1*l2

#                 #the same term can be represented by many elements in the matrix
#                 #we satisfy ourselves with picking just one of them
#                 if not(term in termsUsed):
#                     termsUsed.append(term)                
#                     eX = X[sTi[l1]][sTi[l2]] #picks the corresponding element from the variable matrix of the SDP library
#                     #print(eX)
#                     coeff = sp.N(findCoeff(objective_,l1*l2))
#                     ##display(l1*l2)
#                     if isinstance(coeff,sp.Number): #evaluates the coff of l1*l2 in the objective
#                         #coeff = sp.N(objective.coeff(l1*l2)) #evaluates the coff of l1*l2 in the objective                        
#                         if(abs(coeff)>0):
#                             #display(coeff)
#                             y += coeff*eX
#                     #else:
#                         #print("strange coeff")
#                         ##display(coeff)
#     print("done")

#     #A function for directly extracting terms to construct indices
#     def lettersForExpr(termInSum):
#         termInSum_ = pow_to_single_mul(termInSum) #sp.expand(pow_to_mul(termInSum))
#         print(termInSum_)
#         print(termInSum_.args)
#         if(len(termInSum_.args[1:])<= hDepth  ): #if we can handle this using just the letters we produced
#             lett = termInSum_.args[1]
#             for lettInTerm in termInSum_.args[2:]:
#                 lett *= lettInTerm
#             coeff = sp.N(termInSum_.args[0]) #In case we got symbolic square roots
            
#             lett=lett.subs(sqDicts)            
#             return[lett,lI,coeff]
#             #eX = X[sTi[lett]][sTi[lI]]
#             #y += coeff*eX
            
#             ##display(termInSum_)
#             ##display(lett) #These can be directly handled

#         elif(len(termInSum_.args[1:])<= 2*hDepth ): #we can handle at most twice the letters            
#             half=int(hDepth) +1
#             #print(len(termInSum_.args[1:]),2*hDepth)
#             #print(half)
#             #print(termInSum_)
#             lett1 = termInSum_.args[1]
#             for lettInTerm in termInSum_.args[2:half]:
#                 lett1 *= lettInTerm
#             lett2 = termInSum_.args[half]
#             for lettInTerm in termInSum_.args[half+1:]:
#                 lett2 *= lettInTerm
#             # #display(termInSum_)
#             # #display(lett1) 
#             # #display(lett2)
#             coeff = sp.N(termInSum_.args[0]) #In case we got symbolic square roots
#             lett1=lett1.subs(sqDicts)
#             lett2=lett2.subs(sqDicts)
#             return [lett1,lett2,coeff]
#             #eX = X[sTi[lett1]][sTi[lett2]]
#             #y += coeff*eX
#         else:
#             print("objective not fully captured")           
#             return ["objective not fully captured"]
#             #flag="Objective not fully captured"
#             #raise Warning("Increase hDepth: It looks like your objective is not captured by the current depth of the SDP heirarchy")
        
        
    
    print("Constructing the SDP objective from the symbolic objective") 
    for termInSum in objective_.args: #e.g. termInSum could look like a multiplication object with arguments (0.13,Π_4,Π_3) and so on
        flag="No obvious issues."        
        res = lettersForExpr(termInSum) #returns, e.g. [1.23,Π1,Π1]
        y += res[2] * X[sTi[res[0]]][sTi[res[1]]]
        
        #print(sp.expand(termInSum))
#         termInSum_ = pow_to_single_mul(termInSum) #sp.expand(pow_to_mul(termInSum))
#         print(termInSum_)
#         print(termInSum_.args)
#         if(len(termInSum_.args[1:])<= hDepth  ): #if we can handle this using just the letters we produced
#             lett = termInSum_.args[1]
#             for lettInTerm in termInSum_.args[2:]:
#                 lett *= lettInTerm
#             coeff = sp.N(termInSum_.args[0]) #In case we got symbolic square roots
#             eX = X[sTi[lett]][sTi[lI]]
#             y += coeff*eX
            
#             ##display(termInSum_)
#             ##display(lett) #These can be directly handled

#         elif(len(termInSum_.args[1:])<= 2*hDepth ): #we can handle at most twice the letters            
#             half=int(hDepth) +1
#             #print(len(termInSum_.args[1:]),2*hDepth)
#             #print(half)
#             #print(termInSum_)
#             lett1 = termInSum_.args[1]
#             for lettInTerm in termInSum_.args[2:half]:
#                 lett1 *= lettInTerm
#             lett2 = termInSum_.args[half]
#             for lettInTerm in termInSum_.args[half+1:]:
#                 lett2 *= lettInTerm
#             # #display(termInSum_)
#             # #display(lett1) 
#             # #display(lett2)
#             coeff = sp.N(termInSum_.args[0]) #In case we got symbolic square roots
#             eX = X[sTi[lett1]][sTi[lett2]]
#             y += coeff*eX
#         else:
#             flag="Objective not fully captured"
            #raise Warning("Increase hDepth: It looks like your objective is not captured by the current depth of the SDP heirarchy")
    print(flag)
    print("Done")
    #flags.append(flag)
    #len(objective_.args[13].args[1:])


    print("Adding constraints from the graph")
    #Constraints from the graph (these come from single letters)        
    for l1 in L1:
        for l2 in L1:
            eX = X[sTi[l1]][sTi[l2]] #picks the corresponding element from the variable matrix of the SDP library
            if l1 in G[l2]:
                constraints += [eX == 0]

    print("done")
            #print(coeff,sTi[l1])
            ##display(l1)

    #constraints from the fact that Π^2=Π
    # for l in L1:
    #     constraints += [ X[sTi[lI]][sTi[l]] == X[sTi[l]][sTi[l]] ]
    #     #constraints to match with experiment

    print("Adding constraints from Quantum Mechanics")
    tempCount=0
    dictTerms = {}
    halfLoop = 0
    #lettersUsed = []
    for l1 in L:
        print("Row of ",l1)
        if(tempCount>ConstraintsThreshold):
            break

        for l2 in L[halfLoop:]:
            term = l1*l2


            tS=[term]
            ##display(term)
            for i in range(lDepth+1):
                tS.append(tS[-1].subs(sqDicts))
            simTerm=tS[-1]
            ##display(simTerm)

            if not(simTerm in dictTerms):
                dictTerms.update({simTerm:[l1,l2]})            
            else:
                l1_=dictTerms[simTerm][0]
                l2_=dictTerms[simTerm][1]
                #print(l1_,l2_)            
                #print(sTi[l1_],sTi[l2_])

                #print(l1,l2)            
                #print(sTi[l1],sTi[l2])

                #constraints += [ X[sTi[l1]][sTi[l2]] == X[sTi[l1_]][sTi[l2_]] ] 
                # print(X[sTi[l1]][sTi[l2]])
                # print(X[sTi[l1_]][sTi[l2_]])
                
                #I am not sure why the or part of this statement is reducing constraints
                
                if (not (l1==l1_ and l2==l2_) or not (l1==l2_ and l2==l1_)):                
                    #if ((l1==l2_ and l2==l1_)):
#                         print("GOT ONE!")
#                         print("The ls")
#                         #display(l1)
#                         #display(l2)
                        
#                         print("The l_s")
#                         #display(l1_)
#                         #display(l2_)
                        
                    constraints += [ X[sTi[l1]][sTi[l2]] == X[sTi[l1_]][sTi[l2_]] ]
                    
                    tempCount+=1
                    
                    if(tempCount>ConstraintsThreshold and useThreshold==True):
                        break
                    
                    if(tempCount<10 or                       (100<tempCount and tempCount<110) or                       (1000<tempCount and tempCount<1010)):
                       
                        print("Adding a new constraint")
                        print("l1")
                        # #display(l1)
                        print("l2")
                        # #display(l2)
                        print("l1_")
                        # #display(l1_)
                        print("l2_")
                        # #display(l2_)
                        
                

            #Orthogonality from the graph
            if simTerm.subs(gDicts)==sp.Integer(0):
                constraints += [ X[sTi[l1]][sTi[l2]] == 0 ]
            del tS
        halfLoop+=1

    print("done")

    

    
    
    
    
    
    # #display(y)

    # #display(objective.args[0])
    # #display(objective.args[1])


    #expr=A[0]*A[0]*A[0]*A[1]*A[0]*A[1]




    #         term = l1*l2


    #         tS=[term]
    #         ##display(term)
    #         for i in range(lDepth+1):
    #             tS.append(tS[-1].subs(sqDicts))
    #         simTerm=tS[-1]
    #         ##display(simTerm)

    #         if not(simTerm in dictTerms):
    #             dictTerms.update({simTerm:[l1,l2]})            
    #         else:
    #             l1_=dictTerms[simTerm][0]
    #             l2_=dictTerms[simTerm][1]
    #             #print(l1_,l2_)            
    #             #print(sTi[l1_],sTi[l2_])

    #             #print(l1,l2)            
    #             #print(sTi[l1],sTi[l2])

    #             #constraints += [ X[sTi[l1]][sTi[l2]] == X[sTi[l1_]][sTi[l2_]] ] 
    #             # print(X[sTi[l1]][sTi[l2]])
    #             # print(X[sTi[l1_]][sTi[l2_]])
    #             constraints += [ X[sTi[l1]][sTi[l2]] == X[sTi[l1_]][sTi[l2_]] ]

    #         #Orthogonality from the graph
    #         if simTerm.subs(gDicts)==sp.Integer(0):
    #             constraints += [ X[sTi[l1]][sTi[l2]] == 0 ]
    #         del tS



    ##display(tS[-1])


    # powDicts={}
    # for i in range(len(L1)):
    #     powDicts.update({A[i]**k:A[i] for k in range(2,lDepth+4)[::-1]})


    #expr.subs(powDicts)

    # #display(expr.subs(sqDicts))
    # #display(expr.subs(powDicts).subs(powDicts))

    #expr

    #dict={A[i]**k:A[i] for k in range(lDepth)}

    #print(powDicts)

    #result=expr.subs({A[0]*A[0]:A[1]}); result
    #result=sp.refine(expr,)
    print("Constraints", len(constraints))
    print("SDP", X[0])
    dill.dump([X,y,constraints], open(fileName, "wb"))
    print("Saved to disk")


try:
    del fileName
    del ForceCalc
except:
    print("Cleaned.")


# In[25]:


#Localising Matrix Constraints

M = []

for lett1 in I + L1:
    row = []
    for lett2 in I + L1:
        expr = sp.expand(lett1*iKCBS._T*lett2).subs(gDicts).subs(sqDicts).subs(gDicts).subs(sqDicts)
#         print("the expression")
        # #display(expr)
        res = 0
        if expr != sp.Integer(0):
            #if there are many terms in the sum, sum over them
            if(isinstance(expr,sp.Add)):
                for _ in expr.args:
#                     print("Going in")
                    res_ = lettersForExpr(_)
#                     print("printing res ")
#                     print(res_)
                    __ = res_[2]*X[sTi[res_[0]]][sTi[res_[1]]]            
                    res += __ #this will contain something like 1.13*Π_1*Π_2 + 1.42*Π_1*Π_2
            #If it is just one term, compute directly
            else:
                res_ = lettersForExpr(expr)
#                 print("printing res ")
#                 print(res_)
                __ = res_[2]*X[sTi[res_[0]]][sTi[res_[1]]]            
                res += __ #this will contain something like 1.13*Π_1*Π_2 + 1.42*Π_1*Π_2

        row.append(res) #this appends that calculation into the row
    M.append(row) #this appends the row into the list M
        
lM = cp.bmat(M) #this converts the list M into a matrix which can be used to impose constraints in cvxpy


locConstraints = [lM >> 0]


# In[26]:


#######################Constraints from the observations
obsConstraints=[]
a=1/np.sqrt(5)-0.001
data = [a,a,a,a,a]

j=0
for l1 in L1[:4]:    
    obsConstraints += [ X[sTi[l1],sTi[l1]] == data[j] ]
    j+=1
    
del j


# In[27]:


probMin = cp.Problem(cp.Maximize(-y),constraints+obsConstraints + locConstraints)
probMax = cp.Problem(cp.Maximize(y),constraints+obsConstraints + locConstraints)
#prob.solve()
#prob.solve(solver=cp.CVXOPT,verbose=True,**{'':ldl,'refinement':20,'show_progress':True,'feastol':10e-10,'reltol':10e-10,'abstol':10e-10})
try: 
    print("max:",probMax.solve(solver=cp.CVXOPT,verbose=True))
except:
    print("max:gla")
try:
    print("min:",-probMin.solve(solver=cp.CVXOPT,verbose=True))
except:
    print("min:ba")


# In[238]:


#import cvxopt as co

