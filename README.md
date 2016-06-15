# Tarea5
import numpy as np
import math
import matplotlib.pyplot as plt
import pyfits
import random
import glob
from matplotlib.figure import Figure
import scipy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB



def columnas(filee,x,indicador):
##      Si indicador es 1, columna es float, si es 0, columna es string,si es 2, integer
        n=0
        file=open(filee)        
        read1=file.readline()
        while len(read1)>1:
            n=n+1
            read1=file.readline()              
        c=range(n)
        file.close()
        file=open(filee)        
        read=file.readline()
        i=0
        if indicador==1:
            while i<n:                    
                ID=read.split("\t")
                c[i]=float(ID[x-1])
                i=i+1
                read=file.readline()
        if indicador==0:
            while i<n:
                ID=read.split("\t")
                c[i]=ID[x-1]
                if c[i][len(c[i])-1]=='\n':
                    c[i]=substring(c[i],0,len(c[i])-2)
                i=i+1
                read=file.readline()
        if indicador==2:
            while i<n:
                ID=read.split("\t")
                c[i]=int(float(ID[x-1]))
                i=i+1
                read=file.readline()               
        file.close()
        return c
def Get_Data(filee):
    x=columnas(filee,1,1)
    y=columnas(filee,2,1)
    z=columnas(filee,3,2)
    return x,y,z


def Plot_Data(x,y,z,char):
    for i in xrange(0,len(x)):
        if z[i]==1:
            plt.plot(x[i],y[i],char,color='R')
        if z[i]==2:
            plt.plot(x[i],y[i],char,color='B')
def Class_Separate(x,y,z):
    Class1x=[]
    Class2x=[]
    Class1y=[]
    Class2y=[]
    for i in xrange(0,len(x)):
        if z[i]==1:
            Class1x.append(x[i])
            Class1y.append(y[i])
        if z[i]==2:
            Class2x.append(x[i])
            Class2y.append(y[i])
    return Class1x,Class1y,Class2x,Class2y

def Matr_build(x,y):
        N=len(x)
        Matr=range(N)
        for i in xrange(0,N):
                Matr[i]=[x[i],y[i]]
        return Matr
def Cov_Matr_build(M):
        return np.cov(M)
def prior(filee,Clase):
        A=Get_Data(filee)
        Class=A[2]
        N=len(Class)
        cont=0
        for i in xrange(0,N):
                if Class[i]==Clase:
                        cont=cont+1
        prior=cont*1./N
        return prior


def Get_Means(filee):
        A=Get_Data(filee)
        X=A[0]
        Y=A[1]
        C=A[2]
        M=Class_Separate(X,Y,C)
        x1=M[0]
        y1=M[1]
        x2=M[2]
        y2=M[3]
        Mu_x1=np.mean(x1)
        Mu_y1=np.mean(y1)
        Mu_x2=np.mean(x2)
        Mu_y2=np.mean(y2)
        p1=prior(filee,1)
        p2=prior(filee,2)
        Mu_x3=p1*Mu_x1+p2*Mu_x2
        Mu_y3=p1*Mu_y1+p2*Mu_y2
        return [Mu_x1,Mu_y1],[Mu_x2,Mu_y2],[Mu_x3,Mu_y3]
def Sum_Vec(A,B):
        A2=np.array(A)
        B2=np.array(B)
        return A2+B2
def Rest_Vec(A,B):
        A2=np.array(A)
        B2=np.array(B)
        return A2-B2
def Matrix_Sw(p1,cov1,p2,cov2):
        return p1*np.array(cov1)+p2*np.array(cov2)
def delta_k(prior,muKx,muKy,cov,x1,x2):
        Mu=[muKx,muKy]
        MuT=np.transpose(Mu)
        inv=np.linalg.inv(cov)
        X=np.array([x1,x2])
        Xt=np.transpose(X)
        return np.log(prior)-0.5*np.dot(Mu,np.dot(inv,Mu))+np.dot(Xt,np.dot(inv,Mu))

def delta_k_QDA(prior,muKx,muKy,cov,x1,x2):
        Mu=[muKx,muKy]
        MuT=np.transpose(Mu)
        inv=np.linalg.inv(cov)
        X=np.array([x1,x2])
        Xt=np.transpose(X)
        return np.log(prior)-0.5*np.log(np.linalg.det(cov))-0.5*np.dot(np.transpose(Rest_Vec(X,Mu)),np.dot(inv,Rest_Vec(X,Mu)))
def Conf_matr(n11,n12,n21,n22):
        M = [[0 for x in range(2)] for y in range(2)]
        M[0][0]=n11
        M[1][1]=n22
        M[0][1]=n12
        M[1][0]=n21
        return M
        
def Design_Matrix(x1,x2):
        n=len(x1)
        M = [[0 for x in range(3)] for y in range(n)]
        for i in xrange(0,n):
                M[i][0]=1
                M[i][1]=x1[i]
                M[i][2]=x2[i]
        return M

def Get_Coefs(x1,x2,Class):
        M=Design_Matrix(x1,x2)
        Mt=np.transpose(M)
        C=np.dot(Mt,M)
        inv=np.linalg.inv(C)
        D=np.dot(inv,Mt)
        return np.dot(D,Class)
def MAP_Linear_Reg(filee,x_lower,x_upper,y_lower,y_upper,Nx,Ny):
        D=Get_Data(filee)
        x1=D[0]
        x2=D[1]
        Class=D[2]
        Dat=Class_Separate(x1,x2,Class)
        C1x=Dat[0]
        C1y=Dat[1]
        C2x=Dat[2]
        C2y=Dat[3]
        Coefs=Get_Coefs(x1,x2,Class)
        a0=Coefs[0]
        a1=Coefs[1]
        a2=Coefs[2]
        x=np.linspace(x_lower,x_upper,Nx)
        y=np.linspace(y_lower,y_upper,Ny)
        POS=[]
        Boundx=[]
        Boundy=[]
        epsilon=0.01
        for i in xrange(0,Nx):
                for j in xrange(0,Ny):
                        POS.append([x[i],y[j]])
        n=len(POS)
        class_map=range(n)
        for i in xrange(0,n):
                if a0+a1*POS[i][0]+a2*POS[i][1]>1.5:
                        class_map[i]=2
                if a0+a1*POS[i][0]+a2*POS[i][1]<=1.5:
                        class_map[i]=1
                if abs(a0+a1*POS[i][0]+a2*POS[i][1]-1.5)<epsilon:
                        Boundx.append(POS[i][0])
                        Boundy.append(POS[i][1])
                        
        POSx=[]
        POSy=[]
        plt.plot(Boundx,Boundy,'G')
        for i in xrange(0,len(POS)):
                POSx.append(POS[i][0])
                POSy.append(POS[i][1])
        n1=len(C1x)
        n2=len(C2x)
        n11=0#clase 1 clasificados como 1
        n12=0#clase 1 clasificados como 2
        n21=0#clase 2 clasificados como 1
        n22=0#clase 2 clasificados como 2
        for i in xrange(0,n1):
                if a0+a1*C1x[i]+a2*C1y[i]<=1.5:
                        n11=n11+1
                if a0+a1*C1x[i]+a2*C1y[i]>1.5:
                        n12=n12+1
        for i in xrange(0,n2):
                if a0+a1*C2x[i]+a2*C2y[i]<=1.5:
                        n21=n21+1
                if a0+a1*C2x[i]+a2*C2y[i]>1.5:
                        n22=n22+1
        Conf_Matr=Conf_matr(n11,n12,n21,n22)
        traza=n11+n22
        N_tot=n11+n12+n21+n22
        misclasification_rate=1-1.*traza/N_tot
        
        return POS,class_map,POSx,POSy,Conf_Matr,misclasification_rate
def Map_LDA(filee,x_lower,x_upper,y_lower,y_upper,Nx,Ny):
        prior1=prior(filee,1)
        prior2=prior(filee,2)
        Means=Get_Means(filee)
        Mu1=Means[0]
        Mu2=Means[1]
        D=Get_Data(filee)
        x1=D[0]
        x2=D[1]
        Class=D[2]
        Dat=Class_Separate(x1,x2,Class)
        C1x=Dat[0]
        C1y=Dat[1]
        C2x=Dat[2]
        C2y=Dat[3]
        M1=Matr_build(C1x,C1y)
        M2=Matr_build(C2x,C2y)
        M_all=Matr_build(x1,x2)
        Cov1=Cov_Matr_build(np.transpose(M1))
        Cov2=Cov_Matr_build(np.transpose(M2))
        Cov_all=Cov_Matr_build(np.transpose(M_all))
        Cov_all=Matrix_Sw(prior1,Cov1,prior2,Cov2)
        x=np.linspace(x_lower,x_upper,Nx)
        y=np.linspace(y_lower,y_upper,Ny)
        POS=[]
        for i in xrange(0,Nx):
                for j in xrange(0,Ny):
                        POS.append([x[i],y[j]])
        n=len(POS)
        class_map=range(n)
        Boundx=[]
        Boundy=[]
        epsilon=0.2
        for i in xrange(0,n):
                d1=delta_k(prior1,Mu1[0],Mu1[1],Cov_all,POS[i][0],POS[i][1])
                d2=delta_k(prior2,Mu2[0],Mu2[1],Cov_all,POS[i][0],POS[i][1])
                if d1>d2:
                        class_map[i]=1
                if d2>=d1:
                        class_map[i]=2
                if abs(d2-d1)<=epsilon:
                        Boundx.append(POS[i][0])
                        Boundy.append(POS[i][1])
                
        POSx=[]
        POSy=[]
        plt.plot(Boundx,Boundy,'G')    
        for i in xrange(0,len(POS)):
                POSx.append(POS[i][0])
                POSy.append(POS[i][1])
        n1=len(C1x)
        n2=len(C2x)
        n11=0#clase 1 clasificados como 1
        n12=0#clase 1 clasificados como 2
        n21=0#clase 2 clasificados como 1
        n22=0#clase 2 clasificados como 2
        for i in xrange(0,n1):
                d1=delta_k(prior1,Mu1[0],Mu1[1],Cov_all,C1x[i],C1y[i])
                d2=delta_k(prior2,Mu2[0],Mu2[1],Cov_all,C1x[i],C1y[i])
                if d1>d2:
                        n11=n11+1
                if d2>=d1:
                        n12=n12+1
        for i in xrange(0,n2):
                d1=delta_k(prior1,Mu1[0],Mu1[1],Cov_all,C2x[i],C2y[i])
                d2=delta_k(prior2,Mu2[0],Mu2[1],Cov_all,C2x[i],C2y[i])
                if d1>d2:
                        n21=n21+1
                if d2>=d1:
                        n22=n22+1
        Conf_Matr=Conf_matr(n11,n12,n21,n22)
        traza=n11+n22
        N_tot=n11+n12+n21+n22
        misclasification_rate=1-1.*traza/N_tot    
        return POS,class_map,POSx,POSy,Conf_Matr,misclasification_rate,Boundx,Boundy
def MAP_QDA(filee,x_lower,x_upper,y_lower,y_upper,Nx,Ny):
        prior1=prior(filee,1)
        prior2=prior(filee,2)
        Means=Get_Means(filee)
        Mu1=Means[0]
        Mu2=Means[1]
        D=Get_Data(filee)
        x1=D[0]
        x2=D[1]
        Class=D[2]
        Dat=Class_Separate(x1,x2,Class)
        C1x=Dat[0]
        C1y=Dat[1]
        C2x=Dat[2]
        C2y=Dat[3]
        M1=Matr_build(C1x,C1y)
        M2=Matr_build(C2x,C2y)
        M_all=Matr_build(x1,x2)
        Cov1=Cov_Matr_build(np.transpose(M1))
        Cov2=Cov_Matr_build(np.transpose(M2))
        Cov_all=Cov_Matr_build(np.transpose(M_all))
        #Cov_all=Matrix_Sw(prior1,Cov1,prior2,Cov2)
        x=np.linspace(x_lower,x_upper,Nx)
        y=np.linspace(y_lower,y_upper,Ny)
        POS=[]
        for i in xrange(0,Nx):
                for j in xrange(0,Ny):
                        POS.append([x[i],y[j]])
        n=len(POS)
        class_map=range(n)
        Boundx=[]
        Boundy=[]
        epsilon=0.8
        for i in xrange(0,n):
                d1=delta_k_QDA(prior1,Mu1[0],Mu1[1],Cov1,POS[i][0],POS[i][1])
                d2=delta_k_QDA(prior2,Mu2[0],Mu2[1],Cov2,POS[i][0],POS[i][1])
                if d1>d2:
                        class_map[i]=1
                if d2>=d1:
                        class_map[i]=2
                if abs(d2-d1)<=epsilon:
                        Boundx.append(POS[i][0])
                        Boundy.append(POS[i][1])
        plt.plot(Boundx,Boundy,'g^')
        POSx=[]
        POSy=[]
        for i in xrange(0,len(POS)):
                POSx.append(POS[i][0])
                POSy.append(POS[i][1])
        n1=len(C1x)
        n2=len(C2x)
        n11=0#clase 1 clasificados como 1
        n12=0#clase 1 clasificados como 2
        n21=0#clase 2 clasificados como 1
        n22=0#clase 2 clasificados como 2
        for i in xrange(0,n1):
                d1=delta_k_QDA(prior1,Mu1[0],Mu1[1],Cov1,C1x[i],C1y[i])
                d2=delta_k_QDA(prior2,Mu2[0],Mu2[1],Cov2,C1x[i],C1y[i])
                if d1>d2:
                        n11=n11+1
                if d2>=d1:
                        n12=n12+1
        for i in xrange(0,n2):
                d1=delta_k_QDA(prior1,Mu1[0],Mu1[1],Cov1,C2x[i],C2y[i])
                d2=delta_k_QDA(prior2,Mu2[0],Mu2[1],Cov2,C2x[i],C2y[i])
                if d1>d2:
                        n21=n21+1
                if d2>=d1:
                        n22=n22+1
        Conf_Matr=Conf_matr(n11,n12,n21,n22)
        traza=n11+n22
        N_tot=n11+n12+n21+n22
        misclasification_rate=1-1.*traza/N_tot
        
        return POS,class_map,POSx,POSy,Conf_Matr,misclasification_rate,Boundx,Boundy
def indexOf(array,num):#index of first char in string, return -1 if not
    N=len(array)
    for i in xrange(0,N):
        if array[i]==num:
            return i
    return -1
def cosorting(array1,array2):#Ordena array1 primero, y despues ordena array 2 de la misma forma
    array1Ord,array2Ord=zip(*sorted(zip(array1,array2)))
    return array1Ord,array2Ord
def Reduce_array(x):#pasa de [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]  a [1,2,3,4]
        n=len(x)
        out=[]
        for i in xrange(0,n):
                if indexOf(out,x[i])==-1:
                        out.append(x[i])
        return out

def Boundarys(POSx,POSy,Class_map):
        POSx_aux=POSx
        POSx_ord,POSy_ord=cosorting(POSx,POSy)
        aux,Class_map_ord=cosorting(POSx_aux,Class_map)
        Bound=[]
        n=len(POSx_ord)
        POSx_ord_red=Reduce_array(POSx_ord)
        n_ele=len(POSx_ord_red)
        j=0
        #for i in xrange(0,n):
                
                
                
        
        
def MAP_Neight(filee,n_neight,x_lower,x_upper,y_lower,y_upper,Nx,Ny):
        neigh = KNeighborsClassifier(n_neighbors=n_neight)
        D=Get_Data(filee)
        x1=D[0]
        x2=D[1]
        Class=D[2]
        Dat=Class_Separate(x1,x2,Class)
        C1x=Dat[0]
        C1y=Dat[1]
        C2x=Dat[2]
        C2y=Dat[3]
        M_all=Matr_build(x1,x2)
        neigh.fit(M_all, Class)
        x=np.linspace(x_lower,x_upper,Nx)
        y=np.linspace(y_lower,y_upper,Ny)
        POS=[]
        for i in xrange(0,Nx):
                for j in xrange(0,Ny):
                        POS.append([x[i],y[j]])
        n=len(POS)
        class_map=range(n)
        for i in xrange(0,n):
                if neigh.predict([POS[i][0],POS[i][1]])<1.5:
                        class_map[i]=1
                if neigh.predict([POS[i][0],POS[i][1]])>=1.5:
                        class_map[i]=2
        POSx=[]
        POSy=[]
        for i in xrange(0,len(POS)):
                POSx.append(POS[i][0])
                POSy.append(POS[i][1])

        n1=len(C1x)
        n2=len(C2x)
        n11=0#clase 1 clasificados como 1
        n12=0#clase 1 clasificados como 2
        n21=0#clase 2 clasificados como 1
        n22=0#clase 2 clasificados como 2
        for i in xrange(0,n1):
                if neigh.predict([C1x[i],C1y[i]])<1.5:
                        n11=n11+1
                if neigh.predict([C1x[i],C1y[i]])>=1.5:
                        n12=n12+1
        for i in xrange(0,n2):
                if neigh.predict([C2x[i],C2y[i]])<1.5:
                        n21=n21+1
                if neigh.predict([C2x[i],C2y[i]])>=1.5:
                        n22=n22+1
        Conf_Matr=Conf_matr(n11,n12,n21,n22)
        traza=n11+n22
        N_tot=n11+n12+n21+n22
        misclasification_rate=1-1.*traza/N_tot
        
        return POS,class_map,POSx,POSy,Conf_Matr,misclasification_rate

def MAP_GaussianNB(filee,x_lower,x_upper,y_lower,y_upper,Nx,Ny):
        
        D=Get_Data(filee)
        x1=D[0]
        x2=D[1]
        Class=D[2]
        Dat=Class_Separate(x1,x2,Class)
        C1x=Dat[0]
        C1y=Dat[1]
        C2x=Dat[2]
        C2y=Dat[3]
        M_all=Matr_build(x1,x2)
        clf = GaussianNB()
        clf.fit(M_all,Class)
        x=np.linspace(x_lower,x_upper,Nx)
        y=np.linspace(y_lower,y_upper,Ny)
        POS=[]
        for i in xrange(0,Nx):
                for j in xrange(0,Ny):
                        POS.append([x[i],y[j]])
        n=len(POS)
        class_map=range(n)
        for i in xrange(0,n):
                if clf.predict([POS[i][0],POS[i][1]])<1.5:
                        class_map[i]=1
                if clf.predict([POS[i][0],POS[i][1]])>=1.5:
                        class_map[i]=2
        POSx=[]
        POSy=[]
        for i in xrange(0,len(POS)):
                POSx.append(POS[i][0])
                POSy.append(POS[i][1])

        n1=len(C1x)
        n2=len(C2x)
        n11=0#clase 1 clasificados como 1
        n12=0#clase 1 clasificados como 2
        n21=0#clase 2 clasificados como 1
        n22=0#clase 2 clasificados como 2
        for i in xrange(0,n1):
                if clf.predict([C1x[i],C1y[i]])<1.5:
                        n11=n11+1
                if clf.predict([C1x[i],C1y[i]])>=1.5:
                        n12=n12+1
        for i in xrange(0,n2):
                if clf.predict([C2x[i],C2y[i]])<1.5:
                        n21=n21+1
                if clf.predict([C2x[i],C2y[i]])>=1.5:
                        n22=n22+1
        Conf_Matr=Conf_matr(n11,n12,n21,n22)
        traza=n11+n22
        N_tot=n11+n12+n21+n22
        misclasification_rate=1-1.*traza/N_tot
        

        return POS,class_map,POSx,POSy,Conf_Matr,misclasification_rate
def Posteriori(x,y,mux,muy,cov_matr,prior):
        X=[x,y]
        Mu=[mux,muy]
        Z=Rest_Vec(X,Mu)
        Zt=np.transpose(Z)
        inv=np.linalg.inv(cov_matr)
        det=np.linalg.det(cov_matr)
        fact=1./(math.sqrt(((2.*np.pi)**2)*det))
        exponente=-0.5*np.dot(Zt,np.dot(inv,Z))
        return prior*fact*(np.e**exponente)
        
        
        
def Bayesian_Clasification(filee,MuX_1,MuY_1,Sig11_1,Sig12_1,Sig21_1,Sig22_1,MuX_2,MuY_2,Sig11_2,Sig12_2,Sig21_2,Sig22_2,x_lower,x_upper,y_lower,y_upper,Nx,Ny,):
        Cov_Matr_1=Conf_matr(Sig11_1,Sig12_1,Sig21_1,Sig22_1)
        Cov_Matr_2=Conf_matr(Sig11_2,Sig12_2,Sig21_2,Sig22_2)
        prior1=prior(filee,1)
        prior2=prior(filee,2)
        #Means=Get_Means(filee)
        #Mu1=Means[0]
        #Mu2=Means[1]
        D=Get_Data(filee)
        x1=D[0]
        x2=D[1]
        Class=D[2]
        Dat=Class_Separate(x1,x2,Class)
        C1x=Dat[0]
        C1y=Dat[1]
        C2x=Dat[2]
        C2y=Dat[3]
        x=np.linspace(x_lower,x_upper,Nx)
        y=np.linspace(y_lower,y_upper,Ny)
        POS=[]
        for i in xrange(0,Nx):
                for j in xrange(0,Ny):
                        POS.append([x[i],y[j]])
        n=len(POS)
        class_map=range(n)
        for i in xrange(0,n):
                if Posteriori(POS[i][0],POS[i][1],MuX_1,MuY_1,Cov_Matr_1,prior1)>Posteriori(POS[i][0],POS[i][1],MuX_2,MuY_2,Cov_Matr_2,prior2):
                        class_map[i]=1
                if Posteriori(POS[i][0],POS[i][1],MuX_1,MuY_1,Cov_Matr_1,prior1)<=Posteriori(POS[i][0],POS[i][1],MuX_2,MuY_2,Cov_Matr_2,prior2):
                        class_map[i]=2
        POSx=[]
        POSy=[]
        for i in xrange(0,len(POS)):
                POSx.append(POS[i][0])
                POSy.append(POS[i][1])
        n1=len(C1x)
        n2=len(C2x)
        n11=0#clase 1 clasificados como 1
        n12=0#clase 1 clasificados como 2
        n21=0#clase 2 clasificados como 1
        n22=0#clase 2 clasificados como 2
        for i in xrange(0,n1):
                if Posteriori(C1x[i],C1y[i],MuX_1,MuY_1,Cov_Matr_1,prior1)>Posteriori(C1x[i],C1y[i],MuX_2,MuY_2,Cov_Matr_2,prior2):
                        n11=n11+1
                if Posteriori(C1x[i],C1y[i],MuX_1,MuY_1,Cov_Matr_1,prior1)<=Posteriori(C1x[i],C1y[i],MuX_2,MuY_2,Cov_Matr_2,prior2):
                        n12=n12+1
        for i in xrange(0,n2):
                if Posteriori(C2x[i],C2y[i],MuX_1,MuY_1,Cov_Matr_1,prior1)>Posteriori(C2x[i],C2y[i],MuX_2,MuY_2,Cov_Matr_2,prior2):
                        n21=n21+1
                if Posteriori(C2x[i],C2y[i],MuX_1,MuY_1,Cov_Matr_1,prior1)<=Posteriori(C2x[i],C2y[i],MuX_2,MuY_2,Cov_Matr_2,prior2):
                        n22=n22+1
        Conf_Matr=Conf_matr(n11,n12,n21,n22)
        traza=n11+n22
        N_tot=n11+n12+n21+n22
        misclasification_rate=1-1.*traza/N_tot

        

        return POS,class_map,POSx,POSy,Conf_Matr,misclasification_rate
        
        
def All_task():
        filee='datos_clasificacion.dat'
        x_lower=-4
        x_upper=12
        y_lower=-4
        y_upper=10
        Nx=50
        Ny=50
        n_neigh1=10
        n_neigh2=30
        n_neigh3=60

        Sig11_1=5
        Sig12_1=-2
        Sig21_1=-2
        Sig22_1=5    
        MuX_1=2
        MuY_1=3

        Sig11_2=1
        Sig12_2=0
        Sig21_2=0
        Sig22_2=1
        MuX_2=6
        MuY_2=6
        ###################################################################################
        H=Map_LDA(filee,x_lower,x_upper,y_lower,y_upper,Nx,Ny)
        print 'Confusion Matrix for LDA= '+str(H[4])
        print 'Misclassification rate for LDA= '+str(H[5])
        POS=H[0]
        POSx=H[2]
        POSy=H[3]
        Class_map=H[1]
        Plot_Data(POSx,POSy,Class_map,'+')
        A=Get_Data(filee)
        x=A[0]
        y=A[1]
        z=A[2]
        Plot_Data(x,y,z,'o')
        plt.title('LDA classification, misclassification rate= '+str(H[5]))
        plt.xlabel('Feature x1')
        plt.ylabel('Feature x2')
        plt.savefig('LDA_Classification.png')
        plt.close()
        ###################################################################################
        H=MAP_QDA(filee,x_lower,x_upper,y_lower,y_upper,Nx,Ny)
        print 'Confusion Matrix for QDA= '+str(H[4])
        print 'Misclassification rate for QDA= '+str(H[5])
        POS=H[0]
        POSx=H[2]
        POSy=H[3]
        Class_map=H[1]
        Plot_Data(POSx,POSy,Class_map,'+')
        Plot_Data(x,y,z,'o')
        plt.title('QDA classification, misclassification rate= '+str(H[5]))
        plt.xlabel('Feature x1')
        plt.ylabel('Feature x2')
        plt.savefig('QDA_Classification.png')
        plt.close()
        ###################################################################################
        H=MAP_Linear_Reg(filee,x_lower,x_upper,y_lower,y_upper,Nx,Ny)
        print 'Confusion Matrix for Linear Regresion Clasification= '+str(H[4])
        print 'Misclassification rate for Linear Regresion Classification= '+str(H[5])
        POS=H[0]
        POSx=H[2]
        POSy=H[3]
        Class_map=H[1]
        Plot_Data(POSx,POSy,Class_map,'+')
        Plot_Data(x,y,z,'o')
        plt.title('Linear Regresion classification, misclassification rate= '+str(H[5]))
        plt.xlabel('Feature x1')
        plt.ylabel('Feature x2')
        plt.savefig('Linear_Reg_Classification.png')
        plt.close()
        ###################################################################################
        H=MAP_GaussianNB(filee,x_lower,x_upper,y_lower,y_upper,Nx,Ny)
        print 'Confusion Matrix for GaussianNB Classification= '+str(H[4])
        print 'Misclassification rate for GaussianNB Classification= '+str(H[5])
        POS=H[0]
        POSx=H[2]
        POSy=H[3]
        Class_map=H[1]
        Plot_Data(POSx,POSy,Class_map,'+')
        Plot_Data(x,y,z,'o')
        plt.title('GaussianNB classification, misclassification rate= '+str(H[5]))
        plt.xlabel('Feature x1')
        plt.ylabel('Feature x2')
        plt.savefig('GaussianNB_Classification.png')
        plt.close()
        ###################################################################################
        n_neigh=n_neigh1
        H=MAP_Neight(filee,n_neigh1,x_lower,x_upper,y_lower,y_upper,Nx,Ny)
        print 'Confusion Matrix for KNeighbors Classification= '+str(H[4])
        print 'Misclassification rate for KNeighbors Classification= '+str(H[5])
        print 'N_Neigh= '+str(n_neigh1)
        POS=H[0]
        POSx=H[2]
        POSy=H[3]
        Class_map=H[1]
        Plot_Data(POSx,POSy,Class_map,'+')
        Plot_Data(x,y,z,'o')
        plt.title('K_neighbors classification,N_Neigh= '+str(n_neigh1)+', misclassification rate= '+str(H[5]))
        plt.xlabel('Feature x1')
        plt.ylabel('Feature x2')
        plt.savefig('KNeigh_'+str(n_neigh1)+'_Classification.png')
        plt.close()
        ###################################################################################
        n_neigh=n_neigh2
        H=MAP_Neight(filee,n_neigh,x_lower,x_upper,y_lower,y_upper,Nx,Ny)
        print 'Confusion Matrix for KNeighbors Classification= '+str(H[4])
        print 'Misclassification rate for KNeighbors Classification= '+str(H[5])
        print 'N_Neigh= '+str(n_neigh)
        POS=H[0]
        POSx=H[2]
        POSy=H[3]
        Class_map=H[1]
        Plot_Data(POSx,POSy,Class_map,'+')
        Plot_Data(x,y,z,'o')
        plt.title('K_neighbors classification,N_Neigh= '+str(n_neigh)+', misclassification rate= '+str(H[5]))
        plt.xlabel('Feature x1')
        plt.ylabel('Feature x2')
        plt.savefig('KNeigh_'+str(n_neigh)+'_Classification.png')
        plt.close()
        ###################################################################################
        n_neigh=n_neigh3
        H=MAP_Neight(filee,n_neigh,x_lower,x_upper,y_lower,y_upper,Nx,Ny)
        print 'Confusion Matrix for KNeighbors Classification= '+str(H[4])
        print 'Misclassification rate for KNeighbors Clasification= '+str(H[5])
        print 'N_Neigh= '+str(n_neigh)
        POS=H[0]
        POSx=H[2]
        POSy=H[3]
        Class_map=H[1]
        Plot_Data(POSx,POSy,Class_map,'+')
        Plot_Data(x,y,z,'o')
        plt.title('K_neighbors classification,N_Neigh= '+str(n_neigh)+', misclassification rate= '+str(H[5]))
        plt.xlabel('Feature x1')
        plt.ylabel('Feature x2')
        plt.savefig('KNeigh_'+str(n_neigh)+'_Classification.png')
        plt.close()
        ###################################################################################
        H=Bayesian_Clasification(filee,MuX_1,MuY_1,Sig11_1,Sig12_1,Sig21_1,Sig22_1,MuX_2,MuY_2,Sig11_2,Sig12_2,Sig21_2,Sig22_2,x_lower,x_upper,y_lower,y_upper,Nx,Ny,)
        print 'Confusion Matrix for Bayesian Classificator= '+str(H[4])
        print 'Misclassification rate for Bayesian Classification= '+str(H[5])
        POS=H[0]
        POSx=H[2]
        POSy=H[3]
        Class_map=H[1]
        Plot_Data(POSx,POSy,Class_map,'+')
        Plot_Data(x,y,z,'o')
        plt.title('Bayesian classification, misclassification rate= '+str(H[5]))
        plt.xlabel('Feature x1')
        plt.ylabel('Feature x2')
        plt.savefig('Bayesian_Classification.png')
        plt.close()
           


filee='datos_clasificacion.dat'
x_lower=-4
x_upper=12
y_lower=-4
y_upper=10
Nx=50
Ny=50
n_neigth=30

Sig11_1=5
Sig12_1=-2
Sig21_1=-2
Sig22_1=5
MuX_1=2
MuY_1=3

Sig11_2=1
Sig12_2=0
Sig21_2=0
Sig22_2=1
MuX_2=6
MuY_2=6


All_task()
