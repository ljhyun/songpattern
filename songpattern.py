from geopy.geocoders import Nominatim
from sklearn.decomposition import PCA
import numpy as np
import pandas
import random
import operator
import sys, getopt
import math
import warnings
import argparse

class GMMCEM:
    '''
    Input: feat: feature matrix 
           label: numerical encoded label (from 0 to k)
    '''
    def __init__(self,feat,label):
        self.feat = feat
        self.label = label

    '''
    Test for positive semidefinite
    Input: num: matrix to test
    Output: True if posdef, False if not posdef
    '''
    def __isposdef(self,num):
        return np.all(np.linalg.eigvals(num)>0)

    '''
    Simple test to check if the number of infinite or NaN, returns default if invalid
    Input: num: number to test
           default: default return value upon failed test
    Output: Num if valid, default if not valid
    '''
    def __valid(self,num, default=0.0):
        return num if not np.isnan(num) and not np.isinf(num) else default
    
    '''
    Creates test data for cross validation by filling all label with -1
    Input: minidx: Inclusive index in label of starting index
           maxindx: Exclusive index in label of ending index
    Output: Resulting test label list
    '''
    def __createTest(self,minidx,maxidx):
        return self.label[0:minidx]+[-1 for _ in range(maxidx-minidx+1)]+self.label[maxidx+1:]

    '''
    Calculates the normal pdf 
    Input: x: row vector
           mean: corresponding mean vector
           cov: corresponding covariance matrix
    Output: Resulting singular pdf value
    '''
    def __normal_pdf(self,x,mean,cov):
        #Optional e to add to matrix before inversing to ensure matrix is invertible
        e = np.add(cov,np.eye(len(x))*10**-6)
        xmu = np.subtract(x,mean);
        inv = np.linalg.inv(cov)
        detr = np.linalg.det(cov) 
        exp = np.dot(np.dot(xmu, inv), xmu)
        divisor =  np.sqrt(detr)*(2*np.pi**(len(x)/2))
        return np.exp(-0.5*exp)

    '''
    M-step of the CEM algorithm
    '''
    def __findGMValue(self):
        k = len(set(self.label))  
        w = [float(0) for _ in range(k)]
        m = [[float(0)]*len(self.feat[0]) for _ in range(k)]
        c =  [[[float(0)]*len(self.feat[0]) for _ in range(len(self.feat[0]))] for _ in range(k)]

        #sphere = [0 for _ in range(k)]
        #Weight Calculation
        for i1 in range(len(self.feat)):
            idx = self.label[i1]
            if idx >= 0:
                w[idx] += 1
                m[idx] = np.add(m[idx],self.feat[i1])
        #Mean Calculation
        for wgt in range(k):
            if w[wgt] == 0:
                w[wgt] = 1e-20
            if sum(m[wgt]) == 0:
                m[wgt] = np.array([1e-20]*len(self.feat[0]))
        for cls in range(k):
            m[cls] = np.divide(m[cls],float(w[cls]))

        #Covariance Calculation
        for i2 in range(len(self.feat)):
            idx = self.label[i2]
            if idx >= 0:
                xmu = np.asmatrix(np.subtract(self.feat[i2],m[idx]))
                #sphere[idx]+=xmu.sum()**2
                c[idx] = np.add(c[idx],np.multiply(xmu.T,xmu))
        for cls1 in range(k):
            if self.__isposdef(c[cls1]):
                c[cls1] = np.divide(c[cls1],float(w[cls1]))
            #If not positive semidefinite, create a spherical matrix or an identity to force positive semidefinite
            else:
                #c[cls1] = 1/(len(feat[0])*w[cls1])*np.sqrt(sphere[cls1])*np.identity(len(feat[0]))
                #if not isposdef(c[cls1]):
                c[cls1] = np.identity(len(self.feat[0]))
         
        w = np.divide(w,float(len(self.feat)))
        return {'mean':m, 'cov':c, 'weight':w}
    
    '''
    Executes the EM algorithm and outputs the final parameters
    Input: minidx: Inclusive index in label of starting index
           maxindx: Exclusive index in label of ending index
    Output: Predicted class label
    '''
    def trainGMM(self,minidx,maxidx):
        #PCA reduction to 90 rank
        self.feat = np.matrix(self.feat)
        pca = PCA(n_components=90,whiten=True)
        self.feat = pca.fit_transform(self.feat)
        label = self.__createTest(minidx,maxidx)
        init = self.__findGMValue()
        mk = init['mean']
        c = init['cov']
        w = init['weight']
        eps = 1e-16
        n = len(self.feat)
        m = len(self.feat[0])
        p1 = -1
        while(True): 
            gm = np.zeros((maxidx-minidx+1,len(w)),np.float)
            sumgm = 0
            for i in range(minidx,maxidx+1):
                for k in range(len(w)):
                    #Compute E-step
                    gm[i-minidx][k] =  w[k]*self.__normal_pdf(self.feat[i],mk[k],c[k])
                    sumgm += gm[i-minidx][k]
                label[i] = np.argmax(gm[i-minidx])
            #Break if converged
            print abs(p1-sumgm)
            if (abs(p1-sumgm) < eps):
                break
            p1 = sumgm
            
            final = self.__findGMValue()
            mk = final['mean']
            c = final['cov']
            w = final['weight']
             
        return label
    
    '''
    Execute the cross validation 
    Input: k: k-fold cross validation
    Output: Accuracy of the cross validation
    '''
    def crossValidation(self,k):
        step = len(self.label)/k
        acc = 0
        for i in range(k):
            r = step*i
            s = step*(i+1)
            predicted = self.trainGMM(r,s)
            acc += sum(pre==post for pre,post in zip(self.label[r:s],predicted[r:s]))/float(s-r)
        return acc/k

'''
Converts MARSYAS[1] lat,long class vector to specific country class
Input: filein: MARSYAS encoded file name
fileout: output file name 
'''
def parseData(filein, fileout):
    geolocator = Nominatim()
    locfile = open(filein,'rb')
    wrtfile = open(fileout,'w')
    data = [line.strip().split(',') for line in locfile]
    for row in range(0,len(data)):
    #Try to use the geolocator service with timeout of 5 seconds, if the country does not exist/timeout fill the country as 'nb'
        try:
            ct_code = geolocator.reverse(data[row][len(data[row])-2]+","+data[row][len(data[row])-1],timeout=5).raw
            if (ct_code != None and ct_code.get('address') != None):
                data[row][len(data[row])-2] = ct_code.get('address').get('country_code')
            else:
                data[row][len(data[row])-2] = 'nb'
        except geopy.exc.GeocoderTimedOut:
            data[row][len(data[row])-2] = 'nb'
            print 'Row:',row,'Timeout'

    for item in data:
        print>>wrtfile, item

'''
Encodes countries into numerical label according to the list
Input: data: data to be encoded 
       cmap: lookup map for the countries
Output: Encoded data matrix
'''
def encodeData(data, cmap):
    lookup = cmap
    ret = [-1 for _ in range(len(data))]
    for n in range(len(data)):
        ret[n] = lookup.index(data[n])
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parse',dest='par',type=bool,default=False,help='Set true to enable data parsing')
    parser.add_argument('--infile',dest='filein',type=str,default='',help='Input file to parse')
    parser.add_argument('--outfile',dest='newdata',type=str,default='',help='Output file for parsing/file to read for processing',required=True)
    parser.add_argument('--xss',dest='cross',type=int,default=10,help='Cross validation parameter k')
    results = parser.parse_args()

    if results.par:
        if (results.infile==''):
            print "usage --infile filename"
            return
        parseData(results.filein,results.newdata)

    newfile = open(results.newdata,'rb')
    data = [line.strip('[] ').split(',') for line in newfile]
    feat = []
    classe = []
    c_list = {}
    label = [-1 for _ in range(len(data))]
    
    #Split data into features and classes (countries)
    for i in range(0,len(data)):
        feat.append(data[i][0:len(data[0])-2])
        country = data[i][len(data[0])-2]
        label[i] = country
        classe.append(country)
        #Find all unique class (countries)
        if country not in c_list.keys():
            c_list[country] = i
    label = encodeData(label,c_list.keys())
    feat = [[float(ele.replace('\'','')) for ele in row] for row in feat]
   
    model = GMMCEM(feat,label)
    print model.crossValidation(results.cross)    
          
if __name__ == "__main__":
    main()
