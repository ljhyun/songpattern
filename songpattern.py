from geopy.geocoders import Nominatim
import pandas
from sklearn.decomposition import PCA
import numpy as np
import random
import operator
import sys
import math
import warnings
import pickle



#countrymap = [2,2,2,2,4,5,2,3,2,3,5,3,5,3,3,4,4,2,4,1,4,5,4,3,3,6,2,2,1,5,6,5,2]
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

def isposdef(num):
    return np.all(np.linalg.eigvals(num)>0)

def valid(num, default=0.0):
    return num if not np.isnan(num) and not np.isinf(num) else default

def normal_pdf(x,mean,cov):
    e = np.add(cov,np.eye(len(x))*10**-6)
    xmu = np.subtract(x,mean);
    inv = np.linalg.inv(cov)
    detr = np.linalg.det(cov) 
    exp = np.dot(np.dot(xmu, inv), xmu)
    divisor =  np.sqrt(detr)*(2*np.pi**(len(x)/2))
    return np.exp(-0.5*exp)
    #return np.exp(-0.5*exp)/divisor if divisor != 0 else 0
'''
Initalization using knn method from [3]
Input: feature: feature vector from MARSYAS[1] generated output 
       centroid: list containing the index of the initial centers corresponding to the feature vector
Output: Map value containing initial mean 1xC, weight 1xC, covariance CxXxX
'''
def findInitValue(feature, centroid):
    partition = [[] for l in range(0,len(centroid))]
    #Covariance 
    covk =  [[[0]*len(feature[0]) for k in range(0,len(feature[0]))] for l in range(0,len(centroid))]
    #Mean
    meank = [[0]*len(feature[0]) for l in range(0,len(centroid))]

    for f in range(0,len(feature)):
        kmin = sys.maxint
        kidx = -1
        for kth in range(0,len(centroid)):
            mdist = sum([s**2 for s in map(operator.sub,feature[f],feature[centroid[kth]])])
            if (min(mdist,kmin) is not kmin):
                kmin = mdist
                kidx = kth
        partition[kidx].append(f)
    #Weight
    wgtk = [len(k)/float(len(feature)) for k in partition]
    for c in range(0,len(partition)):
        for item in partition[c]:
            meank[c] = np.add(meank[c],feature[item])

    #TODO: Add check for spherical covariance matrix ||x(n) - u(k)||2 * I
    meank = [map(operator.div,mean,[len(mean)]*len(mean)) for mean in meank]
    for cluster in range(0,len(centroid)):
        diff = [0 for l in range(0,len(partition[cluster]))]
        for index in partition[cluster]:
            diff = map(operator.sub,feature[index],meank[cluster])
            covk[cluster] = np.add(covk[cluster],np.multiply(diff,np.matrix.transpose(np.matrix(diff))))
        #covk[cluster] = np.divide(covk[cluster],float(len(partition[cluster])))
        #if not isposdef(covk[cluster]):
        covk[cluster] = np.identity(len(feature[0]))

    return {'mean':meank, 'covar':covk, 'weight':wgtk}


def findGMValue(feat,label):
    k = len(set(label))  
    w = [float(0) for _ in range(k)]
    m = [[float(0)]*len(feat[0]) for _ in range(k)]
    c =  [[[float(0)]*len(feat[0]) for _ in range(len(feat[0]))] for _ in range(k)]
    sphere = [0 for _ in range(k)]
    for i1 in range(len(feat)):
        idx = label[i1]
        if idx >= 0:
            w[idx] += 1
            m[idx] = np.add(m[idx],feat[i1])

    for wgt in range(k):
        if w[wgt] == 0:
            w[wgt] = 1e-20
        if sum(m[wgt]) == 0:
            m[wgt] = np.array([1e-20]*len(feat[0]))
    #print m     
    for cls in range(k):
        m[cls] = np.divide(m[cls],float(w[cls]))

    for i2 in range(len(feat)):
        idx = label[i2]
        if idx >= 0:
            xmu = np.asmatrix(np.subtract(feat[i2],m[idx]))
            sphere[idx]+=xmu.sum()**2
            c[idx] = np.add(c[idx],np.multiply(xmu.T,xmu))
    #print c[0]
    for cls1 in range(k):
        if isposdef(c[cls1]):
            c[cls1] = np.divide(c[cls1],float(w[cls1]))
        else:
            #print sphere[cls1]
            #c[cls1] = 1/(len(feat[0])*w[cls1])*np.sqrt(sphere[cls1])*np.identity(len(feat[0]))
            #if not isposdef(c[cls1]):
            c[cls1] = np.identity(len(feat[0]))
     
    w = np.divide(w,float(len(feat)))
    return {'mean':m, 'cov':c, 'weight':w}
    

def createTest(label,minidx,maxidx):
    return label[0:minidx]+[-1 for _ in range(maxidx-minidx+1)]+label[maxidx+1:]

'''
Executes the EM algorithm and outputs the final parameters
Input: init: Initialization Value from findInitValue
       traindata: Training Data 
Output: Map value containing initial mean 1xC, weight 1xC, covariance CxXxX
'''
def trainGMM(data,label,minidx,maxidx):   
    data = np.subtract(data,np.mean(data,0))/np.std(data,0)
    #data = np.matrix(data)
    #pca = PCA(n_components=90,whiten=True)
    #data = pca.fit_transform(data)
    label = createTest(label,minidx,maxidx)
    #print label
    init = findGMValue(data,label)
    meank = init['mean']
    covk = init['cov']
    weight = init['weight']
    eps = 1e-16
    n = len(data)
    m = len(data[0])
    p1 = -1
    while(True): 
    #for _ in range(25):
        gm = np.zeros((maxidx-minidx+1,len(weight)),np.float)
        sumgm = 0
        for i in range(minidx,maxidx+1):
            for k in range(len(weight)):
                gm[i-minidx][k] =  weight[k]*normal_pdf(data[i],meank[k],covk[k])
                #print gm[i][k]
                sumgm += gm[i-minidx][k]
            label[i] = np.argmax(gm[i-minidx])
        #print sumgm
        print abs(p1-sumgm)
        if (abs(p1-sumgm) < eps):
            break
        p1 = sumgm

        final = findGMValue(data,label)
        meank = final['mean']
        covk = final['cov']
        weight = final['weight']

    return label

'''
Calculates the probabiliy and estimate the country
Input: final: Final Parameter Value from trainGMM
       testData: Test Data 
Output: Accuracy of the Test
'''
def testGMM(final, testdata, label):
    meank = final['mean']
    covk = final['covar']
    weight = final['weight']
    accuracy = 0
    n = len(testdata)
    m = len(testdata[0])
    e = np.eye(m)*10**-10
    #print weight
    #return 0
    for i in range(n):
        le = [float(0) for _ in range(len(weight))]
        for k in range(len(weight)):
            #1xn nxn nx1
            le[k] = weight[k]*normal_pdf(testdata[i],meank[k],covk[k])
        bestk = le.index(max(le))
        print le, bestk, label[i]
        if (bestk == label[i]):
            accuracy+=1

    return accuracy


def crossValidation(feat,label,k):
    step = len(label)/k
    acc = 0
    for i in range(k):
        r = step*i
        s = step*(i+1)
        predicted = trainGMM(feat,label,r,s)
        #acc += sum(pre==post for pre,post in zip(label[r:s],predicted[r:s]))/float(s-r)
        #print predicted
        acc += sum(pre==post for pre,post in zip(label[r:s],predicted[r:s]))/float(s-r)
    return acc/k
    
if __name__ == "__main__":
    newdata = './Geographical Original of Music-2/translate_default_plus_chromatic_features_1059_tracks.txt';
    #newdata = './Geographical Original of Music-2/translate_default_plus_chromatic_features_1059_tracks.txt;
    filein = './Geographical Original of Music-2/default_plus_chromatic_features_1059_tracks.txt'
    #parseData(filein,newdata)
    statefile = "./state/3.pik"
    newfile = open(newdata,'rb')
    data = [line.strip('[] ').split(',') for line in newfile]
    
    print '\n'.join(''.join(str(cell) for cell in row) for row in data)    
    state = open(statefile,'w')
    #state = open(statefile,'r')
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
    #print c_list, len(c_list)
    #print c_list.keys()
    label = encodeData(label,c_list.keys())
    #print label
    feat = [[float(ele.replace('\'','')) for ele in row] for row in feat]
    #Initalize using kmeans
    
    #init = findGMValue(feat,label,covk,meank,weight)
    #print crossValidation(feat,label,5)    
    '''        
    maroonstr = "6.49117198e-02   4.23617692e-02   1.64886487e-01   1.02127330e-01 -4.46011378e+01   4.33712752e+00   7.54656599e-01   1.11692618e+00 2.68403545e-01   1.08982054e-01  -3.37572369e-02  -6.48599850e-02 -6.80427279e-03  -4.24881523e-03 3.16121636e-01   1.88269549e-01 1.84049929e-01   2.03713739e-03   1.90928242e-03   1.87348115e-03 2.07733288e-03   2.26481009e-03   2.42385966e-03   2.53559261e-03 2.53591996e-03   2.44388724e-03   2.32054284e-03   2.24894499e-03 2.18444265e-03   1.53672613e+00   3.51276791e+00   3.01449594e-02 2.18114366e-02   1.77536551e-01   5.60397976e-02   1.84695369e+00 1.10806029e+0 8.12335339e-01   7.61876958e-01   6.18888946e-01 6.12616863e-01   5.83189801e-01   5.69341644e-01   5.39498392e-01 5.24563600e-01   5.09021164e-01   5.00333075e-01   4.85054624e-01 2.16046345e-03   2.01355920e-03   1.98726516e-03   2.20078187e-03 2.37392789e-03   2.51255346e-03   2.61490809e-03   2.64940530e-03 2.61842270e-03   2.54638211e-03   2.43969649e-03   2.31273045e-03 3.01036854e-01   2.12083919e+00"
    maroon = map(float,maroonstr.split())
    label = [10] + label
    timbal = list(zip(*feat)[0:62])
    feat = maroon+np.matrix(timbal).T
    print(trainGMM(feat.tolist(),label,0,1))
    '''
    #pickle.dump(final,state)
   # final = pickle.load(state)
    #print c_list.keys()
    #print(testGMM(final,feat,label))      
