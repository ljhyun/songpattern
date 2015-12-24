import geopy
import numpy
import random
import operator
import sys

def parseData(filein, fileout):
    geolocator = Nominatim()
    locfile = open(filein,'rb')
    wrtfile = open(fileout,'w')
    data = [line.strip().split(',') for line in locfile]
    for row in range(0,len(data)):
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

def isposdef(num):
    return numpy.all(numpy.linalg.eigvals(num)>0)

def findInitValue(feature, centroid):
    partition = [[] for l in range(0,len(centroid))]
    covk =  [[[]] for l in range(0,len(centroid))]
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
   
    #TODO: catch division by zero exception
    wgtk = [len(k)/len(feature) for k in partition]
    for c in range(0,len(partition)):
        meank[c] = numpy.add(meank[c],[feature[item] for item in partition[c]])
    #TODO: Check covariance matrix for positive semidefinite
    meank = [map(operator.div,mean,[len(mean)]*len(mean)) for mean in meank]
    for cluster in range(0,len(centroid)):
        diff = [0 for l in range(0,len(partition[cluster]))]
        for row in partition:
            print feature[row[cluster]],meank[cluster]
            diff = map(operator.sub,feature[row[cluster]],meank[cluster])
            covk[cluster] = numpy.add(covk[cluster],numpy.multiply(diff,numpy.matrix.transpose(numpy.matrix(diff))))
        covk[cluster] = map(operator.div,covk[cluster],len(partition[cluster]))
        if not isposdef(covk[cluster]):
            covk[cluster] = numpy.identity(len(feature[0]))
            
    return {'mean':meank, 'covar':covk, 'weight':wgtk}

if __name__ == "__main__":
    newdata = './Geographical Original of Music-2/translate_default_features_1059_tracks.txt';
    #filein = './Geographical Original of Music-2/default_features_1059_tracks.txt'
    #parseData(filein,newdata)
    newfile = open(newdata,'rb')
    data = [line.strip('[] ').split(',') for line in newfile]
    
    feat = []
    classe = []
    c_list = {}
    #Split data into features and classes (countries)
    for i in range(0,len(data)):
        feat.append(data[i][0:len(data[0])-2])
        country = data[i][len(data[0])-2]
        classe.append(country)
        #Find all unique class (countries)
        if country not in c_list.keys():
            c_list[country] = i
    feat = [[float(ele.replace('\'','')) for ele in row] for row in feat]
    #Initalize using kmeans
    print findInitValue(feat,c_list.values())
    

    
