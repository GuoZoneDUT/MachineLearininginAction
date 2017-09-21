"""
    主成分分析 机器学习实战
    
    郭宗沂，2017.9.13
"""
from numpy import *

#数据转化为矩阵
def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float,line)) for line in stringArr]
    return mat(datArr)

#将矩阵转化进新的空间
def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat,axis = 0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:,eigValInd]
    lowDDataMat = meanRemoved*redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat,reconMat
    
#将NaN替换成平均值的函数
def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat =shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i])

        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal

    return datMat

