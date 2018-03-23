import configparser
import os


def rekey(x):
    subject_id = str(x[0].split("/")[4])
    file_id = str(x[0].split("/")[7].split(".")[0])
    return (subject_id, file_id), x[1]


def filterbvs(x):
    if x[0][1] == 'bvals' or x[0][1] == 'bvecs':
        return True
    else:
        return False

def rekeygtab(x):
    return x[0][0], (x[0][1], x[1])

# build gtab
def buildgtab(x):
    import dipy.core.gradients as dpg
    if x[1][0][0] == 'bvals':
        bvals = x[1][0][1]
        bvecs = x[1][1][1]
    else:
        bvals = x[1][1][1]
        bvecs = x[1][0][1]
    bvals = np.fromstring(bvals, dtype=int, sep=" ")
    bvecs = np.fromstring(bvecs, dtype=float, sep=" ").reshape(3,len(bvals))
    return (x[0], dpg.gradient_table(bvals, bvecs, b0_threshold=10) )

#filter imgRDD to only contain images
def filterforimage(x):
    if x[0][1] == 'bvals' or x[0][1] == 'bvecs':
        return False
    else:
        return True

def unpick(x):
    import cPickle  # Note: for python 3: import _pickle as cPickle
    return cPickle.loads(x)

#filter for mask
def filterformask(x):
    if(int(x[0][1]) in maskids[0]):
        return True
    else:
        return False

def meanval(x):
    s = sum(a[1] for a in list(x[1]))
    c = len(x[1])
    return (x[0],s/c)

#create mask
def buildmask(x, gtab):
    subjectid = x[0]
    from dipy.segment.mask import median_otsu
    _, mask = median_otsu(x[1], 4, 2, False, vol_idx=np.where(gtab.value[str(subjectid)].b0s_mask), dilate=1)
    return (subjectid,mask)



def denoise(x, mask):
    subjectid = x[0][0]
    from dipy.denoise import nlmeans
    from dipy.denoise.noise_estimate import estimate_sigma
    sigma = estimate_sigma(x[1])
    #return(x[0], nlmeans.nlmeans(x[1], num_threads=1, sigma=sigma, mask=mask.value[str(subjectid)]))
    return(x[0], nlmeans.nlmeans(x[1], sigma=sigma, mask=mask.value[str(subjectid)]))


# repartition data for tensor model
def repart(x, mask):
    import itertools
    [xp,yp,zp]=[4,4,4]
    subjid = x[0][0]
    imgid = x[0][1]
    img = x[1]
    mask = mask.value[str(subjid)]
    [xSize,ySize,zSize] = [img.shape[0]/xp,img.shape[1]/yp, img.shape[2]/zp]
    i = 0
    datalist = []
    for x,y,z in itertools.product(range(xp), range(yp), range(zp)):
        [xS, yS, zS] = [x*xSize, y*ySize, z*zSize]
        [xE, yE, zE] = [img.shape[0] if x == xp - 1 else (x+1)*xSize, \
                        img.shape[1] if y == yp - 1 else (y+1)*ySize, \
                        img.shape[2] if z == zp - 1 else (z+1)*zSize]
        datalist.append(((subjid, i, imgid),(img[xS:xE, yS:yE, zS:zE],mask[xS:xE, yS:yE, zS:zE])))
        i=i+1
    return datalist


## regroup after re-part
def regroup(x):
    import numpy as np
    #print x[0] #('subjid','imgid')
    state = None
    for a in list(x[1]):
        imgid = a[0][2]
        print(imgid)
        img = np.asarray(a[1][0])
        if state is None:
            shape = img.shape + (288,)
            state = np.empty(shape)
            state[:,:,:,imgid]=img
        else:
            state[:,:,:,imgid]=img
    mask = a[1][1]
    return (x[0],(np.asarray(state),mask))


def fit_model(dt, gtab):
    import dipy.reconst.dti as dti
    subjid = dt[0][0]
    ten_model = dti.TensorModel(gtab.value[str(subjid)],min_signal=1)
    ten_fit = ten_model.fit(dt[1][0], mask=dt[1][1])
    return (dt[0],ten_fit)

import numpy as np
import sys

if len(sys.argv) == 1:
    print("Please provide subject ids as parameters")
    sys.exit(0)

from pyspark import SparkContext, SparkConf
conf = SparkConf()\
    .setAppName('neuroscience')\
    .setMaster('spark://127.0.0.1:7077')\
    .setMaster('spark://127.0.0.1:7077')
sc = SparkContext(conf=conf)

config_parser = configparser.ConfigParser()
config_parser.read_file(open(os.path.join(os.path.expanduser('~'), '.aws', 'credentials')))
config_parser.sections()

sc._jsc.hadoopConfiguration().set("fs.s3a.access.key", config_parser.get('hcp', 'AWS_ACCESS_KEY_ID'))
sc._jsc.hadoopConfiguration().set("fs.s3a.secret.key", config_parser.get('hcp', 'AWS_SECRET_ACCESS_KEY'))

s3Paths = ['s3a://hcp-openaccess/HCP/{0}/T1w/Diffusion/{1}'.format(subject_id, file_name) for subject_id in sys.argv[1:] for file_name in ['bvals', 'bvecs', 'data.nii.gz']]
for path in s3Paths:
    print("file in path: {0}".format(path))

imgRDDs = [sc.binaryFiles(s3path, minPartitions=290) for s3path in s3Paths]
imgRDD = sc.union(imgRDDs).map(rekey).cache()
imgRDD.count()

#get bvals and bvec files
gtabRDD = imgRDD.filter(filterbvs)
gtabRDD = gtabRDD.map(rekeygtab)
gtabRDD = gtabRDD.map(lambda nameTuple: (nameTuple[0], [ nameTuple[1] ])).reduceByKey(lambda a, b: a + b).cache()
print(gtabRDD.take(3))
gtabRDD = gtabRDD.map(buildgtab)

gtb = gtabRDD.first()
maskids = np.where(gtb[1].b0s_mask)
broadcastVar = sc.broadcast(maskids)

# broadcast gtab:
bcastGtab = sc.broadcast(gtabRDD.collectAsMap())

imgOnlyRDD = imgRDD.filter(filterforimage).mapValues(unpick).cache()
maskrawRDD = imgOnlyRDD.filter(filterformask).cache()
maskRDD = maskrawRDD.groupBy(lambda x: (x[0][0]))
maskRDD1 = maskRDD.map(meanval)
maskRDD2 = maskRDD1.map(lambda x: buildmask(x,bcastGtab))
bcastMask = sc.broadcast(maskRDD2.collectAsMap())


denoisedRDD = imgOnlyRDD.map(lambda x:denoise(x,bcastMask))
denoisedRDD = denoisedRDD.flatMap(lambda x: repart(x, bcastMask))


# reduce by key
denoisedRDD = denoisedRDD.groupBy(lambda x: (x[0][0],x[0][1]))
tmRDD = denoisedRDD.map(regroup)

tmmRDD = tmRDD.map(lambda x: fit_model(x,bcastGtab )).cache()
tmmRDD.count()
