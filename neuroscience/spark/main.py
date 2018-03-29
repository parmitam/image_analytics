import boto3
import configparser
import datetime
import nibabel as nib
import os
import os.path as op


def rekey(x):
    subject_id = str(x[0].split("/")[4])
    file_id = str(x[0].split("/")[7].split(".")[0])
    return (subject_id, file_id), x[1]

def download(subject_id, output_directory='.'):
    boto3.setup_default_session(profile_name='hcp')
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('hcp-openaccess')

    file_name = 'data.nii.gz'
    output_path = '{0}/{1}_{2}'.format(output_directory, subject_id, file_name)
    if op.exists(output_path):
        print("Using previously downloaded {0}".format(output_path))
    else:
        source_path = 'HCP/{0}/T1w/Diffusion/{1}'.format(subject_id, file_name)
        print("Downloading {0} to {1}".format(source_path, output_path))
        bucket.download_file(source_path, output_path)
        print("Done")

def downloadImages(x):
    (subject_id, _), s3path = x
    download(subject_id, '/tmp')
    file_name = '/tmp/{0}_data.nii.gz'.format(subject_id)
    img = nib.load(file_name)
    data = img.get_data()
    result = [((subject_id, str(i)), data[..., i]) for i in range(data.shape[-1])]
    os.remove(file_name)
    return result

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
    ten_model = dti.TensorModel(gtab.value[str(subjid)], min_signal=1)
    ten_fit = ten_model.fit(dt[1][0], mask=dt[1][1])
    return (dt[0],ten_fit)

import numpy as np
import sys

if len(sys.argv) == 1:
    print("Please provide subject ids as parameters")
    sys.exit(0)
subject_ids = sys.argv[1:]

from pyspark import SparkContext, SparkConf
conf = SparkConf()\
    .setAppName('neuroscience')\
    .setMaster('spark://127.0.0.1:7077')\
    .set('spark.executor.memory', '12g')
sc = SparkContext(conf=conf)

config_parser = configparser.ConfigParser()
config_parser.read_file(open(os.path.join(os.path.expanduser('~'), '.aws', 'credentials')))
config_parser.sections()

sc._jsc.hadoopConfiguration().set("fs.s3a.access.key", config_parser.get('hcp', 'AWS_ACCESS_KEY_ID'))
sc._jsc.hadoopConfiguration().set("fs.s3a.secret.key", config_parser.get('hcp', 'AWS_SECRET_ACCESS_KEY'))


path_format = 's3a://hcp-openaccess/HCP/{0}/T1w/Diffusion/{1}'

s3BvsPaths = [path_format.format(subject_id, file_name) for file_name in ['bvals', 'bvecs'] for subject_id in subject_ids]
bvsRDD = sc.union([sc.binaryFiles(s3path) for s3path in s3BvsPaths]).map(rekey).repartition(16).cache()

imgRDD = sc.parallelize([((subject_id, 'data.nii.gz'), 's3a://hcp-openaccess/HCP/{0}/T1w/Diffusion/{1}'.format(subject_id, 'data.nii.gz')) for subject_id in subject_ids])
print("{0} initial partitions {1}, objects: {2}".format(datetime.datetime.now(), imgRDD.getNumPartitions(), imgRDD.count()))
imgRDD = imgRDD.flatMap(downloadImages)
print("{0} partitions after flatmap {1}, objects: {1}".format(datetime.datetime.now(), imgRDD.getNumPartitions(), imgRDD.count()))
imgRDD = imgRDD.repartition(288).cache()
print("{0} downloaded images {1} to partitions: {2}".format(datetime.datetime.now(), imgRDD.count(), imgRDD.getNumPartitions()))

#get bvals and bvec files
gtabRDD = bvsRDD
gtabRDD = gtabRDD.map(rekeygtab)
gtabRDD = gtabRDD.map(lambda nameTuple: (nameTuple[0], [ nameTuple[1] ])).reduceByKey(lambda a, b: a + b).cache()
gtabRDD = gtabRDD.map(buildgtab)

gtb = gtabRDD.first()
maskids = np.where(gtb[1].b0s_mask)
broadcastVar = sc.broadcast(maskids)
print("{0} read bvs".format(datetime.datetime.now()))

# broadcast gtab:
bcastGtab = sc.broadcast(gtabRDD.collectAsMap())

maskrawRDD = imgRDD.filter(filterformask).cache()
maskRDD = maskrawRDD.groupBy(lambda x: (x[0][0]))
maskRDD1 = maskRDD.map(meanval)
maskRDD2 = maskRDD1.map(lambda x: buildmask(x, bcastGtab))
bcastMask = sc.broadcast(maskRDD2.collectAsMap())
print("{0} masked".format(datetime.datetime.now()))

denoisedRDD = imgRDD.map(lambda x: denoise(x, bcastMask))
denoisedRDD = denoisedRDD.flatMap(lambda x: repart(x, bcastMask))

print("{0} denoised".format(datetime.datetime.now()))
# reduce by key
denoisedRDD = denoisedRDD.groupBy(lambda x: (x[0][0], x[0][1]))
print("{0} grouped".format(datetime.datetime.now()))
tmRDD = denoisedRDD.map(regroup)
print("{0} regrouped".format(datetime.datetime.now()))

tmmRDD = tmRDD.map(lambda x: fit_model(x, bcastGtab)).cache()
tmmRDD.count()

print("{0} done".format(datetime.datetime.now()))
