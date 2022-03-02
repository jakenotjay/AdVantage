
import os.path
import time
from utils import *
from response import *
from constants import *
from api import RawVantageApi

class VantageServiceApi(RawVantageApi):

    def stabliseVideo(self, url, savePath, label='downloadJob', waitTime=10, chunkSize=1e+7, verbose = False):
        inputs = {
            'VideoFile': [url]
        }
        
        jobConfig = self.postJobConfig(SERVICE_STABLISE,inputs, label)
        job = self.postJobConfigLaunch(jobConfig.getId())
        
        if verbose:
            print('current job status: '+ job.getStatus())

        while(job.getStatus() != JOB_STATUS_COMPLETED):
            time.sleep(waitTime)
            job = self.getJob(job.getId())
            if verbose:
                print('current job status: '+ job.getStatus())


        outputFiles = self.getJobOutputFiles(1667)  
        files = outputFiles.getPlatformFiles()
        for file in files:
            fullSavePath = os.path.join(savePath,os.path.basename(file['filename']))
            if verbose:
                print('downloading '+file['filename']+' to '+fullSavePath)
            self.getPlatformFile(file['id']).saveToPath(fullSavePath, chunkSize=chunkSize)
            if verbose:
                print('downloaded '+fullSavePath)
        
        return outputFiles

    def downloadFiles(self, url, savePath, label='downloadJob', waitTime=10, chunkSize=1e+7, verbose = False):
        inputs = {
            'VideoFile': [url]
        }
        
        jobConfig = self.postJobConfig(SERVICE_DOWNLOAD,inputs, label)
        job = self.postJobConfigLaunch(jobConfig.getId())
        
        if verbose:
            print('current job status: '+ job.getStatus())

        while(job.getStatus() != JOB_STATUS_COMPLETED):
            time.sleep(waitTime)
            job = self.getJob(job.getId())
            if verbose:
                print('current job status: '+ job.getStatus())


        outputFiles = self.getJobOutputFiles(1667)  
        files = outputFiles.getPlatformFiles()
        for file in files:
            fullSavePath = os.path.join(savePath,os.path.basename(file['filename']))
            if verbose:
                print('downloading '+file['filename']+' to '+fullSavePath)
            self.getPlatformFile(file['id']).saveToPath(fullSavePath, chunkSize=chunkSize)
            if verbose:
                print('downloaded '+fullSavePath)
        
        return outputFiles
