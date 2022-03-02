import requests
import json
from .utils import *
from .response import *
from .constants import *

class RawVantageApi:
    dateFormat = '%Y-%m-%dT%H:%I:%S.%fZ'
    username = ''
    password = ''
    
    def __init__(self, username, password):
        self.username = username
        self.password = password


    def getServices(self):
        response = requests.get(ENDPOINT + '/services', auth=(self.username,self.password))
        return ServicesResponse(response)

    def getService(self, id):
        response = requests.get(ENDPOINT + '/services/'+str(id), auth=(self.username,self.password))
        return Response(response)    

    def getSearch(self, geometry: GeoRequest,page=0,perPage=20):
        queryParams = {
            'page': page,
            'resultsPerPage': perPage,
            'catalogue': CATALOGUE,
            'remoteDataCollection_commercialData':DATA_COLLECTION,
            'geometry': geometry
        }
        if geometry.hasStartDate():
            queryParams['productStartDate'] = geometry.getStartDate().strftime(self.dateFormat)
        if geometry.hasEndDate():
            queryParams['productStartDate'] = geometry.getEndDate().strftime(self.dateFormat)   
             
        response = requests.get(ENDPOINT + '/search', auth=(self.username,self.password), params=(queryParams))
        return SearchResponse(response)

    def getEstimateCost(self, jobType, id):
        response = requests.get(ENDPOINT + '/estimateCost/'+ jobType+'/'+str(id), auth=(self.username,self.password))
        return EstimateCostResponse(response)

    def getJob(self, id):
        response = requests.get(ENDPOINT + '/jobs/'+str(id), auth=(self.username,self.password))
        return JobResponse(response)   

    def getJobLogs(self, id):
        response = requests.get(ENDPOINT + '/jobs/'+str(id)+'/logs', auth=(self.username,self.password))
        return Response(response)

    def getJobOutputFiles(self, id):
        response = requests.get(ENDPOINT + '/jobs/'+str(id)+'/outputFiles', auth=(self.username,self.password))
        return JobOutputFilesResponse(response)            

    def postJobConfigLaunch(self, id):
        response = requests.post(ENDPOINT + '/jobConfigs/'+str(id)+'/launch', auth=(self.username,self.password))
        return JobCreatedResponse(response)

    def postJobConfig(self, service, inputs, label, parent = None):
        data = {
            'service': service,
            'inputs': inputs,
            'label': label,
            'parent': parent
        }
        raw_data = json.dumps(data)
        response = requests.post(ENDPOINT + '/jobConfigs', auth=(self.username,self.password), data=raw_data,headers=POST_HEADERS)
        return JobConfigResponse(response)

    def getPlatformFile(self, id):
        response = requests.get(ENDPOINT + '/platformFiles/' + str(id) + '/dl', auth=(self.username,self.password), stream=True)  
        return PlaformFileResponse(response, isStream=True)  
