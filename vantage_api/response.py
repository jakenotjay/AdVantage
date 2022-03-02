class Response:
    data = None
    response = None
    def __init__(self, response, isStream = False) -> None:
        if response.status_code >= 200 and response.status_code < 300:
            if isStream == False:
                self.data = response.json()
            self.response = response
        else:
            raise ValueError('invalid response: ' + str(response.status_code))  

    def totalElements(self):
        return self.data['page']['totalElements']

    def totalPages(self):
        return self.data['page']['totalPages']   

    def links(self):
        return self .data['_links']    

    def json(self):
        return self.data

class SearchResponse(Response):
    
    def getFeatures(self):
        features = {}
        for feature in self.data['features']:
            features[feature['properties']['productIdentifier']] = feature['properties']['platformUrl']
        return features

class ServicesResponse(Response):
    
    def getServices(self):
        features = {}
        for feature in self.data['_embedded']['services']:
            features[feature['id']] = {
                'name':feature['name'],
                'description':feature['description']
            }
        return features

class EstimateCostResponse(Response):
    
    def getEstimatedCost(self):
        return self.data['estimatedCost']

    def getRecurrence(self):
        return self.data['recurrence']    

    def getCurrentWalletBalance(self):
        return self.data['currentWalletBalance']        


class JobConfigResponse(Response):
    
    def getId(self):
        return self.data['id']

class PlaformFileResponse(Response):
    
    def saveToPath(self, savePath, chunkSize = 1024):
        with open(savePath, "wb") as handle:
            for data in self.response.iter_content(chunkSize):
                handle.write(data)   
        return self         

class JobOutputFilesResponse(Response):
    
    def getId(self):
        return self.data['id']

    def getPlatformFiles(self):
        return self.data['_embedded']['platformFiles']    

class JobResponse(Response):
    
    def getId(self):
        return self.data['id']   
    def getStatus(self):
        return self.data['status']
    def getPhase(self):
        return self.data['phase'] 
    def getStage(self):
        return self.data['stage'] 
    def getOutputs(self):
        return self.data['outputs'] 
    def getQueuePosition(self):
        return self.data['queuePosition'] 
    def getWorkerId(self):
        return self.data['workerId']             

class JobCreatedResponse(Response):
    
    def getId(self):
        return self.data['content']['id']        
    def getStatus(self):
        return self.data['content']['status']
    def getPhase(self):
        return self.data['content']['phase'] 
    def getStage(self):
        return self.data['content']['stage'] 
    def getOutputs(self):
        return self.data['content']['outputs'] 
    def getQueuePosition(self):
        return self.data['content']['queuePosition'] 
    def getWorkerId(self):
        return self.data['content']['workerId']                    