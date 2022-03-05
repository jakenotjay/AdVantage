
class Prediction:
    label = None
    score = None
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0

    def __init__(self, label, score, x1, y1, x2, y2):
        self.label = label
        self.score = score
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def getScore(self):
        return self.score

    def getLabel(self):
        return self.label

    def getCenter(self):
        x = self.x1 + ((self.x2 - self.x1) / 2 )
        y = self.y1 + ((self.y2 - self.y1) / 2 )
        return [int(x),int(y)]

    def getBox(self):
        return [int(self.x1), int(self.y1), int(self.x2), int(self.y2)]  

    def toMap(self):
        return {
            'label':self.getLabel(),
            'box':self.getBox(),
            'score':self.getScore()
        }    

class ProcessedVideo:
    frames = None
    def __init__(self) -> None:
        self.frames = []

    def append(self, frame):
        self.frames.append(frame)
        return self    

    def getFrames(self):
        return self.frames    

    def all(self):
        data = []
        for frame in self.frames:
            data.append(frame.all())
        return data     
