

class ProcessedVideo:
    frames = None
    def __init__(self) -> None:
        self.frames = []

    def append(self, frame):
        self.frames.append(frame)
        return self    

    def all(self):
        data = []
        for frame in self.frames:
            data.append(frame.all())
        return data     
