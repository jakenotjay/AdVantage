
class PipelineObject:    
    data = None
    def __init__(self) -> None:
        self.data = {}

    def put(self, key, value):
        self.data[key] = value

    def has(self, key):
        keys = self.data.keys()
        return key in keys

    def get(self, key):
        return self.data[key]        

    def all(self):
        return self.data     

class PipelineHandler:
    def handle(self, task: PipelineObject, next):
        return next(task)


class Pipeline:
    handlers = []
    taskObject = None

    def __init__(self, tasks = []) -> None:
        self.taskObject = None
        for task in tasks:
           self.append(task)

    def reverseHandlers(self):
        self.handlers.reverse()
        return self

    def append(self, task: PipelineHandler):
        self.handlers.append(task)
        return self

    def send(self, taskObject: PipelineObject):
        self.taskObject = taskObject

        def afterTask(taskObject):
            return taskObject

        return self.then(afterTask)

    def then(self, afterTask):
        def carry(stack, handler: PipelineHandler):
            def next(taskObject: PipelineObject):
                if taskObject.continue_frames == False:
                    return taskObject
                return handler.handle(taskObject, stack)
            return next    

        stack = afterTask
        for handler in self.handlers:
            stack = carry(stack, handler)

        return stack(self.taskObject) 
            

        

