from ..pipeline import PipelineHandler
from ..app import VideoProcessingFrame

class Verbose(PipelineHandler):
    def handle(self, task: VideoProcessingFrame, next):
        print('Processing Frame: '+str(task.frame_id))

        handledTask = next(task)

        if task.has('predictions') and task.get('predictions') != None:
            for prediction in task.get('predictions'):
                print('\t Label: '+prediction.getLabel()+ ' Score: '+str(prediction.getScore())+' Box: '+str(prediction.getBox()))

        return handledTask