
class Context(object):
    def __init__(self):
        self.records = []
        self.INPUT_SIZE=448
        self.X=[]
        self.y=[]
        self.Labels = ['VOCmotorbikes', 'VOCbicycles', 'VOCpeople', 'VOCcars']
