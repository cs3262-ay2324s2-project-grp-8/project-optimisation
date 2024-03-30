from utils import Worker, Node

class AgentWorker(Worker):
    
    def __init__(self, type, start, rate, timestamp):
        super().__init__(type=type, start=start, rate=rate, timestamp=timestamp)
        self.model = None # for now
        raise NotImplementedError
    
    def observe(self):
        raise NotImplementedError
    
    def decay_epsilon(self):
        raise NotImplementedError
    
    def replay(self):
        raise NotImplementedError
    
    def update_target_model(self):
        raise NotImplementedError
    
    def move_back_to_origin(self, origin: Node):
        self.move(origin)