class AgentNode:

    def __init__(self, parent=None, state=None, idx=0,  U=0, N=0):
        self.children = dict()
        self.state = state
        self.parent = parent
        self.U = 0
        self.N = 0
        self.idx = idx
        self.is_fully_expanded = False
        self.is_terminal = False
        #print(f"Worker {idx}'s turn")

    def __str__(self):
        string = ""
        for i in range(len(self.state)):
            string += str(self.state[i]) + "\n"
        return string
    
    def __repr__(self) -> str:
        string = ""
        for i in range(len(self.state)):
            string += str(self.state[i]) + "\n"
        return string