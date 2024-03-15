import lyra_graphtool_test as lgtool
from lyra_graphtool_test import Configuration, Config_Single_Time, Edge, Graph, Graph_Type, Parameters, Vertex, Worker_Type, Vertex_Type


#pargs = lgtool.ProcessArgs.load(arguments_file='args_sandbox', graph_file='Test_Graph_Slightly_Off_the_Beaten_Path.json')

for i in range(100):

    pargs = lgtool.ProcessArgs.load(arguments_file='args_cs3263')

    params = lgtool.Parameters(pargs.graph,
                               budget = pargs.args_trial.budget,
                               duration_time = pargs.args_trial.duration,
                               cost_rate = pargs.worker_cost_rate
                               )

    cfg = lgtool.Configuration(params)

    print(f"Problem budget constraint: {cfg.budget}")
    print(f"Problem duration constraint: {cfg.duration_time}")
    print(f"Worker cost rates per timestep: {cfg.worker_cost_rate}")
    # cfg.graph.print_graph()
    name = "graphs/graph" + str(i + 1) + ".json"
    # cfg.graph.save_to_json(name)

print("Imported")
