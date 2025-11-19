import openfgl.config as config


from openfgl.flcore.trainer import FGLTrainer

args = config.args

args.root = "./dataset"

args.dataset = ["Cora"]
args.simulation_mode = "subgraph_fl_louvain"
args.num_clients = 10


if True:
    args.fl_algorithm = "hcfl_plus"
    args.model = ["gcn"]
else:
    args.fl_algorithm = "fedproto"
    args.model = ["gcn", "gat", "sgc", "mlp", "graphsage"] # choose multiple gnn models for model heterogeneity setting.

args.metrics = ["accuracy"]



trainer = FGLTrainer(args)

trainer.train()