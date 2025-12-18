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
    args.hid_dim = 128
    args.num_epochs = 5
    args.lr = 0.02
    args.hcfl_mu = 0.01
    args.hcfl_global_lr = 0.3
    args.hcfl_split_tol = 0.1
    args.hcfl_max_clusters = 16
    args.num_clusters = 1
else:
    args.fl_algorithm = "fedproto"
    args.model = ["gcn", "gat", "sgc", "mlp", "graphsage"] # choose multiple gnn models for model heterogeneity setting.

args.metrics = ["accuracy"]



trainer = FGLTrainer(args)

trainer.train()
