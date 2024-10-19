
# args
from argparse import ArgumentParser
import os.path as osp
def get_args(args=None):
    parser = ArgumentParser()
    # execute
    parser.add_argument("-t", type=str, choices="gen run check clear eval show".split(), default="eval")
    # task
    parser.add_argument("--task", type=str, default='sort_edge', help="task name")
    parser.add_argument("--log_dir", type=str, default=f"G:/multi-agents-4-graph-analysis/total_benchmark/LLM4DyG/logs_n=5/{osp.split(osp.dirname(__file__))[-1]}", help="log directory to put generated data and logs")
    
    # data
    parser.add_argument("--num_seed", type=int, default=40, help="number of problem instances")
    parser.add_argument("--k", type=int, default=1, help="number of examples")
    parser.add_argument("--T", type=int, nargs='+', default=[5], help="number of time steps")
    parser.add_argument("--N", type=int, nargs='+', default=[20], help="number of nodes")
    parser.add_argument("--p", type=float, nargs='+', default=[0.3], help="probability of edges")
    
    # model
    parser.add_argument("--model", type=str, default='gpt-4o-mini', help="model name")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    
    
    # prompt
    parser.add_argument("--add_cot", type=int, default=False, help="1 add chain of thoughts, 0 not add")
    parser.add_argument("--add_role", type=int, default=False, help="1 add role instruction prompts, 0 not add")
    parser.add_argument("--dyg_type", type=int, default=1, help = "different prompt types for dynamic graphs, see utils/prompt.py")
    parser.add_argument("--edge_type", type=int, default=0, help= "different prompt types for edges, see utils/task/base.py")
    parser.add_argument("--imp", type=int, default=0, help= "additional prompt types for improving, see utils/task/prompt.py")
    parser.add_argument("--short", type=int, default=0, help= "additional prompt types for short answers, see utils/task/prompt.py")
    
    # misc
    parser.add_argument("--start_token", type=int, default=900000, help="start token budget to control token consumption")
    
    args = parser.parse_args()
    args.num_examplars = args.k
    return args