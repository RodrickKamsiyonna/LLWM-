"""Plan with predictor-only receding-horizon latent actions."""
import argparse
import torch
from nanochat.common import compute_init, compute_cleanup, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.llwm import LLWM
from nanochat.planning import mpc_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embedding-mode', choices=['soft', 'straight_through'], default='soft',
                        help='embedding feedback during optimized rollout; committed MPC tokens stay discrete')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--goal', required=True, help='desired suffix of the predicted rollout')
    parser.add_argument('--source', choices=['base', 'sft', 'rl'], default='base')
    parser.add_argument('--model-tag', default=None)
    parser.add_argument('--step', type=int, default=None)
    parser.add_argument('--horizon', type=int, default=50)
    parser.add_argument('--execute-steps', type=int, default=5,
                        help='number of generated tokens to commit before replanning')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device-type', default='', choices=['', 'cpu', 'cuda', 'mps', 'xla'])
    args = parser.parse_args()
    _, _, _, world, device = compute_init(args.device_type or autodetect_device_type())
    if world != 1:
        parser.error('planning uses one process; launch XLA with --single-process')
    model, tokenizer, _ = load_model(args.source, device, phase='eval', model_tag=args.model_tag, step=args.step)
    if not isinstance(model, LLWM):
        parser.error('planning requires an LLWM checkpoint')
    model.planning_embedding_mode = args.embedding_mode
    print(f"Planning embedding mode: {args.embedding_mode}", flush=True)
    prompt = tokenizer.encode(args.prompt, prepend=tokenizer.get_bos_token_id())
    goal = tokenizer.encode(args.goal)
    result = mpc_plan(model, torch.tensor([prompt], dtype=torch.long, device=device),
                      torch.tensor(goal, dtype=torch.long, device=device),
                      horizon=args.horizon, execute_steps=args.execute_steps,
                      iterations=args.iterations, lr=args.lr, seed=args.seed)
    print(tokenizer.decode(result['tokens'][0].tolist()))
    metric = result['goal_objective']
    print(f"Best {metric}: {result['goal_loss']:.6f} | goal exact match: {result['goal_exact_match']:.0%}")
    for round_result in result['round_history']:
        print(f"MPC start={round_result['start']} horizon={round_result['horizon']} "
              f"executed={round_result['executed']} {metric}={round_result[metric]:.6f}")
    compute_cleanup()


if __name__ == '__main__':
    main()
