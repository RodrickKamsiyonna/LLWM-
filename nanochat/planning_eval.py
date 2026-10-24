"""Shared goal-directed planning for CORE, ChatCORE and text samples."""
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F
from nanochat.planning import plan


@dataclass(frozen=True)
class PlanningOptions:
    goal: str = '.'
    iterations: int = 100
    lr: float = 0.05
    consistency_weight: float = 1.0
    seed: int = 123

    def __post_init__(self):
        if not self.goal or self.iterations < 1:
            raise ValueError('planning needs a nonempty goal and positive iterations')
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError('planning lr must be finite and positive')
        if not math.isfinite(self.consistency_weight) or self.consistency_weight < 0:
            raise ValueError('planning consistency weight must be finite and nonnegative')


def add_planning_args(parser, max_new_tokens=16):
    parser.add_argument('--eval-mode', choices=['auto', 'plan', 'prior'], default='auto',
                        help='auto uses planning for LLWM, prior scoring for GPT')
    parser.add_argument('--eval-max-new-tokens', type=int, default=max_new_tokens,
                        help='allowed output tokens; also the planning horizon')
    parser.add_argument('--plan-goal', default='.', help='planning goal suffix; never a benchmark answer')
    parser.add_argument('--plan-iterations', type=int, default=100)
    parser.add_argument('--plan-lr', type=float, default=0.05)
    parser.add_argument('--plan-consistency-weight', type=float, default=1.0)
    parser.add_argument('--plan-seed', type=int, default=123)


def planning_for_model(model, args=None):
    mode = getattr(args, 'eval_mode', 'auto')
    is_llwm = getattr(model.config, 'architecture', 'gpt') == 'llwm'
    if mode == 'prior' or (mode == 'auto' and not is_llwm):
        return None
    if not is_llwm:
        raise ValueError('planning evaluation requires an LLWM model')
    if args is None:
        return PlanningOptions()
    if args.eval_max_new_tokens < 1:
        raise ValueError('eval-max-new-tokens must be positive')
    return PlanningOptions(args.plan_goal, args.plan_iterations, args.plan_lr,
                           args.plan_consistency_weight, args.plan_seed)


def planned_completion(model, tokenizer, prompt, max_new_tokens, options, *, num_samples=1, seed=None):
    if max_new_tokens < 1 or num_samples < 1 or not prompt:
        raise ValueError('prompt, output budget and sample count must be nonempty/positive')
    device = model.get_device()
    prompt_ids = torch.tensor([prompt], dtype=torch.long, device=device).expand(num_samples, -1)
    goal_ids = tokenizer.encode(options.goal)
    if not goal_ids or len(goal_ids) > max_new_tokens:
        raise ValueError('planning goal token count must fit the allowed output-token budget')
    goal = torch.tensor(goal_ids, dtype=torch.long, device=device)
    return plan(model, prompt_ids, goal, horizon=max_new_tokens,
                iterations=options.iterations, lr=options.lr,
                seed=options.seed if seed is None else seed,
                consistency_weight=options.consistency_weight, return_logits=True)


def planned_continuation_scores(model, tokenizer, tokens, starts, ends, max_new_tokens, options):
    """Normal CORE candidate NLL/argmax scoring, using planned-memory logits.

    Plans see only the causal prefix and a fixed goal. For continuations longer
    than one output budget, teacher-force completed blocks and replan. Candidate
    continuations and gold labels never enter the action optimization objective.
    Shared prefixes reuse the same plan for fair multiple-choice comparisons.
    """
    scores, exact = [], []
    plans = {}
    for sequence, start, end in zip(tokens, starts, ends):
        if not 0 < start < end <= len(sequence):
            raise ValueError('CORE continuation must have a nonempty prefix and target')
        losses, matches = [], []
        for position in range(start, end, max_new_tokens):
            prefix = sequence[:position]
            key = tuple(prefix)
            if key not in plans:
                plans[key] = planned_completion(model, tokenizer, prefix, max_new_tokens, options)
            logits = plans[key]['logits'][0]
            target = torch.tensor(sequence[position:min(position+max_new_tokens, end)], dtype=torch.long)
            logits = logits[:len(target)].float()
            losses.append(F.cross_entropy(logits, target, reduction='none'))
            matches.append(logits.argmax(-1).eq(target))
        scores.append(torch.cat(losses).mean().item())
        exact.append(torch.cat(matches).all().item())
    return scores, exact


class PlanningEngine:
    """generate_batch adapter for evaluation; rolls out exactly the token budget.

    Planning decodes latent rollouts, with no Python-tool execution or token
    sampling: num_samples selects independent Gaussian action initializations.
    """
    def __init__(self, model, tokenizer, options):
        self.model, self.tokenizer, self.options = model, tokenizer, options

    def generate_batch(self, tokens, num_samples=1, max_tokens=16, temperature=0.0, top_k=None, seed=None):
        result = planned_completion(self.model, self.tokenizer, tokens, max_tokens,
                                    self.options, num_samples=num_samples, seed=seed)
        completions = [row.tolist() for row in result['tokens']]
        sequences = [list(tokens)+row for row in completions]
        masks = [[0]*len(tokens)+[1]*len(row) for row in completions]
        return sequences, masks
