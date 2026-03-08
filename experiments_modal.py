"""
GGDPO Experiments v2 - Modal Runner
Redesigned experiments proving GGDPO's value through rigorous evaluation.

All experiments compare DPO vs GGDPO at the SAME oracle budget.

Usage:
    modal run experiments_modal.py::exp1_sample_efficiency
    modal run experiments_modal.py::exp2_scaling_n
    modal run experiments_modal.py::exp3_bt_ablation
    modal run experiments_modal.py::exp4_real_model_scaled
    modal run experiments_modal.py::exp5_ultrafeedback
    modal run experiments_modal.py::exp6_gradient_variance
    modal run experiments_modal.py::exp7_noisy_oracle
    modal run experiments_modal.py::exp8_heldout_prediction
"""

import modal

app = modal.App("ggdpo-experiments")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12"
    )
    .uv_pip_install(
        "torch",
        "transformers",
        "numpy",
        "matplotlib",
        "tqdm",
        "scipy",
        "datasets",
        "accelerate",
        "peft",
    )
)

hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("ggdpo-results", create_if_missing=True)

RESULTS_DIR = "/results"
HF_CACHE_DIR = "/root/.cache/huggingface"

COMMON_KWARGS = dict(
    image=image,
    gpu="H200:1",
    volumes={HF_CACHE_DIR: hf_cache_vol, RESULTS_DIR: results_vol},
    timeout=6 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)


# ============================================================
# Shared helper code (pure Python, serialized by Modal)
# ============================================================

def _ggdpo_helpers():
    """Returns a dict of helper functions. Call inside Modal functions."""
    import random
    import numpy as np
    import torch
    import torch.nn as nn
    from scipy.stats import kendalltau, spearmanr

    def sample_pairs_random(n_completions, n_pairs):
        """Sample random unique unordered pairs, ensuring every item appears at least once."""
        pairs = []
        existing = set()
        max_possible = n_completions * (n_completions - 1) // 2
        n_pairs = min(n_pairs, max_possible)

        # First ensure coverage: each item appears in at least one pair
        items = list(range(n_completions))
        random.shuffle(items)
        for k in range(n_completions - 1):
            pair = tuple(sorted((items[k], items[k + 1])))
            if pair not in existing:
                existing.add(pair)
                pairs.append(pair)
            if len(pairs) >= n_pairs:
                break

        # Fill remaining with random pairs
        while len(pairs) < n_pairs:
            idx = np.random.choice(n_completions, 2, replace=False)
            idx = tuple(sorted(idx))
            if idx not in existing:
                existing.add(idx)
                pairs.append(idx)
        return pairs

    def label_pairs(pairs, scores, noise_prob=0.0):
        labeled = []
        for i, j in pairs:
            if scores[i] > scores[j]:
                winner, loser = i, j
            else:
                winner, loser = j, i
            if noise_prob > 0 and np.random.random() < noise_prob:
                winner, loser = loser, winner
            labeled.append((winner, loser))
        return labeled

    def fit_bradley_terry(n_completions, labeled_pairs, n_iters=1000, lr=0.01, l1_reg=0.01):
        scores = torch.randn(n_completions, requires_grad=True)
        optimizer = torch.optim.Adam([scores], lr=lr)
        winners = torch.tensor([w for w, _ in labeled_pairs], dtype=torch.long)
        losers = torch.tensor([l for _, l in labeled_pairs], dtype=torch.long)
        for _ in range(n_iters):
            optimizer.zero_grad()
            diffs = scores[winners] - scores[losers]
            loss = -torch.sum(torch.log(torch.sigmoid(diffs) + 1e-8))
            loss += l1_reg * torch.sum(torch.abs(scores))
            loss.backward()
            optimizer.step()
        scores_np = scores.detach().cpu().numpy()
        scores_np = (scores_np - scores_np.mean()) / (scores_np.std() + 1e-8)
        return scores_np

    def fit_win_rate(n_completions, labeled_pairs):
        wins = np.zeros(n_completions)
        counts = np.zeros(n_completions)
        for w, l in labeled_pairs:
            wins[w] += 1
            counts[w] += 1
            counts[l] += 1
        scores = np.where(counts > 0, wins / counts, 0.5)
        std = scores.std()
        if std > 1e-8:
            scores = (scores - scores.mean()) / std
        return scores

    def fit_transitive_closure(n_completions, labeled_pairs):
        adj = np.zeros((n_completions, n_completions), dtype=bool)
        for w, l in labeled_pairs:
            adj[w][l] = True
        for k in range(n_completions):
            for i in range(n_completions):
                for j in range(n_completions):
                    if adj[i][k] and adj[k][j]:
                        adj[i][j] = True
        scores = adj.sum(axis=1).astype(float)
        std = scores.std()
        if std > 1e-8:
            scores = (scores - scores.mean()) / std
        return scores

    def construct_full_graph(scores):
        n = len(scores)
        pairs = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                if scores[i] > scores[j]:
                    pairs.append((i, j))
                else:
                    pairs.append((j, i))
        return pairs

    def get_log_prob_sums(model, input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        mask = attention_mask[:, 1:]
        return (token_log_probs * mask).sum(dim=-1)

    def train_dpo(policy_model, ref_log_probs_tensor, input_ids_all, attention_mask_all,
                  pairs, epochs=300, lr=1e-5, beta=0.1, device="cuda",
                  model_is_bf16=False):
        optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr)
        winners = torch.tensor([w for w, _ in pairs], dtype=torch.long, device=device)
        losers = torch.tensor([l for _, l in pairs], dtype=torch.long, device=device)

        log_probs_over_time = []
        use_amp = (device == "cuda") and (not model_is_bf16)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        for epoch in range(epochs):
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                policy_log_probs = get_log_prob_sums(policy_model, input_ids_all, attention_mask_all)
                policy_w = policy_log_probs[winners]
                policy_l = policy_log_probs[losers]
                ref_w = ref_log_probs_tensor[winners]
                ref_l = ref_log_probs_tensor[losers]
                dpo_logits = beta * ((policy_w - ref_w) - (policy_l - ref_l))
                losses = -nn.functional.logsigmoid(dpo_logits)
                loss = losses.mean()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            log_probs_over_time.append(policy_log_probs.detach().float().cpu().numpy().tolist())
        return log_probs_over_time

    def count_pair_agreements(model_scores, oracle_scores):
        model_scores = np.array(model_scores)
        oracle_scores = np.array(oracle_scores)
        n = len(oracle_scores)
        agreements = 0
        total = n * (n - 1) // 2
        for i in range(n):
            for j in range(i + 1, n):
                oracle_pref = oracle_scores[i] - oracle_scores[j]
                model_pref = model_scores[i] - model_scores[j]
                if oracle_pref * model_pref > 0:
                    agreements += 1
        return agreements, total

    def compute_ranking_metrics(model_scores, oracle_scores):
        agree, total = count_pair_agreements(model_scores, oracle_scores)
        frac = agree / total if total > 0 else 0.0
        tau, _ = kendalltau(model_scores, oracle_scores)
        rho, _ = spearmanr(model_scores, oracle_scores)
        return {"pairwise_agreement": frac, "kendall_tau": tau, "spearman_rho": rho}

    return {
        "sample_pairs_random": sample_pairs_random,
        "label_pairs": label_pairs,
        "fit_bradley_terry": fit_bradley_terry,
        "fit_win_rate": fit_win_rate,
        "fit_transitive_closure": fit_transitive_closure,
        "construct_full_graph": construct_full_graph,
        "get_log_prob_sums": get_log_prob_sums,
        "train_dpo": train_dpo,
        "count_pair_agreements": count_pair_agreements,
        "compute_ranking_metrics": compute_ranking_metrics,
    }


# ============================================================
# Experiment 1: Core Sample Efficiency (Same Oracle Budget)
# ============================================================

@app.function(**COMMON_KWARGS)
def exp1_sample_efficiency():
    """
    Core proof: GGDPO achieves higher agreement than DPO at the same oracle budget.
    N=15 completions, sweep K (oracle pairs).
    At low K, GGDPO has 105 training pairs vs K for DPO.
    At K=105 (all pairs), they converge.
    """
    import copy
    import random
    import string
    import json
    import os
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    h = _ggdpo_helpers()
    BASE_SEED = 44

    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    N = 15
    MAX_PAIRS = N * (N - 1) // 2  # 105
    K_values = [N - 1, N, 2 * N, 3 * N, MAX_PAIRS]  # [14, 15, 30, 45, 105]
    NUM_RUNS = 20
    EPOCHS = 200

    results = []

    for K in K_values:
        K = min(K, MAX_PAIRS)
        print(f"\n=== K={K} oracle pairs (N={N}, max={MAX_PAIRS}) ===")

        run_data = {"dpo_agreement": [], "ggdpo_agreement": [],
                    "dpo_kendall": [], "ggdpo_kendall": []}

        for run_idx in range(NUM_RUNS):
            seed = BASE_SEED + run_idx + K * 100
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            prompt = "".join(random.choices(string.ascii_letters + string.digits, k=20))

            pi_ref = GPT2LMHeadModel.from_pretrained(model_name).to(device)
            pi_ref.config.use_cache = False
            pi_ref.eval()
            pi_dpo = copy.deepcopy(pi_ref).train()
            pi_ggdpo = copy.deepcopy(pi_ref).train()

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            completions = []
            with torch.no_grad():
                for _ in range(N):
                    out = pi_ref.generate(
                        **{kk: v.clone() for kk, v in inputs.items()},
                        max_length=20, do_sample=True, top_k=50,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    completions.append(tokenizer.decode(out[0], skip_special_tokens=True))

            perm = np.random.permutation(N)
            oracle_scores = np.empty(N)
            for rank, idx in enumerate(perm):
                oracle_scores[idx] = rank + 1

            sampled_pairs = h["sample_pairs_random"](N, K)
            labeled_pairs = h["label_pairs"](sampled_pairs, oracle_scores)

            bt_scores = h["fit_bradley_terry"](N, labeled_pairs)
            full_graph = h["construct_full_graph"](bt_scores)

            tokens = tokenizer(completions, return_tensors="pt", padding=True, truncation=True)
            ids = tokens["input_ids"].to(device)
            mask = tokens["attention_mask"].to(device)

            with torch.no_grad():
                ref_lp = h["get_log_prob_sums"](pi_ref, ids, mask)

            dpo_lps = h["train_dpo"](pi_dpo, ref_lp, ids, mask, labeled_pairs,
                                     epochs=EPOCHS, device=device)
            ggdpo_lps = h["train_dpo"](pi_ggdpo, ref_lp, ids, mask, full_graph,
                                       epochs=EPOCHS, device=device)

            dpo_final = np.array(dpo_lps[-1])
            ggdpo_final = np.array(ggdpo_lps[-1])

            dpo_m = h["compute_ranking_metrics"](dpo_final, oracle_scores)
            ggdpo_m = h["compute_ranking_metrics"](ggdpo_final, oracle_scores)

            run_data["dpo_agreement"].append(dpo_m["pairwise_agreement"])
            run_data["ggdpo_agreement"].append(ggdpo_m["pairwise_agreement"])
            run_data["dpo_kendall"].append(dpo_m["kendall_tau"])
            run_data["ggdpo_kendall"].append(ggdpo_m["kendall_tau"])

            del pi_ref, pi_dpo, pi_ggdpo
            torch.cuda.empty_cache()

            if (run_idx + 1) % 5 == 0:
                print(f"  Run {run_idx+1}/{NUM_RUNS}: DPO={dpo_m['pairwise_agreement']:.3f} GGDPO={ggdpo_m['pairwise_agreement']:.3f}")

        result = {
            "N": N, "K": K, "max_pairs": MAX_PAIRS,
            "dpo_agreement_mean": float(np.mean(run_data["dpo_agreement"])),
            "dpo_agreement_std": float(np.std(run_data["dpo_agreement"])),
            "ggdpo_agreement_mean": float(np.mean(run_data["ggdpo_agreement"])),
            "ggdpo_agreement_std": float(np.std(run_data["ggdpo_agreement"])),
            "dpo_kendall_mean": float(np.mean(run_data["dpo_kendall"])),
            "ggdpo_kendall_mean": float(np.mean(run_data["ggdpo_kendall"])),
            "improvement": float(np.mean(run_data["ggdpo_agreement"]) - np.mean(run_data["dpo_agreement"])),
        }
        results.append(result)
        print(f"  K={K}: DPO={result['dpo_agreement_mean']:.3f}+-{result['dpo_agreement_std']:.3f} "
              f"GGDPO={result['ggdpo_agreement_mean']:.3f}+-{result['ggdpo_agreement_std']:.3f} "
              f"Improvement={result['improvement']:+.3f}")

    # Save JSON
    with open(os.path.join(RESULTS_DIR, "exp1_sample_efficiency.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    Ks = [r["K"] for r in results]
    dpo_means = [r["dpo_agreement_mean"] for r in results]
    dpo_stds = [r["dpo_agreement_std"] for r in results]
    ggdpo_means = [r["ggdpo_agreement_mean"] for r in results]
    ggdpo_stds = [r["ggdpo_agreement_std"] for r in results]
    improvements = [r["improvement"] for r in results]

    ax1.errorbar(Ks, dpo_means, yerr=dpo_stds, marker='o', capsize=4, label='DPO (K pairs)')
    ax1.errorbar(Ks, ggdpo_means, yerr=ggdpo_stds, marker='x', capsize=4, label='GGDPO (K -> 105 pairs)')
    ax1.set_xlabel("K (oracle pairs)")
    ax1.set_ylabel("Oracle Pairwise Agreement")
    ax1.set_title(f"Sample Efficiency: N={N} completions")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    bar_colors = ['green' if v > 0 else 'red' for v in improvements]
    ax2.bar([str(k) for k in Ks], improvements, color=bar_colors, alpha=0.7)
    ax2.set_xlabel("K (oracle pairs)")
    ax2.set_ylabel("GGDPO - DPO Agreement")
    ax2.set_title("GGDPO Improvement over DPO")
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp1_sample_efficiency.png"), dpi=150, bbox_inches="tight")
    plt.close()

    results_vol.commit()
    print("\n=== Experiment 1 Complete ===")
    print(json.dumps(results, indent=2))


# ============================================================
# Experiment 2: Scaling-N (The Killer Chart)
# ============================================================

@app.function(**COMMON_KWARGS)
def exp2_scaling_n():
    """
    As N grows, GGDPO advantage should widen.
    Fixed K=2N oracle pairs, N varies.
    GGDPO expands to C(N,2) pairs.
    Includes Full DPO upper bound.
    """
    import copy
    import random
    import string
    import json
    import os
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    h = _ggdpo_helpers()
    BASE_SEED = 42

    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    N_values = [5, 10, 15, 20, 30, 50]
    NUM_RUNS = 10
    EPOCHS = 200

    results = []

    for N in N_values:
        K = 2 * N
        max_pairs = N * (N - 1) // 2
        K = min(K, max_pairs)
        print(f"\n=== N={N}, K={K} oracle pairs, C(N,2)={max_pairs} ===")

        run_data = {"dpo_agreement": [], "ggdpo_agreement": [], "full_dpo_agreement": [],
                    "dpo_kendall": [], "ggdpo_kendall": [], "full_dpo_kendall": []}

        for run_idx in range(NUM_RUNS):
            seed = BASE_SEED + run_idx + N * 100
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            prompt = "".join(random.choices(string.ascii_letters + string.digits, k=20))

            pi_ref = GPT2LMHeadModel.from_pretrained(model_name).to(device)
            pi_ref.config.use_cache = False
            pi_ref.eval()
            pi_dpo = copy.deepcopy(pi_ref).train()
            pi_ggdpo = copy.deepcopy(pi_ref).train()
            pi_full = copy.deepcopy(pi_ref).train()

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            completions = []
            with torch.no_grad():
                for _ in range(N):
                    out = pi_ref.generate(
                        **{kk: v.clone() for kk, v in inputs.items()},
                        max_length=20, do_sample=True, top_k=50,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    completions.append(tokenizer.decode(out[0], skip_special_tokens=True))

            perm = np.random.permutation(N)
            oracle_scores = np.empty(N)
            for rank, idx in enumerate(perm):
                oracle_scores[idx] = rank + 1

            sampled_pairs = h["sample_pairs_random"](N, K)
            labeled_pairs = h["label_pairs"](sampled_pairs, oracle_scores)

            bt_scores = h["fit_bradley_terry"](N, labeled_pairs)
            ggdpo_graph = h["construct_full_graph"](bt_scores)

            all_pairs = h["sample_pairs_random"](N, max_pairs)
            all_labeled = h["label_pairs"](all_pairs, oracle_scores)

            tokens = tokenizer(completions, return_tensors="pt", padding=True, truncation=True)
            ids = tokens["input_ids"].to(device)
            mask = tokens["attention_mask"].to(device)

            with torch.no_grad():
                ref_lp = h["get_log_prob_sums"](pi_ref, ids, mask)

            dpo_lps = h["train_dpo"](pi_dpo, ref_lp, ids, mask, labeled_pairs,
                                     epochs=EPOCHS, device=device)
            ggdpo_lps = h["train_dpo"](pi_ggdpo, ref_lp, ids, mask, ggdpo_graph,
                                       epochs=EPOCHS, device=device)
            full_lps = h["train_dpo"](pi_full, ref_lp, ids, mask, all_labeled,
                                      epochs=EPOCHS, device=device)

            dpo_m = h["compute_ranking_metrics"](np.array(dpo_lps[-1]), oracle_scores)
            ggdpo_m = h["compute_ranking_metrics"](np.array(ggdpo_lps[-1]), oracle_scores)
            full_m = h["compute_ranking_metrics"](np.array(full_lps[-1]), oracle_scores)

            run_data["dpo_agreement"].append(dpo_m["pairwise_agreement"])
            run_data["ggdpo_agreement"].append(ggdpo_m["pairwise_agreement"])
            run_data["full_dpo_agreement"].append(full_m["pairwise_agreement"])
            run_data["dpo_kendall"].append(dpo_m["kendall_tau"])
            run_data["ggdpo_kendall"].append(ggdpo_m["kendall_tau"])
            run_data["full_dpo_kendall"].append(full_m["kendall_tau"])

            del pi_ref, pi_dpo, pi_ggdpo, pi_full
            torch.cuda.empty_cache()

            if (run_idx + 1) % 5 == 0:
                print(f"  Run {run_idx+1}/{NUM_RUNS}: DPO={dpo_m['pairwise_agreement']:.3f} "
                      f"GGDPO={ggdpo_m['pairwise_agreement']:.3f} Full={full_m['pairwise_agreement']:.3f}")

        result = {
            "N": N, "K": K, "max_pairs": max_pairs,
            "dpo_agreement_mean": float(np.mean(run_data["dpo_agreement"])),
            "dpo_agreement_std": float(np.std(run_data["dpo_agreement"])),
            "ggdpo_agreement_mean": float(np.mean(run_data["ggdpo_agreement"])),
            "ggdpo_agreement_std": float(np.std(run_data["ggdpo_agreement"])),
            "full_dpo_agreement_mean": float(np.mean(run_data["full_dpo_agreement"])),
            "full_dpo_agreement_std": float(np.std(run_data["full_dpo_agreement"])),
            "improvement_over_dpo": float(np.mean(run_data["ggdpo_agreement"]) - np.mean(run_data["dpo_agreement"])),
            "gap_to_full": float(np.mean(run_data["full_dpo_agreement"]) - np.mean(run_data["ggdpo_agreement"])),
        }
        results.append(result)
        print(f"  N={N}: DPO={result['dpo_agreement_mean']:.3f} GGDPO={result['ggdpo_agreement_mean']:.3f} "
              f"Full={result['full_dpo_agreement_mean']:.3f} Improvement={result['improvement_over_dpo']:+.3f}")

    with open(os.path.join(RESULTS_DIR, "exp2_scaling_n.json"), "w") as f:
        json.dump(results, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    Ns = [r["N"] for r in results]
    dpo_m_list = [r["dpo_agreement_mean"] for r in results]
    dpo_s = [r["dpo_agreement_std"] for r in results]
    ggdpo_m_list = [r["ggdpo_agreement_mean"] for r in results]
    ggdpo_s = [r["ggdpo_agreement_std"] for r in results]
    full_m_list = [r["full_dpo_agreement_mean"] for r in results]
    full_s = [r["full_dpo_agreement_std"] for r in results]
    improvements = [r["improvement_over_dpo"] for r in results]

    ax1.errorbar(Ns, dpo_m_list, yerr=dpo_s, marker='o', capsize=4, label='DPO (K=2N pairs)')
    ax1.errorbar(Ns, ggdpo_m_list, yerr=ggdpo_s, marker='x', capsize=4, label='GGDPO (K=2N -> C(N,2) pairs)')
    ax1.errorbar(Ns, full_m_list, yerr=full_s, marker='s', capsize=4, linestyle='--', label='Full DPO (all C(N,2) pairs)')
    ax1.set_xlabel("N (completions per prompt)")
    ax1.set_ylabel("Oracle Pairwise Agreement")
    ax1.set_title("GGDPO Advantage Grows with N")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    bar_colors = ['green' if v > 0 else 'red' for v in improvements]
    ax2.bar([str(n) for n in Ns], improvements, color=bar_colors, alpha=0.7)
    ax2.set_xlabel("N")
    ax2.set_ylabel("GGDPO - DPO Agreement")
    ax2.set_title("GGDPO Improvement over DPO")
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp2_scaling_n.png"), dpi=150, bbox_inches="tight")
    plt.close()

    results_vol.commit()
    print("\n=== Experiment 2 Complete ===")
    print(json.dumps(results, indent=2))


# ============================================================
# Experiment 3: BT Estimation Ablation
# ============================================================

@app.function(**COMMON_KWARGS)
def exp3_bt_ablation():
    """
    Justify Bradley-Terry as the graph estimation method.
    Compare BT vs Win Rate vs Transitive Closure vs DPO baseline.
    Also: K sweep showing BT accuracy vs #pairs.
    """
    import copy
    import random
    import string
    import json
    import os
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    h = _ggdpo_helpers()
    BASE_SEED = 42

    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    N = 20
    MAX_PAIRS = N * (N - 1) // 2  # 190
    NUM_RUNS = 20
    EPOCHS = 200

    # Part 1: Compare graph estimation methods at K=20
    print("=== Part 1: Graph Estimation Methods ===")
    K_fixed = 20
    methods = {
        "bradley_terry": h["fit_bradley_terry"],
        "win_rate": h["fit_win_rate"],
        "transitive_closure": h["fit_transitive_closure"],
    }
    method_results = {}

    for method_name, fit_fn in methods.items():
        run_data = {"agreement": [], "kendall": []}
        for run_idx in range(NUM_RUNS):
            seed = BASE_SEED + run_idx
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            prompt = "".join(random.choices(string.ascii_letters + string.digits, k=20))
            pi_ref = GPT2LMHeadModel.from_pretrained(model_name).to(device)
            pi_ref.config.use_cache = False
            pi_ref.eval()
            pi_model = copy.deepcopy(pi_ref).train()

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            completions = []
            with torch.no_grad():
                for _ in range(N):
                    out = pi_ref.generate(
                        **{kk: v.clone() for kk, v in inputs.items()},
                        max_length=20, do_sample=True, top_k=50,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    completions.append(tokenizer.decode(out[0], skip_special_tokens=True))

            perm = np.random.permutation(N)
            oracle_scores = np.empty(N)
            for rank, idx in enumerate(perm):
                oracle_scores[idx] = rank + 1

            sampled_pairs = h["sample_pairs_random"](N, K_fixed)
            labeled_pairs = h["label_pairs"](sampled_pairs, oracle_scores)

            est_scores = fit_fn(N, labeled_pairs)
            full_graph = h["construct_full_graph"](est_scores)

            tokens = tokenizer(completions, return_tensors="pt", padding=True, truncation=True)
            ids = tokens["input_ids"].to(device)
            mask_tensor = tokens["attention_mask"].to(device)

            with torch.no_grad():
                ref_lp = h["get_log_prob_sums"](pi_ref, ids, mask_tensor)

            lps = h["train_dpo"](pi_model, ref_lp, ids, mask_tensor, full_graph,
                                 epochs=EPOCHS, device=device)

            m = h["compute_ranking_metrics"](np.array(lps[-1]), oracle_scores)
            run_data["agreement"].append(m["pairwise_agreement"])
            run_data["kendall"].append(m["kendall_tau"])

            del pi_ref, pi_model
            torch.cuda.empty_cache()

        method_results[method_name] = {
            "agreement_mean": float(np.mean(run_data["agreement"])),
            "agreement_std": float(np.std(run_data["agreement"])),
            "kendall_mean": float(np.mean(run_data["kendall"])),
        }
        print(f"  {method_name}: agreement={method_results[method_name]['agreement_mean']:.3f}"
              f"+-{method_results[method_name]['agreement_std']:.3f}")

    # DPO baseline (no expansion)
    dpo_data = {"agreement": [], "kendall": []}
    for run_idx in range(NUM_RUNS):
        seed = BASE_SEED + run_idx
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        prompt = "".join(random.choices(string.ascii_letters + string.digits, k=20))
        pi_ref = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        pi_ref.config.use_cache = False
        pi_ref.eval()
        pi_dpo = copy.deepcopy(pi_ref).train()

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        completions = []
        with torch.no_grad():
            for _ in range(N):
                out = pi_ref.generate(
                    **{kk: v.clone() for kk, v in inputs.items()},
                    max_length=20, do_sample=True, top_k=50,
                    pad_token_id=tokenizer.eos_token_id
                )
                completions.append(tokenizer.decode(out[0], skip_special_tokens=True))

        perm = np.random.permutation(N)
        oracle_scores = np.empty(N)
        for rank, idx in enumerate(perm):
            oracle_scores[idx] = rank + 1

        sampled_pairs = h["sample_pairs_random"](N, K_fixed)
        labeled_pairs = h["label_pairs"](sampled_pairs, oracle_scores)

        tokens = tokenizer(completions, return_tensors="pt", padding=True, truncation=True)
        ids = tokens["input_ids"].to(device)
        mask_tensor = tokens["attention_mask"].to(device)

        with torch.no_grad():
            ref_lp = h["get_log_prob_sums"](pi_ref, ids, mask_tensor)

        lps = h["train_dpo"](pi_dpo, ref_lp, ids, mask_tensor, labeled_pairs,
                             epochs=EPOCHS, device=device)
        m = h["compute_ranking_metrics"](np.array(lps[-1]), oracle_scores)
        dpo_data["agreement"].append(m["pairwise_agreement"])
        dpo_data["kendall"].append(m["kendall_tau"])

        del pi_ref, pi_dpo
        torch.cuda.empty_cache()

    method_results["dpo_baseline"] = {
        "agreement_mean": float(np.mean(dpo_data["agreement"])),
        "agreement_std": float(np.std(dpo_data["agreement"])),
        "kendall_mean": float(np.mean(dpo_data["kendall"])),
    }

    # Part 2: K sweep for BT estimation accuracy
    print("\n=== Part 2: BT Accuracy vs K ===")
    K_sweep = [10, 15, 20, 30, 40, 60]
    bt_k_results = []

    for K in K_sweep:
        K = min(K, MAX_PAIRS)
        accuracies = []
        for run_idx in range(NUM_RUNS):
            seed = BASE_SEED + run_idx + K * 50
            np.random.seed(seed)
            random.seed(seed)

            oracle_scores = np.random.permutation(N).astype(float) + 1
            sampled_pairs = h["sample_pairs_random"](N, K)
            labeled_pairs = h["label_pairs"](sampled_pairs, oracle_scores)
            bt_scores = h["fit_bradley_terry"](N, labeled_pairs)
            agree, total = h["count_pair_agreements"](bt_scores, oracle_scores)
            accuracies.append(agree / total)

        bt_k_results.append({
            "K": K,
            "bt_accuracy_mean": float(np.mean(accuracies)),
            "bt_accuracy_std": float(np.std(accuracies)),
        })
        print(f"  K={K}: BT accuracy={np.mean(accuracies):.3f}+-{np.std(accuracies):.3f}")

    all_results = {
        "graph_estimation_methods": method_results,
        "bt_accuracy_vs_k": bt_k_results,
    }
    with open(os.path.join(RESULTS_DIR, "exp3_bt_ablation.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    names = list(method_results.keys())
    means = [method_results[n]["agreement_mean"] for n in names]
    stds = [method_results[n]["agreement_std"] for n in names]
    plot_colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e']
    ax1.bar(names, means, yerr=stds, capsize=5, color=plot_colors[:len(names)], alpha=0.8)
    ax1.set_ylabel("Oracle Pairwise Agreement")
    ax1.set_title(f"Graph Estimation Methods (N={N}, K={K_fixed})")
    ax1.grid(True, alpha=0.3, axis='y')

    Ks = [r["K"] for r in bt_k_results]
    bt_means = [r["bt_accuracy_mean"] for r in bt_k_results]
    bt_stds = [r["bt_accuracy_std"] for r in bt_k_results]
    ax2.errorbar(Ks, bt_means, yerr=bt_stds, marker='o', capsize=4, color='#1f77b4')
    ax2.set_xlabel("K (oracle pairs)")
    ax2.set_ylabel("BT Estimation Accuracy")
    ax2.set_title(f"BT Accuracy vs Oracle Pairs (N={N})")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp3_bt_ablation.png"), dpi=150, bbox_inches="tight")
    plt.close()

    results_vol.commit()
    print("\n=== Experiment 3 Complete ===")
    print(json.dumps(all_results, indent=2))


# ============================================================
# Experiment 4: Real Model at Scale (Qwen3-1.7B + Reward Model)
# ============================================================

@app.function(**COMMON_KWARGS)
def exp4_real_model_scaled():
    """
    Substantially expanded real-model experiment.
    Qwen3-1.7B on UltraFeedback prompts, LoRA DPO, gradient variance measurement.
    """
    import random
    import json
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    h = _ggdpo_helpers()

    model_name = "Qwen/Qwen3-1.7B"
    reward_model_name = "Skywork/Skywork-Reward-V2-Qwen3-1.7B"

    print("Loading tokenizer and reward model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_name, torch_dtype=torch.bfloat16, num_labels=1
    ).to(device)
    reward_model.eval()

    print("Loading UltraFeedback prompts...")
    ds = load_dataset("openbmb/UltraFeedback", split="train")
    prompts = [ex["instruction"] for ex in ds]
    random.seed(42)
    random.shuffle(prompts)
    prompts = prompts[:200]

    N_values = [10, 20, 30]
    NUM_RUNS = 3
    EPOCHS_DPO = 2
    BATCH_SIZE = 4
    LR = 5e-6

    results = []

    for N in N_values:
        K = N
        max_pairs = N * (N - 1) // 2
        print(f"\n=== N={N}, K={K}, C(N,2)={max_pairs} ===")

        run_data = {
            "dpo_reward": [], "ggdpo_reward": [],
            "dpo_grad_var": [], "ggdpo_grad_var": [],
        }

        for run_idx in range(NUM_RUNS):
            seed = 42 + run_idx + N * 10
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            run_prompts = prompts[run_idx * 10:(run_idx + 1) * 10]

            print(f"  Run {run_idx+1}/{NUM_RUNS}: Generating {N} completions per prompt...")
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16
            ).to(device)
            base_model.eval()

            all_dpo_pairs = []
            all_ggdpo_pairs = []

            for p_idx, prompt_text in enumerate(run_prompts):
                inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=256).to(device)
                completions = []
                with torch.no_grad():
                    for _ in range(N):
                        out = base_model.generate(
                            **{kk: v.clone() for kk, v in inputs.items()},
                            max_new_tokens=96, do_sample=True, top_k=50, temperature=0.8,
                            pad_token_id=tokenizer.pad_token_id
                        )
                        text = tokenizer.decode(out[0], skip_special_tokens=True)
                        completions.append(text)

                scores = []
                for comp in completions:
                    enc = reward_tokenizer(comp, return_tensors="pt", truncation=True, max_length=512).to(device)
                    with torch.no_grad():
                        score = reward_model(**enc).logits.squeeze().float().item()
                    scores.append(score)
                scores = np.array(scores)

                sampled_pairs = h["sample_pairs_random"](N, K)
                labeled_pairs = h["label_pairs"](sampled_pairs, scores)

                bt_scores = h["fit_bradley_terry"](N, labeled_pairs)
                ggdpo_graph = h["construct_full_graph"](bt_scores)

                for w, l in labeled_pairs:
                    all_dpo_pairs.append((prompt_text, completions[w], completions[l]))
                for w, l in ggdpo_graph:
                    all_ggdpo_pairs.append((prompt_text, completions[w], completions[l]))

                if (p_idx + 1) % 5 == 0:
                    print(f"    Prompt {p_idx+1}/{len(run_prompts)} processed")

            del base_model
            torch.cuda.empty_cache()

            print(f"  DPO pairs: {len(all_dpo_pairs)}, GGDPO pairs: {len(all_ggdpo_pairs)}")

            def train_lora_dpo(pairs, label=""):
                ref_model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.bfloat16
                ).to(device)
                ref_model.eval()
                for p in ref_model.parameters():
                    p.requires_grad = False

                model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.bfloat16
                ).to(device)
                model.config.use_cache = False
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    lora_dropout=0.0,
                )
                model = get_peft_model(model, lora_config)
                model.enable_input_require_grads()
                model.train()

                optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
                total_loss = 0
                num_batches = 0
                grad_norms = []

                for epoch in range(EPOCHS_DPO):
                    random.shuffle(pairs)
                    for start in range(0, len(pairs), BATCH_SIZE):
                        batch = pairs[start:start + BATCH_SIZE]
                        chosen_texts = [f"{p}\n\n{c}" for p, c, _ in batch]
                        rejected_texts = [f"{p}\n\n{r}" for p, _, r in batch]

                        chosen_enc = tokenizer(chosen_texts, return_tensors="pt", padding=True,
                                             truncation=True, max_length=512).to(device)
                        rejected_enc = tokenizer(rejected_texts, return_tensors="pt", padding=True,
                                               truncation=True, max_length=512).to(device)

                        optimizer.zero_grad()

                        chosen_lp = h["get_log_prob_sums"](model, chosen_enc["input_ids"], chosen_enc["attention_mask"])
                        rejected_lp = h["get_log_prob_sums"](model, rejected_enc["input_ids"], rejected_enc["attention_mask"])

                        with torch.no_grad():
                            ref_chosen_lp = h["get_log_prob_sums"](ref_model, chosen_enc["input_ids"], chosen_enc["attention_mask"])
                            ref_rejected_lp = h["get_log_prob_sums"](ref_model, rejected_enc["input_ids"], rejected_enc["attention_mask"])

                        beta = 0.1
                        dpo_logits = beta * ((chosen_lp - ref_chosen_lp) - (rejected_lp - ref_rejected_lp))
                        loss = -nn.functional.logsigmoid(dpo_logits).mean()

                        loss.backward()

                        grad_norm = 0.0
                        for p in model.parameters():
                            if p.grad is not None:
                                grad_norm += p.grad.data.norm(2).item() ** 2
                        grad_norms.append(grad_norm ** 0.5)

                        optimizer.step()
                        total_loss += loss.item()
                        num_batches += 1

                        if num_batches % 100 == 0:
                            print(f"    {label} Epoch {epoch+1}/{EPOCHS_DPO}, Batch {num_batches}, loss={total_loss/num_batches:.4f}")

                avg_loss = total_loss / max(num_batches, 1)
                grad_variance = float(np.var(grad_norms)) if grad_norms else 0.0

                del ref_model
                torch.cuda.empty_cache()
                return model, avg_loss, grad_variance

            print(f"  Training DPO...")
            dpo_model, dpo_loss, dpo_grad_var = train_lora_dpo(all_dpo_pairs, "DPO")
            print(f"  Training GGDPO...")
            ggdpo_model, ggdpo_loss, ggdpo_grad_var = train_lora_dpo(all_ggdpo_pairs, "GGDPO")

            def evaluate_model(model, eval_prompts, n_gen=2, max_new_tokens=96):
                model.eval()
                all_scores = []
                for idx, pt in enumerate(eval_prompts[:20]):
                    inputs = tokenizer(pt, return_tensors="pt", truncation=True, max_length=256).to(device)
                    with torch.no_grad():
                        for _ in range(n_gen):
                            out = model.generate(
                                **{kk: v.clone() for kk, v in inputs.items()},
                                max_new_tokens=max_new_tokens, do_sample=True, top_k=50,
                                pad_token_id=tokenizer.pad_token_id
                            )
                            text = tokenizer.decode(out[0], skip_special_tokens=True)
                            enc = reward_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                            with torch.no_grad():
                                score = reward_model(**enc).logits.squeeze().float().item()
                            all_scores.append(score)
                    if (idx + 1) % 10 == 0:
                        print(f"    Eval: {idx+1}/20 prompts scored")
                return float(np.mean(all_scores)), float(np.std(all_scores))

            eval_prompts = prompts[100:130]
            print(f"  Evaluating DPO model...")
            dpo_reward_mean, dpo_reward_std = evaluate_model(dpo_model, eval_prompts)
            print(f"  Evaluating GGDPO model...")
            ggdpo_reward_mean, ggdpo_reward_std = evaluate_model(ggdpo_model, eval_prompts)

            run_data["dpo_reward"].append(dpo_reward_mean)
            run_data["ggdpo_reward"].append(ggdpo_reward_mean)
            run_data["dpo_grad_var"].append(dpo_grad_var)
            run_data["ggdpo_grad_var"].append(ggdpo_grad_var)

            del dpo_model, ggdpo_model
            torch.cuda.empty_cache()

            print(f"  Run {run_idx+1}: DPO reward={dpo_reward_mean:.3f} GGDPO reward={ggdpo_reward_mean:.3f} "
                  f"DPO grad_var={dpo_grad_var:.4f} GGDPO grad_var={ggdpo_grad_var:.4f}")

        result = {
            "N": N, "K": K, "max_pairs": max_pairs,
            "dpo_reward_mean": float(np.mean(run_data["dpo_reward"])),
            "dpo_reward_std": float(np.std(run_data["dpo_reward"])),
            "ggdpo_reward_mean": float(np.mean(run_data["ggdpo_reward"])),
            "ggdpo_reward_std": float(np.std(run_data["ggdpo_reward"])),
            "dpo_grad_variance_mean": float(np.mean(run_data["dpo_grad_var"])),
            "ggdpo_grad_variance_mean": float(np.mean(run_data["ggdpo_grad_var"])),
            "reward_improvement": float(np.mean(run_data["ggdpo_reward"]) - np.mean(run_data["dpo_reward"])),
        }
        results.append(result)
        print(f"  N={N}: DPO reward={result['dpo_reward_mean']:.3f} GGDPO reward={result['ggdpo_reward_mean']:.3f} "
              f"Improvement={result['reward_improvement']:+.3f}")

    with open(os.path.join(RESULTS_DIR, "exp4_real_model.json"), "w") as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    Ns = [r["N"] for r in results]
    x = np.arange(len(Ns))
    width = 0.35

    dpo_rewards = [r["dpo_reward_mean"] for r in results]
    ggdpo_rewards = [r["ggdpo_reward_mean"] for r in results]
    dpo_rstd = [r["dpo_reward_std"] for r in results]
    ggdpo_rstd = [r["ggdpo_reward_std"] for r in results]
    axes[0].bar(x - width/2, dpo_rewards, width, yerr=dpo_rstd, label='DPO', capsize=4)
    axes[0].bar(x + width/2, ggdpo_rewards, width, yerr=ggdpo_rstd, label='GGDPO', capsize=4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(Ns)
    axes[0].set_xlabel("N (completions)")
    axes[0].set_ylabel("Post-Alignment Reward Score")
    axes[0].set_title("Reward Model Score")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    dpo_gv = [r["dpo_grad_variance_mean"] for r in results]
    ggdpo_gv = [r["ggdpo_grad_variance_mean"] for r in results]
    axes[1].bar(x - width/2, dpo_gv, width, label='DPO', capsize=4)
    axes[1].bar(x + width/2, ggdpo_gv, width, label='GGDPO', capsize=4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(Ns)
    axes[1].set_xlabel("N (completions)")
    axes[1].set_ylabel("Gradient Norm Variance")
    axes[1].set_title("Training Gradient Variance")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    improvements = [r["reward_improvement"] for r in results]
    bar_colors = ['green' if v > 0 else 'red' for v in improvements]
    axes[2].bar([str(n) for n in Ns], improvements, color=bar_colors, alpha=0.7)
    axes[2].set_xlabel("N")
    axes[2].set_ylabel("GGDPO - DPO Reward")
    axes[2].set_title("Reward Improvement")
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp4_real_model.png"), dpi=150, bbox_inches="tight")
    plt.close()

    results_vol.commit()
    print("\n=== Experiment 4 Complete ===")
    print(json.dumps(results, indent=2))


# ============================================================
# Experiment 5: UltraFeedback Sample Efficiency
# ============================================================

@app.function(**COMMON_KWARGS)
def exp5_ultrafeedback():
    """
    UltraFeedback with ground truth from its own ratings.
    4 completions/prompt, K in {1,2,3,6}, 3 runs per K.
    Evaluate with both reward model AND UltraFeedback's overall_score.
    """
    import random
    import json
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    h = _ggdpo_helpers()

    model_name = "Qwen/Qwen3-1.7B"
    reward_model_name = "Skywork/Skywork-Reward-V2-Qwen3-1.7B"

    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_name, torch_dtype=torch.bfloat16, num_labels=1
    ).to(device)
    reward_model.eval()

    print("Loading UltraFeedback dataset...")
    ds = load_dataset("openbmb/UltraFeedback", split="train")

    # Filter for examples with 4+ completions that have overall_score
    # UltraFeedback schema: each completion has 'overall_score' as a direct field (float),
    # and 'annotations' dict with per-aspect ratings (helpfulness, honesty, etc.)
    valid_examples = []
    for ex in ds:
        if "completions" in ex and len(ex["completions"]) >= 4:
            completions = ex["completions"][:4]
            scores = []
            valid = True
            for c in completions:
                if "overall_score" in c and c["overall_score"] is not None:
                    try:
                        scores.append(float(c["overall_score"]))
                    except (ValueError, TypeError):
                        valid = False
                        break
                else:
                    valid = False
                    break
            if valid and len(scores) == 4:
                valid_examples.append({
                    "instruction": ex["instruction"],
                    "completions": [c["response"] for c in completions],
                    "scores": scores,
                })
        if len(valid_examples) >= 500:
            break

    print(f"Found {len(valid_examples)} valid examples with 4 completions + scores")

    N = 4
    MAX_PAIRS = N * (N - 1) // 2  # 6
    K_values = [1, 2, 3, 6]
    NUM_RUNS = 3
    NUM_PROMPTS = 200
    EPOCHS_DPO = 2
    BATCH_SIZE = 4
    LR = 5e-6

    results = []

    for K in K_values:
        K = min(K, MAX_PAIRS)
        print(f"\n=== K={K} oracle pairs per prompt (out of {MAX_PAIRS}) ===")

        run_data = {
            "reward_score": {"dpo": [], "ggdpo": []},
            "uf_agreement": {"dpo": [], "ggdpo": []},
        }

        for run_idx in range(NUM_RUNS):
            seed = 42 + run_idx + K * 100
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            examples = valid_examples[:NUM_PROMPTS]

            all_dpo_pairs = []
            all_ggdpo_pairs = []
            uf_scores_list = []

            for ex in examples:
                instruction = ex["instruction"]
                completions = ex["completions"]
                uf_scores = np.array(ex["scores"])
                uf_scores_list.append(uf_scores)

                sampled_pairs = h["sample_pairs_random"](N, K)
                labeled_pairs = h["label_pairs"](sampled_pairs, uf_scores)

                bt_scores = h["fit_bradley_terry"](N, labeled_pairs)
                ggdpo_graph = h["construct_full_graph"](bt_scores)

                for w, l in labeled_pairs:
                    all_dpo_pairs.append((instruction, completions[w], completions[l]))
                for w, l in ggdpo_graph:
                    all_ggdpo_pairs.append((instruction, completions[w], completions[l]))

            print(f"  Run {run_idx+1}: DPO pairs={len(all_dpo_pairs)}, GGDPO pairs={len(all_ggdpo_pairs)}")

            def train_and_eval(pairs, label=""):
                ref_model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.bfloat16
                ).to(device)
                ref_model.eval()
                for p in ref_model.parameters():
                    p.requires_grad = False

                model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.bfloat16
                ).to(device)
                model.config.use_cache = False
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    lora_dropout=0.0,
                )
                model = get_peft_model(model, lora_config)
                model.enable_input_require_grads()
                model.train()

                optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
                total_loss = 0
                num_batches = 0

                for epoch in range(EPOCHS_DPO):
                    random.shuffle(pairs)
                    for start in range(0, len(pairs), BATCH_SIZE):
                        batch = pairs[start:start + BATCH_SIZE]
                        chosen_texts = [f"{p}\n\n{c}" for p, c, _ in batch]
                        rejected_texts = [f"{p}\n\n{r}" for p, _, r in batch]

                        chosen_enc = tokenizer(chosen_texts, return_tensors="pt", padding=True,
                                             truncation=True, max_length=512).to(device)
                        rejected_enc = tokenizer(rejected_texts, return_tensors="pt", padding=True,
                                               truncation=True, max_length=512).to(device)

                        optimizer.zero_grad()

                        chosen_lp = h["get_log_prob_sums"](model, chosen_enc["input_ids"], chosen_enc["attention_mask"])
                        rejected_lp = h["get_log_prob_sums"](model, rejected_enc["input_ids"], rejected_enc["attention_mask"])

                        with torch.no_grad():
                            ref_chosen_lp = h["get_log_prob_sums"](ref_model, chosen_enc["input_ids"], chosen_enc["attention_mask"])
                            ref_rejected_lp = h["get_log_prob_sums"](ref_model, rejected_enc["input_ids"], rejected_enc["attention_mask"])

                        beta = 0.1
                        dpo_logits = beta * ((chosen_lp - ref_chosen_lp) - (rejected_lp - ref_rejected_lp))
                        loss = -nn.functional.logsigmoid(dpo_logits).mean()

                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        num_batches += 1

                        if num_batches % 200 == 0:
                            print(f"    {label} batch {num_batches}, loss={total_loss/num_batches:.4f}")

                # Evaluate: reward scores on held-out generations
                model.eval()
                eval_examples = valid_examples[NUM_PROMPTS:NUM_PROMPTS + 50]
                reward_scores = []
                uf_agreements = 0
                uf_total = 0

                for idx, ex in enumerate(eval_examples):
                    instruction = ex["instruction"]
                    inputs = tokenizer(instruction, return_tensors="pt", truncation=True, max_length=256).to(device)
                    with torch.no_grad():
                        out = model.generate(
                            **{kk: v.clone() for kk, v in inputs.items()},
                            max_new_tokens=128, do_sample=True, top_k=50,
                            pad_token_id=tokenizer.pad_token_id
                        )
                        text = tokenizer.decode(out[0], skip_special_tokens=True)
                        enc = reward_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                        score = reward_model(**enc).logits.squeeze().float().item()
                        reward_scores.append(score)

                    # UltraFeedback agreement: check if model's logprobs agree with UF scores
                    completions = ex["completions"]
                    uf_scores = np.array(ex["scores"])
                    comp_encs = tokenizer(completions, return_tensors="pt", padding=True,
                                        truncation=True, max_length=256).to(device)
                    with torch.no_grad():
                        model_lps = h["get_log_prob_sums"](model, comp_encs["input_ids"],
                                                           comp_encs["attention_mask"])
                        model_lps_np = model_lps.float().cpu().numpy()

                    agree, total = h["count_pair_agreements"](model_lps_np, uf_scores)
                    uf_agreements += agree
                    uf_total += total

                del ref_model, model
                torch.cuda.empty_cache()

                return float(np.mean(reward_scores)), uf_agreements / max(uf_total, 1)

            print(f"  Training & evaluating DPO...")
            dpo_reward, dpo_uf_agree = train_and_eval(all_dpo_pairs, "DPO")
            print(f"  Training & evaluating GGDPO...")
            ggdpo_reward, ggdpo_uf_agree = train_and_eval(all_ggdpo_pairs, "GGDPO")

            run_data["reward_score"]["dpo"].append(dpo_reward)
            run_data["reward_score"]["ggdpo"].append(ggdpo_reward)
            run_data["uf_agreement"]["dpo"].append(dpo_uf_agree)
            run_data["uf_agreement"]["ggdpo"].append(ggdpo_uf_agree)

            print(f"  Run {run_idx+1}: DPO reward={dpo_reward:.3f} GGDPO reward={ggdpo_reward:.3f} "
                  f"DPO UF agree={dpo_uf_agree:.3f} GGDPO UF agree={ggdpo_uf_agree:.3f}")

        result = {
            "K": K, "N": N, "max_pairs": MAX_PAIRS,
            "dpo_reward_mean": float(np.mean(run_data["reward_score"]["dpo"])),
            "dpo_reward_std": float(np.std(run_data["reward_score"]["dpo"])),
            "ggdpo_reward_mean": float(np.mean(run_data["reward_score"]["ggdpo"])),
            "ggdpo_reward_std": float(np.std(run_data["reward_score"]["ggdpo"])),
            "dpo_uf_agreement_mean": float(np.mean(run_data["uf_agreement"]["dpo"])),
            "dpo_uf_agreement_std": float(np.std(run_data["uf_agreement"]["dpo"])),
            "ggdpo_uf_agreement_mean": float(np.mean(run_data["uf_agreement"]["ggdpo"])),
            "ggdpo_uf_agreement_std": float(np.std(run_data["uf_agreement"]["ggdpo"])),
            "reward_improvement": float(np.mean(run_data["reward_score"]["ggdpo"]) - np.mean(run_data["reward_score"]["dpo"])),
            "uf_agreement_improvement": float(np.mean(run_data["uf_agreement"]["ggdpo"]) - np.mean(run_data["uf_agreement"]["dpo"])),
        }
        results.append(result)
        print(f"  K={K}: DPO reward={result['dpo_reward_mean']:.3f}+-{result['dpo_reward_std']:.3f} "
              f"GGDPO reward={result['ggdpo_reward_mean']:.3f}+-{result['ggdpo_reward_std']:.3f}")

    with open(os.path.join(RESULTS_DIR, "exp5_ultrafeedback.json"), "w") as f:
        json.dump(results, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    Ks = [r["K"] for r in results]

    dpo_r = [r["dpo_reward_mean"] for r in results]
    dpo_rs = [r["dpo_reward_std"] for r in results]
    ggdpo_r = [r["ggdpo_reward_mean"] for r in results]
    ggdpo_rs = [r["ggdpo_reward_std"] for r in results]
    ax1.errorbar(Ks, dpo_r, yerr=dpo_rs, marker='o', capsize=4, label='DPO')
    ax1.errorbar(Ks, ggdpo_r, yerr=ggdpo_rs, marker='x', capsize=4, label='GGDPO')
    ax1.set_xlabel("K (oracle pairs per prompt)")
    ax1.set_ylabel("Reward Model Score")
    ax1.set_title("UltraFeedback: Reward Score vs Oracle Budget")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    dpo_uf = [r["dpo_uf_agreement_mean"] for r in results]
    dpo_ufs = [r["dpo_uf_agreement_std"] for r in results]
    ggdpo_uf = [r["ggdpo_uf_agreement_mean"] for r in results]
    ggdpo_ufs = [r["ggdpo_uf_agreement_std"] for r in results]
    ax2.errorbar(Ks, dpo_uf, yerr=dpo_ufs, marker='o', capsize=4, label='DPO')
    ax2.errorbar(Ks, ggdpo_uf, yerr=ggdpo_ufs, marker='x', capsize=4, label='GGDPO')
    ax2.set_xlabel("K (oracle pairs per prompt)")
    ax2.set_ylabel("UltraFeedback Agreement")
    ax2.set_title("UltraFeedback: Agreement with Human Ratings")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp5_ultrafeedback.png"), dpi=150, bbox_inches="tight")
    plt.close()

    results_vol.commit()
    print("\n=== Experiment 5 Complete ===")
    print(json.dumps(results, indent=2))


# ============================================================
# Experiment 6: Gradient Variance Reduction (Direct Measurement)
# ============================================================

@app.function(**COMMON_KWARGS)
def exp6_gradient_variance():
    """
    Directly measure gradient variance for DPO vs GGDPO.
    GPT-2, N=20, K=20. Compute DPO loss gradient for multiple mini-batches.
    Measure variance of gradient norm across batches + cosine similarity to full-data gradient.
    """
    import copy
    import random
    import string
    import json
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    h = _ggdpo_helpers()
    BASE_SEED = 42

    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    N = 20
    K = 20
    MAX_PAIRS = N * (N - 1) // 2  # 190
    NUM_RUNS = 20
    MINI_BATCH_SIZE = 10

    results = {"dpo": [], "ggdpo": []}

    for run_idx in range(NUM_RUNS):
        seed = BASE_SEED + run_idx
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        prompt = "".join(random.choices(string.ascii_letters + string.digits, k=20))

        pi_ref = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        pi_ref.config.use_cache = False
        pi_ref.eval()

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        completions = []
        with torch.no_grad():
            for _ in range(N):
                out = pi_ref.generate(
                    **{kk: v.clone() for kk, v in inputs.items()},
                    max_length=20, do_sample=True, top_k=50,
                    pad_token_id=tokenizer.eos_token_id
                )
                completions.append(tokenizer.decode(out[0], skip_special_tokens=True))

        perm = np.random.permutation(N)
        oracle_scores = np.empty(N)
        for rank, idx in enumerate(perm):
            oracle_scores[idx] = rank + 1

        sampled_pairs = h["sample_pairs_random"](N, K)
        labeled_pairs = h["label_pairs"](sampled_pairs, oracle_scores)

        bt_scores = h["fit_bradley_terry"](N, labeled_pairs)
        ggdpo_graph = h["construct_full_graph"](bt_scores)

        tokens = tokenizer(completions, return_tensors="pt", padding=True, truncation=True)
        ids = tokens["input_ids"].to(device)
        mask = tokens["attention_mask"].to(device)

        with torch.no_grad():
            ref_lp = h["get_log_prob_sums"](pi_ref, ids, mask)

        def compute_gradient_stats(model, pairs, label):
            """Compute gradient norms for mini-batches and full batch."""
            model = copy.deepcopy(model).train()
            winners = torch.tensor([w for w, _ in pairs], dtype=torch.long, device=device)
            losers = torch.tensor([l for _, l in pairs], dtype=torch.long, device=device)
            beta = 0.1

            # Full-batch gradient
            model.zero_grad()
            policy_log_probs = h["get_log_prob_sums"](model, ids, mask)
            policy_w = policy_log_probs[winners]
            policy_l = policy_log_probs[losers]
            ref_w = ref_lp[winners]
            ref_l = ref_lp[losers]
            dpo_logits = beta * ((policy_w - ref_w) - (policy_l - ref_l))
            loss = -nn.functional.logsigmoid(dpo_logits).mean()
            loss.backward()

            full_grad = []
            for p in model.parameters():
                if p.grad is not None:
                    full_grad.append(p.grad.data.clone().flatten())
            full_grad = torch.cat(full_grad)
            full_grad_norm = full_grad.norm().item()

            # Mini-batch gradients
            n_pairs = len(pairs)
            indices = list(range(n_pairs))
            batch_grad_norms = []
            batch_cosine_sims = []

            for mb_start in range(0, n_pairs, MINI_BATCH_SIZE):
                mb_end = min(mb_start + MINI_BATCH_SIZE, n_pairs)
                mb_winners = winners[mb_start:mb_end]
                mb_losers = losers[mb_start:mb_end]

                model.zero_grad()
                policy_log_probs = h["get_log_prob_sums"](model, ids, mask)
                pw = policy_log_probs[mb_winners]
                pl = policy_log_probs[mb_losers]
                rw = ref_lp[mb_winners]
                rl = ref_lp[mb_losers]
                logits = beta * ((pw - rw) - (pl - rl))
                mb_loss = -nn.functional.logsigmoid(logits).mean()
                mb_loss.backward()

                mb_grad = []
                for p in model.parameters():
                    if p.grad is not None:
                        mb_grad.append(p.grad.data.clone().flatten())
                mb_grad = torch.cat(mb_grad)

                batch_grad_norms.append(mb_grad.norm().item())
                cos_sim = torch.nn.functional.cosine_similarity(
                    mb_grad.unsqueeze(0), full_grad.unsqueeze(0)
                ).item()
                batch_cosine_sims.append(cos_sim)

            del model
            return {
                "full_grad_norm": full_grad_norm,
                "batch_grad_norms": batch_grad_norms,
                "grad_norm_variance": float(np.var(batch_grad_norms)),
                "grad_norm_mean": float(np.mean(batch_grad_norms)),
                "cosine_sim_mean": float(np.mean(batch_cosine_sims)),
                "cosine_sim_std": float(np.std(batch_cosine_sims)),
            }

        dpo_stats = compute_gradient_stats(pi_ref, labeled_pairs, "DPO")
        ggdpo_stats = compute_gradient_stats(pi_ref, ggdpo_graph, "GGDPO")

        results["dpo"].append(dpo_stats)
        results["ggdpo"].append(ggdpo_stats)

        del pi_ref
        torch.cuda.empty_cache()

        if (run_idx + 1) % 5 == 0:
            print(f"Run {run_idx+1}/{NUM_RUNS}: DPO grad_var={dpo_stats['grad_norm_variance']:.6f} "
                  f"GGDPO grad_var={ggdpo_stats['grad_norm_variance']:.6f} "
                  f"DPO cos_sim={dpo_stats['cosine_sim_mean']:.3f} "
                  f"GGDPO cos_sim={ggdpo_stats['cosine_sim_mean']:.3f}")

    # Aggregate results
    summary = {
        "N": N, "K": K, "max_pairs": MAX_PAIRS, "num_runs": NUM_RUNS,
        "dpo_grad_norm_variance_mean": float(np.mean([r["grad_norm_variance"] for r in results["dpo"]])),
        "dpo_grad_norm_variance_std": float(np.std([r["grad_norm_variance"] for r in results["dpo"]])),
        "ggdpo_grad_norm_variance_mean": float(np.mean([r["grad_norm_variance"] for r in results["ggdpo"]])),
        "ggdpo_grad_norm_variance_std": float(np.std([r["grad_norm_variance"] for r in results["ggdpo"]])),
        "dpo_cosine_sim_mean": float(np.mean([r["cosine_sim_mean"] for r in results["dpo"]])),
        "ggdpo_cosine_sim_mean": float(np.mean([r["cosine_sim_mean"] for r in results["ggdpo"]])),
        "variance_reduction_ratio": float(
            np.mean([r["grad_norm_variance"] for r in results["dpo"]]) /
            max(np.mean([r["grad_norm_variance"] for r in results["ggdpo"]]), 1e-12)
        ),
    }

    with open(os.path.join(RESULTS_DIR, "exp6_gradient_variance.json"), "w") as f:
        json.dump({"summary": summary, "raw": {
            "dpo": [{k: v for k, v in r.items() if k != "batch_grad_norms"} for r in results["dpo"]],
            "ggdpo": [{k: v for k, v in r.items() if k != "batch_grad_norms"} for r in results["ggdpo"]],
        }}, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    dpo_vars = [r["grad_norm_variance"] for r in results["dpo"]]
    ggdpo_vars = [r["grad_norm_variance"] for r in results["ggdpo"]]
    ax1.boxplot([dpo_vars, ggdpo_vars], labels=["DPO", "GGDPO"])
    ax1.set_ylabel("Gradient Norm Variance")
    ax1.set_title(f"Gradient Variance (N={N}, K={K})")
    ax1.grid(True, alpha=0.3, axis='y')

    dpo_cos = [r["cosine_sim_mean"] for r in results["dpo"]]
    ggdpo_cos = [r["cosine_sim_mean"] for r in results["ggdpo"]]
    ax2.boxplot([dpo_cos, ggdpo_cos], labels=["DPO", "GGDPO"])
    ax2.set_ylabel("Cosine Similarity to Full Gradient")
    ax2.set_title(f"Gradient Direction Consistency (N={N}, K={K})")
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp6_gradient_variance.png"), dpi=150, bbox_inches="tight")
    plt.close()

    results_vol.commit()
    print("\n=== Experiment 6 Complete ===")
    print(json.dumps(summary, indent=2))


# ============================================================
# Experiment 7: Noisy Oracle Denoising
# ============================================================

@app.function(**COMMON_KWARGS)
def exp7_noisy_oracle():
    """
    Show GGDPO's BT fitting acts as built-in denoiser for noisy preferences.
    GPT-2, N=15, noise levels p in {0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3}.
    K=2N=30 noisy oracle pairs.
    Compare DPO on K noisy pairs vs GGDPO on BT-expanded pairs from K noisy pairs.
    Metric: agreement with TRUE noiseless ranking.
    """
    import copy
    import random
    import string
    import json
    import os
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    h = _ggdpo_helpers()
    BASE_SEED = 42

    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    N = 15
    K = 2 * N  # 30
    MAX_PAIRS = N * (N - 1) // 2  # 105
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    NUM_RUNS = 20
    EPOCHS = 200

    results = []

    for noise in noise_levels:
        print(f"\n=== Noise level p={noise} ===")

        run_data = {"dpo_agreement": [], "ggdpo_agreement": [],
                    "bt_accuracy": []}

        for run_idx in range(NUM_RUNS):
            seed = BASE_SEED + run_idx + int(noise * 1000)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            prompt = "".join(random.choices(string.ascii_letters + string.digits, k=20))

            pi_ref = GPT2LMHeadModel.from_pretrained(model_name).to(device)
            pi_ref.config.use_cache = False
            pi_ref.eval()
            pi_dpo = copy.deepcopy(pi_ref).train()
            pi_ggdpo = copy.deepcopy(pi_ref).train()

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            completions = []
            with torch.no_grad():
                for _ in range(N):
                    out = pi_ref.generate(
                        **{kk: v.clone() for kk, v in inputs.items()},
                        max_length=20, do_sample=True, top_k=50,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    completions.append(tokenizer.decode(out[0], skip_special_tokens=True))

            perm = np.random.permutation(N)
            oracle_scores = np.empty(N)
            for rank, idx in enumerate(perm):
                oracle_scores[idx] = rank + 1

            sampled_pairs = h["sample_pairs_random"](N, K)
            # Label with noise
            noisy_labeled = h["label_pairs"](sampled_pairs, oracle_scores, noise_prob=noise)

            # BT fitting on noisy pairs
            bt_scores = h["fit_bradley_terry"](N, noisy_labeled)
            bt_agree, bt_total = h["count_pair_agreements"](bt_scores, oracle_scores)
            run_data["bt_accuracy"].append(bt_agree / bt_total)

            ggdpo_graph = h["construct_full_graph"](bt_scores)

            tokens = tokenizer(completions, return_tensors="pt", padding=True, truncation=True)
            ids = tokens["input_ids"].to(device)
            mask = tokens["attention_mask"].to(device)

            with torch.no_grad():
                ref_lp = h["get_log_prob_sums"](pi_ref, ids, mask)

            # DPO trains on noisy labeled pairs directly
            dpo_lps = h["train_dpo"](pi_dpo, ref_lp, ids, mask, noisy_labeled,
                                     epochs=EPOCHS, device=device)
            # GGDPO trains on BT-expanded pairs (denoised)
            ggdpo_lps = h["train_dpo"](pi_ggdpo, ref_lp, ids, mask, ggdpo_graph,
                                       epochs=EPOCHS, device=device)

            # Agreement with TRUE noiseless ranking
            dpo_m = h["compute_ranking_metrics"](np.array(dpo_lps[-1]), oracle_scores)
            ggdpo_m = h["compute_ranking_metrics"](np.array(ggdpo_lps[-1]), oracle_scores)

            run_data["dpo_agreement"].append(dpo_m["pairwise_agreement"])
            run_data["ggdpo_agreement"].append(ggdpo_m["pairwise_agreement"])

            del pi_ref, pi_dpo, pi_ggdpo
            torch.cuda.empty_cache()

            if (run_idx + 1) % 5 == 0:
                print(f"  Run {run_idx+1}/{NUM_RUNS}: DPO={dpo_m['pairwise_agreement']:.3f} "
                      f"GGDPO={ggdpo_m['pairwise_agreement']:.3f} BT_acc={bt_agree/bt_total:.3f}")

        result = {
            "noise": noise, "N": N, "K": K,
            "dpo_agreement_mean": float(np.mean(run_data["dpo_agreement"])),
            "dpo_agreement_std": float(np.std(run_data["dpo_agreement"])),
            "ggdpo_agreement_mean": float(np.mean(run_data["ggdpo_agreement"])),
            "ggdpo_agreement_std": float(np.std(run_data["ggdpo_agreement"])),
            "bt_accuracy_mean": float(np.mean(run_data["bt_accuracy"])),
            "bt_accuracy_std": float(np.std(run_data["bt_accuracy"])),
            "improvement": float(np.mean(run_data["ggdpo_agreement"]) - np.mean(run_data["dpo_agreement"])),
        }
        results.append(result)
        print(f"  p={noise}: DPO={result['dpo_agreement_mean']:.3f}+-{result['dpo_agreement_std']:.3f} "
              f"GGDPO={result['ggdpo_agreement_mean']:.3f}+-{result['ggdpo_agreement_std']:.3f} "
              f"Improvement={result['improvement']:+.3f}")

    with open(os.path.join(RESULTS_DIR, "exp7_noisy_oracle.json"), "w") as f:
        json.dump(results, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    noises = [r["noise"] for r in results]
    dpo_means = [r["dpo_agreement_mean"] for r in results]
    dpo_stds = [r["dpo_agreement_std"] for r in results]
    ggdpo_means = [r["ggdpo_agreement_mean"] for r in results]
    ggdpo_stds = [r["ggdpo_agreement_std"] for r in results]
    bt_means = [r["bt_accuracy_mean"] for r in results]

    ax1.errorbar(noises, dpo_means, yerr=dpo_stds, marker='o', capsize=4, label='DPO (noisy pairs)')
    ax1.errorbar(noises, ggdpo_means, yerr=ggdpo_stds, marker='x', capsize=4, label='GGDPO (BT-denoised)')
    ax1.plot(noises, bt_means, marker='s', linestyle='--', alpha=0.5, label='BT Estimation Accuracy')
    ax1.set_xlabel("Noise Level (p)")
    ax1.set_ylabel("Oracle Agreement (with true ranking)")
    ax1.set_title("Noisy Oracle: GGDPO Denoising Effect")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    improvements = [r["improvement"] for r in results]
    bar_colors = ['green' if v > 0 else 'red' for v in improvements]
    ax2.bar([f"{n:.2f}" for n in noises], improvements, color=bar_colors, alpha=0.7)
    ax2.set_xlabel("Noise Level (p)")
    ax2.set_ylabel("GGDPO - DPO Agreement")
    ax2.set_title("GGDPO Improvement at Each Noise Level")
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp7_noisy_oracle.png"), dpi=150, bbox_inches="tight")
    plt.close()

    results_vol.commit()
    print("\n=== Experiment 7 Complete ===")
    print(json.dumps(results, indent=2))


# ============================================================
# Experiment 8: Held-Out Pair Prediction (Cross-Validation)
# ============================================================

@app.function(**COMMON_KWARGS)
def exp8_heldout_prediction():
    """
    Directly test whether GGDPO's BT-inferred preferences are correct on unseen pairs.
    GPT-2, N=20, all 190 oracle pairs known (ground truth).
    Split: K train pairs, (190-K) held-out test pairs.
    K sweep: {19, 30, 40, 60, 95}.
    Evaluate: agreement with held-out (190-K) test pairs.
    """
    import copy
    import random
    import string
    import json
    import os
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    h = _ggdpo_helpers()
    BASE_SEED = 42

    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    N = 20
    MAX_PAIRS = N * (N - 1) // 2  # 190
    K_values = [19, 30, 40, 60, 95]
    NUM_RUNS = 20
    EPOCHS = 200

    results = []

    for K in K_values:
        print(f"\n=== K={K} train pairs, {MAX_PAIRS - K} held-out test pairs ===")

        run_data = {
            "bt_heldout_accuracy": [],
            "dpo_heldout_agreement": [],
            "ggdpo_heldout_agreement": [],
            "dpo_full_agreement": [],
            "ggdpo_full_agreement": [],
        }

        for run_idx in range(NUM_RUNS):
            seed = BASE_SEED + run_idx + K * 100
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            prompt = "".join(random.choices(string.ascii_letters + string.digits, k=20))

            pi_ref = GPT2LMHeadModel.from_pretrained(model_name).to(device)
            pi_ref.config.use_cache = False
            pi_ref.eval()
            pi_dpo = copy.deepcopy(pi_ref).train()
            pi_ggdpo = copy.deepcopy(pi_ref).train()

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            completions = []
            with torch.no_grad():
                for _ in range(N):
                    out = pi_ref.generate(
                        **{kk: v.clone() for kk, v in inputs.items()},
                        max_length=20, do_sample=True, top_k=50,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    completions.append(tokenizer.decode(out[0], skip_special_tokens=True))

            perm = np.random.permutation(N)
            oracle_scores = np.empty(N)
            for rank, idx in enumerate(perm):
                oracle_scores[idx] = rank + 1

            # Generate ALL pairs and label them
            all_pairs = []
            for i in range(N):
                for j in range(i + 1, N):
                    all_pairs.append((i, j))
            random.shuffle(all_pairs)

            all_labeled = h["label_pairs"](all_pairs, oracle_scores)

            # Split into train and held-out
            train_pairs = all_pairs[:K]
            train_labeled = all_labeled[:K]
            heldout_pairs = all_pairs[K:]
            heldout_labeled = all_labeled[K:]

            # BT estimation from train pairs
            bt_scores = h["fit_bradley_terry"](N, train_labeled)
            ggdpo_graph = h["construct_full_graph"](bt_scores)

            # BT accuracy on held-out pairs
            bt_heldout_agree = 0
            for (w, l) in heldout_labeled:
                if bt_scores[w] > bt_scores[l]:
                    bt_heldout_agree += 1
            bt_heldout_accuracy = bt_heldout_agree / len(heldout_labeled) if heldout_labeled else 0
            run_data["bt_heldout_accuracy"].append(bt_heldout_accuracy)

            tokens = tokenizer(completions, return_tensors="pt", padding=True, truncation=True)
            ids = tokens["input_ids"].to(device)
            mask = tokens["attention_mask"].to(device)

            with torch.no_grad():
                ref_lp = h["get_log_prob_sums"](pi_ref, ids, mask)

            # DPO trains on K train pairs only
            dpo_lps = h["train_dpo"](pi_dpo, ref_lp, ids, mask, train_labeled,
                                     epochs=EPOCHS, device=device)
            # GGDPO trains on BT-expanded full graph
            ggdpo_lps = h["train_dpo"](pi_ggdpo, ref_lp, ids, mask, ggdpo_graph,
                                       epochs=EPOCHS, device=device)

            dpo_final = np.array(dpo_lps[-1])
            ggdpo_final = np.array(ggdpo_lps[-1])

            # Agreement on held-out pairs
            dpo_heldout_agree = 0
            ggdpo_heldout_agree = 0
            for (w, l) in heldout_labeled:
                if dpo_final[w] > dpo_final[l]:
                    dpo_heldout_agree += 1
                if ggdpo_final[w] > ggdpo_final[l]:
                    ggdpo_heldout_agree += 1

            n_heldout = len(heldout_labeled)
            run_data["dpo_heldout_agreement"].append(dpo_heldout_agree / n_heldout if n_heldout > 0 else 0)
            run_data["ggdpo_heldout_agreement"].append(ggdpo_heldout_agree / n_heldout if n_heldout > 0 else 0)

            # Full agreement
            dpo_m = h["compute_ranking_metrics"](dpo_final, oracle_scores)
            ggdpo_m = h["compute_ranking_metrics"](ggdpo_final, oracle_scores)
            run_data["dpo_full_agreement"].append(dpo_m["pairwise_agreement"])
            run_data["ggdpo_full_agreement"].append(ggdpo_m["pairwise_agreement"])

            del pi_ref, pi_dpo, pi_ggdpo
            torch.cuda.empty_cache()

            if (run_idx + 1) % 5 == 0:
                print(f"  Run {run_idx+1}/{NUM_RUNS}: "
                      f"DPO heldout={dpo_heldout_agree/n_heldout:.3f} "
                      f"GGDPO heldout={ggdpo_heldout_agree/n_heldout:.3f} "
                      f"BT heldout={bt_heldout_accuracy:.3f}")

        result = {
            "K": K, "N": N, "heldout_size": MAX_PAIRS - K,
            "bt_heldout_accuracy_mean": float(np.mean(run_data["bt_heldout_accuracy"])),
            "bt_heldout_accuracy_std": float(np.std(run_data["bt_heldout_accuracy"])),
            "dpo_heldout_agreement_mean": float(np.mean(run_data["dpo_heldout_agreement"])),
            "dpo_heldout_agreement_std": float(np.std(run_data["dpo_heldout_agreement"])),
            "ggdpo_heldout_agreement_mean": float(np.mean(run_data["ggdpo_heldout_agreement"])),
            "ggdpo_heldout_agreement_std": float(np.std(run_data["ggdpo_heldout_agreement"])),
            "dpo_full_agreement_mean": float(np.mean(run_data["dpo_full_agreement"])),
            "ggdpo_full_agreement_mean": float(np.mean(run_data["ggdpo_full_agreement"])),
            "heldout_improvement": float(np.mean(run_data["ggdpo_heldout_agreement"]) - np.mean(run_data["dpo_heldout_agreement"])),
        }
        results.append(result)
        print(f"  K={K}: DPO heldout={result['dpo_heldout_agreement_mean']:.3f} "
              f"GGDPO heldout={result['ggdpo_heldout_agreement_mean']:.3f} "
              f"Improvement={result['heldout_improvement']:+.3f}")

    with open(os.path.join(RESULTS_DIR, "exp8_heldout_prediction.json"), "w") as f:
        json.dump(results, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    Ks = [r["K"] for r in results]
    dpo_h = [r["dpo_heldout_agreement_mean"] for r in results]
    dpo_hs = [r["dpo_heldout_agreement_std"] for r in results]
    ggdpo_h = [r["ggdpo_heldout_agreement_mean"] for r in results]
    ggdpo_hs = [r["ggdpo_heldout_agreement_std"] for r in results]
    bt_h = [r["bt_heldout_accuracy_mean"] for r in results]
    bt_hs = [r["bt_heldout_accuracy_std"] for r in results]

    ax1.errorbar(Ks, dpo_h, yerr=dpo_hs, marker='o', capsize=4, label='DPO (K pairs)')
    ax1.errorbar(Ks, ggdpo_h, yerr=ggdpo_hs, marker='x', capsize=4, label='GGDPO (BT-expanded)')
    ax1.errorbar(Ks, bt_h, yerr=bt_hs, marker='s', capsize=4, linestyle='--', alpha=0.5, label='BT Estimation Only')
    ax1.set_xlabel("K (train pairs)")
    ax1.set_ylabel("Held-Out Pair Agreement")
    ax1.set_title(f"Held-Out Prediction (N={N}, {MAX_PAIRS} total pairs)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    improvements = [r["heldout_improvement"] for r in results]
    bar_colors = ['green' if v > 0 else 'red' for v in improvements]
    ax2.bar([str(k) for k in Ks], improvements, color=bar_colors, alpha=0.7)
    ax2.set_xlabel("K (train pairs)")
    ax2.set_ylabel("GGDPO - DPO Held-Out Agreement")
    ax2.set_title("GGDPO Improvement on Held-Out Pairs")
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp8_heldout_prediction.png"), dpi=150, bbox_inches="tight")
    plt.close()

    results_vol.commit()
    print("\n=== Experiment 8 Complete ===")
    print(json.dumps(results, indent=2))
