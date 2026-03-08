"""
GGDPO Experiments - Modal Runner
Runs all proposed experiments on Modal with H200 GPU.

Usage:
    modal deploy experiments_modal.py
    modal run experiments_modal.py::exp1_synthetic_sweep
    modal run experiments_modal.py::exp2_scaling_n
    modal run experiments_modal.py::exp3_ablations
    modal run experiments_modal.py::exp4_reward_model_scaled
    modal run experiments_modal.py::exp5_ultrafeedback
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
    import copy
    import random
    import string
    import numpy as np
    import torch
    import torch.nn as nn
    from scipy.stats import kendalltau, spearmanr

    def sample_pairs_random(n_completions, n_pairs):
        pairs = []
        existing = set()
        max_possible = n_completions * (n_completions - 1) // 2
        n_pairs = min(n_pairs, max_possible)
        while len(pairs) < n_pairs:
            idx = np.random.choice(n_completions, 2, replace=False)
            idx = tuple(sorted(idx))
            if idx not in existing:
                existing.add(idx)
                pairs.append(idx)
        return pairs

    def sample_pairs_chain(n_completions, n_pairs):
        """Chain: compare consecutive items, then random extras."""
        pairs = []
        existing = set()
        perm = np.random.permutation(n_completions)
        for k in range(min(n_completions - 1, n_pairs)):
            pair = tuple(sorted((int(perm[k]), int(perm[k + 1]))))
            if pair not in existing:
                existing.add(pair)
                pairs.append(pair)
        while len(pairs) < n_pairs:
            idx = np.random.choice(n_completions, 2, replace=False)
            idx = tuple(sorted(idx))
            if idx not in existing:
                existing.add(idx)
                pairs.append(idx)
        return pairs

    def sample_pairs_star(n_completions, n_pairs):
        """Star: one hub item compared to all others, then random extras."""
        pairs = []
        existing = set()
        hub = np.random.randint(n_completions)
        for i in range(n_completions):
            if i != hub and len(pairs) < n_pairs:
                pair = tuple(sorted((hub, i)))
                if pair not in existing:
                    existing.add(pair)
                    pairs.append(pair)
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
        """Simple win-rate estimation (no BT model)."""
        wins = np.zeros(n_completions)
        counts = np.zeros(n_completions)
        for w, l in labeled_pairs:
            wins[w] += 1
            counts[w] += 1
            counts[l] += 1
        scores = np.where(counts > 0, wins / counts, 0.5)
        scores = (scores - scores.mean()) / (scores.std() + 1e-8)
        return scores

    def fit_transitive_closure(n_completions, labeled_pairs):
        """Transitive closure: infer from direct comparisons + transitivity."""
        # Build adjacency: adj[i][j] = True means i > j observed
        adj = np.zeros((n_completions, n_completions), dtype=bool)
        for w, l in labeled_pairs:
            adj[w][l] = True
        # Warshall's algorithm for transitive closure
        for k in range(n_completions):
            for i in range(n_completions):
                for j in range(n_completions):
                    if adj[i][k] and adj[k][j]:
                        adj[i][j] = True
        # Scores = number of items beaten
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

    def construct_weighted_graph(scores):
        """Construct full graph with confidence weights based on score gaps."""
        n = len(scores)
        pairs = []
        weights = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                gap = abs(scores[i] - scores[j])
                weight = float(1.0 / (1.0 + np.exp(-gap)))  # sigmoid of gap
                if scores[i] > scores[j]:
                    pairs.append((i, j))
                else:
                    pairs.append((j, i))
                weights.append(weight)
        return pairs, weights

    def get_log_prob_sums(model, input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        mask = attention_mask[:, 1:]
        return (token_log_probs * mask).sum(dim=-1)

    def train_dpo(policy_model, ref_log_probs_tensor, input_ids_all, attention_mask_all,
                  pairs, epochs=300, lr=1e-5, beta=0.1, weights=None, device="cuda",
                  model_is_bf16=False):
        optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr)
        winners = torch.tensor([w for w, _ in pairs], dtype=torch.long, device=device)
        losers = torch.tensor([l for _, l in pairs], dtype=torch.long, device=device)
        if weights is not None:
            w_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        else:
            w_tensor = None

        log_probs_over_time = []
        # BFloat16 models don't support GradScaler
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
                logits = beta * ((policy_w - ref_w) - (policy_l - ref_l))
                losses = -nn.functional.logsigmoid(logits)
                if w_tensor is not None:
                    losses = losses * w_tensor
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
        """Compute pairwise agreement, Kendall tau, Spearman rho."""
        agree, total = count_pair_agreements(model_scores, oracle_scores)
        frac = agree / total if total > 0 else 0.0
        tau, _ = kendalltau(model_scores, oracle_scores)
        rho, _ = spearmanr(model_scores, oracle_scores)
        return {"pairwise_agreement": frac, "kendall_tau": tau, "spearman_rho": rho}

    return {
        "sample_pairs_random": sample_pairs_random,
        "sample_pairs_chain": sample_pairs_chain,
        "sample_pairs_star": sample_pairs_star,
        "label_pairs": label_pairs,
        "fit_bradley_terry": fit_bradley_terry,
        "fit_win_rate": fit_win_rate,
        "fit_transitive_closure": fit_transitive_closure,
        "construct_full_graph": construct_full_graph,
        "construct_weighted_graph": construct_weighted_graph,
        "get_log_prob_sums": get_log_prob_sums,
        "train_dpo": train_dpo,
        "count_pair_agreements": count_pair_agreements,
        "compute_ranking_metrics": compute_ranking_metrics,
    }


# ============================================================
# Experiment 1: Synthetic N/K Sweep + Noise + Graph Structures
# ============================================================

@app.function(**COMMON_KWARGS)
def exp1_synthetic_sweep():
    """
    Strengthened synthetic experiment with GPT-2.
    Sweeps: N, K/N ratio, noise levels, graph sampling structures.
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

    # Sweep configurations
    N_values = [10, 20, 50]
    K_ratios = [0.5, 1.0, 2.0, 3.0]
    noise_levels = [0.0, 0.1, 0.2]
    NUM_RUNS = 5
    EPOCHS = 200

    results = []
    total_configs = len(N_values) * len(K_ratios) * len(noise_levels)
    config_idx = 0

    for N in N_values:
        for k_ratio in K_ratios:
            K = max(N - 1, int(N * k_ratio))  # at least N-1 for connectivity
            max_possible = N * (N - 1) // 2
            K = min(K, max_possible)

            for noise in noise_levels:
                config_idx += 1
                print(f"\n=== Config {config_idx}/{total_configs}: N={N}, K={K} (ratio={k_ratio}), noise={noise} ===")

                run_results = {"dpo_agreement": [], "ggdpo_agreement": [],
                               "dpo_kendall": [], "ggdpo_kendall": [],
                               "dpo_variance": [], "ggdpo_variance": [],
                               "bt_accuracy": []}

                for run_idx in range(NUM_RUNS):
                    seed = BASE_SEED + run_idx + config_idx * 100
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    random.seed(seed)

                    prompt = "".join(random.choices(string.ascii_letters + string.digits, k=20))

                    # Fresh models
                    pi_ref = GPT2LMHeadModel.from_pretrained(model_name).to(device)
                    pi_ref.config.use_cache = False
                    pi_ref.eval()

                    pi_dpo = copy.deepcopy(pi_ref).train()
                    pi_ggdpo = copy.deepcopy(pi_ref).train()

                    # Generate completions
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    completions = []
                    with torch.no_grad():
                        for _ in range(N):
                            out = pi_ref.generate(
                                **{k: v.clone() for k, v in inputs.items()},
                                max_length=20, do_sample=True, top_k=50,
                                pad_token_id=tokenizer.eos_token_id
                            )
                            completions.append(tokenizer.decode(out[0], skip_special_tokens=True))

                    # Oracle ranking
                    perm = np.random.permutation(N)
                    oracle_scores = np.empty(N)
                    for rank, idx in enumerate(perm):
                        oracle_scores[idx] = rank + 1

                    # Sample and label pairs
                    sampled_pairs = h["sample_pairs_random"](N, K)
                    labeled_pairs = h["label_pairs"](sampled_pairs, oracle_scores, noise_prob=noise)

                    # BT estimation
                    bt_scores = h["fit_bradley_terry"](N, labeled_pairs)
                    bt_agree, bt_total = h["count_pair_agreements"](bt_scores, oracle_scores)
                    run_results["bt_accuracy"].append(bt_agree / bt_total)

                    # Full graph from BT
                    full_graph = h["construct_full_graph"](bt_scores)

                    # Tokenize
                    tokens = tokenizer(completions, return_tensors="pt", padding=True, truncation=True)
                    ids = tokens["input_ids"].to(device)
                    mask = tokens["attention_mask"].to(device)

                    with torch.no_grad():
                        ref_lp = h["get_log_prob_sums"](pi_ref, ids, mask)

                    # Train DPO
                    dpo_lps = h["train_dpo"](pi_dpo, ref_lp, ids, mask, labeled_pairs,
                                             epochs=EPOCHS, device=device)
                    # Train GGDPO
                    ggdpo_lps = h["train_dpo"](pi_ggdpo, ref_lp, ids, mask, full_graph,
                                               epochs=EPOCHS, device=device)

                    # Final metrics
                    dpo_final = np.array(dpo_lps[-1])
                    ggdpo_final = np.array(ggdpo_lps[-1])

                    dpo_metrics = h["compute_ranking_metrics"](dpo_final, oracle_scores)
                    ggdpo_metrics = h["compute_ranking_metrics"](ggdpo_final, oracle_scores)

                    run_results["dpo_agreement"].append(dpo_metrics["pairwise_agreement"])
                    run_results["ggdpo_agreement"].append(ggdpo_metrics["pairwise_agreement"])
                    run_results["dpo_kendall"].append(dpo_metrics["kendall_tau"])
                    run_results["ggdpo_kendall"].append(ggdpo_metrics["kendall_tau"])

                    # Variance of updates post-convergence
                    if EPOCHS > 50:
                        dpo_fracs = []
                        ggdpo_fracs = []
                        for e in range(50, EPOCHS):
                            da, dt = h["count_pair_agreements"](np.array(dpo_lps[e]), oracle_scores)
                            ga, _ = h["count_pair_agreements"](np.array(ggdpo_lps[e]), oracle_scores)
                            dpo_fracs.append(da / dt)
                            ggdpo_fracs.append(ga / dt)
                        run_results["dpo_variance"].append(float(np.var(np.diff(dpo_fracs))))
                        run_results["ggdpo_variance"].append(float(np.var(np.diff(ggdpo_fracs))))

                    del pi_ref, pi_dpo, pi_ggdpo
                    torch.cuda.empty_cache()

                    print(f"  Run {run_idx+1}/{NUM_RUNS}: DPO={dpo_metrics['pairwise_agreement']:.3f} GGDPO={ggdpo_metrics['pairwise_agreement']:.3f} BT_acc={bt_agree/bt_total:.3f}")

                # Average over runs
                result = {
                    "N": N, "K": K, "k_ratio": k_ratio, "noise": noise,
                    "dpo_agreement_mean": float(np.mean(run_results["dpo_agreement"])),
                    "dpo_agreement_std": float(np.std(run_results["dpo_agreement"])),
                    "ggdpo_agreement_mean": float(np.mean(run_results["ggdpo_agreement"])),
                    "ggdpo_agreement_std": float(np.std(run_results["ggdpo_agreement"])),
                    "dpo_kendall_mean": float(np.mean(run_results["dpo_kendall"])),
                    "ggdpo_kendall_mean": float(np.mean(run_results["ggdpo_kendall"])),
                    "bt_accuracy_mean": float(np.mean(run_results["bt_accuracy"])),
                    "dpo_variance_mean": float(np.mean(run_results["dpo_variance"])) if run_results["dpo_variance"] else 0,
                    "ggdpo_variance_mean": float(np.mean(run_results["ggdpo_variance"])) if run_results["ggdpo_variance"] else 0,
                }
                results.append(result)
                print(f"  Avg: DPO={result['dpo_agreement_mean']:.3f} GGDPO={result['ggdpo_agreement_mean']:.3f}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/exp1_synthetic_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    results_vol.commit()

    # Generate plots
    # Plot 1: N/K ratio vs agreement (no noise)
    fig, axes = plt.subplots(1, len(N_values), figsize=(5 * len(N_values), 4), sharey=True)
    if len(N_values) == 1:
        axes = [axes]
    for ax, N in zip(axes, N_values):
        no_noise = [r for r in results if r["N"] == N and r["noise"] == 0.0]
        ratios = [r["k_ratio"] for r in no_noise]
        dpo_means = [r["dpo_agreement_mean"] for r in no_noise]
        ggdpo_means = [r["ggdpo_agreement_mean"] for r in no_noise]
        ax.plot(ratios, dpo_means, "o-", label="DPO", color="tab:blue")
        ax.plot(ratios, ggdpo_means, "x-", label="GGDPO", color="tab:orange")
        ax.set_xlabel("K/N ratio")
        ax.set_ylabel("Pairwise Agreement")
        ax.set_title(f"N={N}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle("Sample Efficiency: DPO vs GGDPO (no noise)")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/exp1_nk_sweep.png", dpi=150)
    plt.close()

    # Plot 2: Noise robustness
    fig, axes = plt.subplots(1, len(N_values), figsize=(5 * len(N_values), 4), sharey=True)
    if len(N_values) == 1:
        axes = [axes]
    for ax, N in zip(axes, N_values):
        for noise in noise_levels:
            subset = [r for r in results if r["N"] == N and r["noise"] == noise]
            ratios = [r["k_ratio"] for r in subset]
            ggdpo_means = [r["ggdpo_agreement_mean"] for r in subset]
            ax.plot(ratios, ggdpo_means, "x-", label=f"GGDPO noise={noise}")
        ax.set_xlabel("K/N ratio")
        ax.set_ylabel("Pairwise Agreement")
        ax.set_title(f"N={N}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle("GGDPO Noise Robustness")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/exp1_noise_robustness.png", dpi=150)
    plt.close()

    results_vol.commit()
    print("\n=== Experiment 1 Complete ===")
    print(json.dumps(results, indent=2))
    return results


# ============================================================
# Experiment 2: Scaling N (the "killer chart")
# ============================================================

@app.function(**COMMON_KWARGS)
def exp2_scaling_n():
    """
    Shows GGDPO advantage grows as n increases.
    The key chart for the paper.
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

    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    h = _ggdpo_helpers()
    BASE_SEED = 44

    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    N_values = [4, 8, 16, 32, 64]
    NUM_RUNS = 10
    EPOCHS = 200

    results = []

    for N in N_values:
        K = 2 * N  # Linear in N
        max_possible = N * (N - 1) // 2
        K = min(K, max_possible)
        total_possible_pairs = max_possible

        print(f"\n=== N={N}, K={K}, total_possible={total_possible_pairs} ===")

        run_data = {"dpo": [], "ggdpo": [], "full_dpo": [], "bt_acc": [],
                    "dpo_kendall": [], "ggdpo_kendall": [], "full_dpo_kendall": [],
                    "dpo_over_time": [], "ggdpo_over_time": [], "full_dpo_over_time": []}

        for run_idx in range(NUM_RUNS):
            seed = BASE_SEED + run_idx + N * 1000
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

            # Generate completions
            completions = []
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                for _ in range(N):
                    out = pi_ref.generate(
                        **{k: v.clone() for k, v in inputs.items()},
                        max_length=20, do_sample=True, top_k=50,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    completions.append(tokenizer.decode(out[0], skip_special_tokens=True))

            # Oracle
            perm = np.random.permutation(N)
            oracle_scores = np.empty(N)
            for rank, idx in enumerate(perm):
                oracle_scores[idx] = rank + 1

            # Sample K pairs
            sampled_pairs = h["sample_pairs_random"](N, K)
            labeled_pairs = h["label_pairs"](sampled_pairs, oracle_scores)

            # Full oracle pairs (upper bound)
            all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
            full_labeled = h["label_pairs"](all_pairs, oracle_scores)

            # BT estimation
            bt_scores = h["fit_bradley_terry"](N, labeled_pairs)
            bt_agree, bt_total = h["count_pair_agreements"](bt_scores, oracle_scores)
            run_data["bt_acc"].append(bt_agree / bt_total)

            full_graph = h["construct_full_graph"](bt_scores)

            # Tokenize
            tokens = tokenizer(completions, return_tensors="pt", padding=True, truncation=True)
            ids = tokens["input_ids"].to(device)
            mask = tokens["attention_mask"].to(device)
            with torch.no_grad():
                ref_lp = h["get_log_prob_sums"](pi_ref, ids, mask)

            # Train all three
            dpo_lps = h["train_dpo"](pi_dpo, ref_lp, ids, mask, labeled_pairs, epochs=EPOCHS, device=device)
            ggdpo_lps = h["train_dpo"](pi_ggdpo, ref_lp, ids, mask, full_graph, epochs=EPOCHS, device=device)
            full_lps = h["train_dpo"](pi_full, ref_lp, ids, mask, full_labeled, epochs=EPOCHS, device=device)

            # Final metrics
            dpo_m = h["compute_ranking_metrics"](np.array(dpo_lps[-1]), oracle_scores)
            ggdpo_m = h["compute_ranking_metrics"](np.array(ggdpo_lps[-1]), oracle_scores)
            full_m = h["compute_ranking_metrics"](np.array(full_lps[-1]), oracle_scores)

            run_data["dpo"].append(dpo_m["pairwise_agreement"])
            run_data["ggdpo"].append(ggdpo_m["pairwise_agreement"])
            run_data["full_dpo"].append(full_m["pairwise_agreement"])
            run_data["dpo_kendall"].append(dpo_m["kendall_tau"])
            run_data["ggdpo_kendall"].append(ggdpo_m["kendall_tau"])
            run_data["full_dpo_kendall"].append(full_m["kendall_tau"])

            # Agreement over time for averaging
            dpo_time = []
            ggdpo_time = []
            full_time = []
            for e in range(EPOCHS):
                da, dt = h["count_pair_agreements"](np.array(dpo_lps[e]), oracle_scores)
                ga, _ = h["count_pair_agreements"](np.array(ggdpo_lps[e]), oracle_scores)
                fa, _ = h["count_pair_agreements"](np.array(full_lps[e]), oracle_scores)
                dpo_time.append(da / dt)
                ggdpo_time.append(ga / dt)
                full_time.append(fa / dt)
            run_data["dpo_over_time"].append(dpo_time)
            run_data["ggdpo_over_time"].append(ggdpo_time)
            run_data["full_dpo_over_time"].append(full_time)

            del pi_ref, pi_dpo, pi_ggdpo, pi_full
            torch.cuda.empty_cache()

            print(f"  Run {run_idx+1}: DPO={dpo_m['pairwise_agreement']:.3f} GGDPO={ggdpo_m['pairwise_agreement']:.3f} Full={full_m['pairwise_agreement']:.3f}")

        result = {
            "N": N, "K": K, "total_pairs": total_possible_pairs,
            "dpo_mean": float(np.mean(run_data["dpo"])),
            "dpo_std": float(np.std(run_data["dpo"])),
            "ggdpo_mean": float(np.mean(run_data["ggdpo"])),
            "ggdpo_std": float(np.std(run_data["ggdpo"])),
            "full_dpo_mean": float(np.mean(run_data["full_dpo"])),
            "full_dpo_std": float(np.std(run_data["full_dpo"])),
            "bt_accuracy_mean": float(np.mean(run_data["bt_acc"])),
            "dpo_kendall_mean": float(np.mean(run_data["dpo_kendall"])),
            "ggdpo_kendall_mean": float(np.mean(run_data["ggdpo_kendall"])),
            "full_dpo_kendall_mean": float(np.mean(run_data["full_dpo_kendall"])),
            "dpo_over_time_avg": np.mean(run_data["dpo_over_time"], axis=0).tolist(),
            "ggdpo_over_time_avg": np.mean(run_data["ggdpo_over_time"], axis=0).tolist(),
            "full_dpo_over_time_avg": np.mean(run_data["full_dpo_over_time"], axis=0).tolist(),
        }
        results.append(result)
        print(f"  Avg: DPO={result['dpo_mean']:.3f} GGDPO={result['ggdpo_mean']:.3f} Full={result['full_dpo_mean']:.3f}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/exp2_scaling_n.json", "w") as f:
        json.dump(results, f, indent=2)

    # Killer chart: Final agreement vs N
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ns = [r["N"] for r in results]
    dpo_means = [r["dpo_mean"] for r in results]
    dpo_stds = [r["dpo_std"] for r in results]
    ggdpo_means = [r["ggdpo_mean"] for r in results]
    ggdpo_stds = [r["ggdpo_std"] for r in results]
    full_means = [r["full_dpo_mean"] for r in results]
    full_stds = [r["full_dpo_std"] for r in results]

    ax1.errorbar(ns, dpo_means, yerr=dpo_stds, fmt="o-", label="DPO (K=2N pairs)", capsize=3)
    ax1.errorbar(ns, ggdpo_means, yerr=ggdpo_stds, fmt="x-", label="GGDPO (K=2N -> N(N-1)/2)", capsize=3)
    ax1.errorbar(ns, full_means, yerr=full_stds, fmt="s--", label="Full DPO (all N(N-1)/2 pairs)", capsize=3, alpha=0.7)
    ax1.set_xlabel("N (completions per prompt)")
    ax1.set_ylabel("Final Pairwise Agreement with Oracle")
    ax1.set_title("GGDPO Advantage Grows with N")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gap plot
    gaps = [g - d for g, d in zip(ggdpo_means, dpo_means)]
    ax2.bar(range(len(ns)), gaps, tick_label=[str(n) for n in ns], color="tab:green", alpha=0.7)
    ax2.set_xlabel("N")
    ax2.set_ylabel("GGDPO - DPO Agreement Gap")
    ax2.set_title("GGDPO Improvement Over DPO")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/exp2_scaling_n.png", dpi=150)
    plt.close()

    # Training curves for each N
    fig, axes = plt.subplots(1, len(N_values), figsize=(4 * len(N_values), 4), sharey=True)
    if len(N_values) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        epochs_x = list(range(EPOCHS))
        ax.plot(epochs_x, r["dpo_over_time_avg"], label="DPO", alpha=0.8)
        ax.plot(epochs_x, r["ggdpo_over_time_avg"], label="GGDPO", alpha=0.8)
        ax.plot(epochs_x, r["full_dpo_over_time_avg"], label="Full DPO", alpha=0.5, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_title(f"N={r['N']}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Pairwise Agreement")
    plt.suptitle("Training Curves: DPO vs GGDPO vs Full DPO")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/exp2_training_curves.png", dpi=150)
    plt.close()

    results_vol.commit()
    print("\n=== Experiment 2 (Scaling N) Complete ===")
    return results


# ============================================================
# Experiment 3: Ablations (graph estimation methods, weighting)
# ============================================================

@app.function(**COMMON_KWARGS)
def exp3_ablations():
    """
    Ablation study: BT vs win-rate vs transitive closure,
    confidence-weighted DPO, graph structure sampling methods.
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

    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    h = _ggdpo_helpers()
    BASE_SEED = 44

    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    N = 30
    K = 30
    NUM_RUNS = 10
    EPOCHS = 200

    # Ablation 1: Graph estimation methods
    print("\n=== Ablation 1: Graph Estimation Methods ===")
    methods = {
        "bradley_terry": h["fit_bradley_terry"],
        "win_rate": h["fit_win_rate"],
        "transitive_closure": h["fit_transitive_closure"],
    }

    method_results = {name: {"agreement": [], "kendall": []} for name in methods}
    method_results["dpo_baseline"] = {"agreement": [], "kendall": []}

    for run_idx in range(NUM_RUNS):
        seed = BASE_SEED + run_idx
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        prompt = "".join(random.choices(string.ascii_letters + string.digits, k=20))

        pi_ref = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        pi_ref.config.use_cache = False
        pi_ref.eval()

        perm = np.random.permutation(N)
        oracle_scores = np.empty(N)
        for rank, idx in enumerate(perm):
            oracle_scores[idx] = rank + 1

        sampled_pairs = h["sample_pairs_random"](N, K)
        labeled_pairs = h["label_pairs"](sampled_pairs, oracle_scores)

        # Generate completions
        completions = []
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            for _ in range(N):
                out = pi_ref.generate(
                    **{k: v.clone() for k, v in inputs.items()},
                    max_length=20, do_sample=True, top_k=50,
                    pad_token_id=tokenizer.eos_token_id
                )
                completions.append(tokenizer.decode(out[0], skip_special_tokens=True))

        tokens = tokenizer(completions, return_tensors="pt", padding=True, truncation=True)
        ids = tokens["input_ids"].to(device)
        mask = tokens["attention_mask"].to(device)
        with torch.no_grad():
            ref_lp = h["get_log_prob_sums"](pi_ref, ids, mask)

        # DPO baseline
        pi_dpo = copy.deepcopy(pi_ref).train()
        dpo_lps = h["train_dpo"](pi_dpo, ref_lp, ids, mask, labeled_pairs, epochs=EPOCHS, device=device)
        dpo_m = h["compute_ranking_metrics"](np.array(dpo_lps[-1]), oracle_scores)
        method_results["dpo_baseline"]["agreement"].append(dpo_m["pairwise_agreement"])
        method_results["dpo_baseline"]["kendall"].append(dpo_m["kendall_tau"])
        del pi_dpo
        torch.cuda.empty_cache()

        # Each graph estimation method
        for name, fit_fn in methods.items():
            est_scores = fit_fn(N, labeled_pairs)
            full_graph = h["construct_full_graph"](est_scores)

            pi_method = copy.deepcopy(pi_ref).train()
            lps = h["train_dpo"](pi_method, ref_lp, ids, mask, full_graph, epochs=EPOCHS, device=device)
            m = h["compute_ranking_metrics"](np.array(lps[-1]), oracle_scores)
            method_results[name]["agreement"].append(m["pairwise_agreement"])
            method_results[name]["kendall"].append(m["kendall_tau"])
            del pi_method
            torch.cuda.empty_cache()

        del pi_ref
        torch.cuda.empty_cache()
        print(f"  Run {run_idx+1}/{NUM_RUNS}: " + " ".join(
            f"{name}={np.mean(method_results[name]['agreement']):.3f}" for name in method_results
        ))

    ablation1_results = {
        name: {
            "agreement_mean": float(np.mean(data["agreement"])),
            "agreement_std": float(np.std(data["agreement"])),
            "kendall_mean": float(np.mean(data["kendall"])),
        }
        for name, data in method_results.items()
    }

    # Ablation 2: Confidence-weighted DPO
    print("\n=== Ablation 2: Confidence-Weighted DPO ===")
    weight_results = {"unweighted": {"agreement": []}, "confidence_weighted": {"agreement": []}}

    for run_idx in range(NUM_RUNS):
        seed = BASE_SEED + run_idx + 5000
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        prompt = "".join(random.choices(string.ascii_letters + string.digits, k=20))
        pi_ref = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        pi_ref.config.use_cache = False
        pi_ref.eval()

        perm = np.random.permutation(N)
        oracle_scores = np.empty(N)
        for rank, idx in enumerate(perm):
            oracle_scores[idx] = rank + 1

        sampled_pairs = h["sample_pairs_random"](N, K)
        labeled_pairs = h["label_pairs"](sampled_pairs, oracle_scores)
        bt_scores = h["fit_bradley_terry"](N, labeled_pairs)

        completions = []
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            for _ in range(N):
                out = pi_ref.generate(
                    **{k: v.clone() for k, v in inputs.items()},
                    max_length=20, do_sample=True, top_k=50,
                    pad_token_id=tokenizer.eos_token_id
                )
                completions.append(tokenizer.decode(out[0], skip_special_tokens=True))

        tokens = tokenizer(completions, return_tensors="pt", padding=True, truncation=True)
        ids = tokens["input_ids"].to(device)
        mask = tokens["attention_mask"].to(device)
        with torch.no_grad():
            ref_lp = h["get_log_prob_sums"](pi_ref, ids, mask)

        # Unweighted
        full_graph = h["construct_full_graph"](bt_scores)
        pi_uw = copy.deepcopy(pi_ref).train()
        uw_lps = h["train_dpo"](pi_uw, ref_lp, ids, mask, full_graph, epochs=EPOCHS, device=device)
        uw_m = h["compute_ranking_metrics"](np.array(uw_lps[-1]), oracle_scores)
        weight_results["unweighted"]["agreement"].append(uw_m["pairwise_agreement"])
        del pi_uw
        torch.cuda.empty_cache()

        # Confidence-weighted
        weighted_graph, weights = h["construct_weighted_graph"](bt_scores)
        pi_cw = copy.deepcopy(pi_ref).train()
        cw_lps = h["train_dpo"](pi_cw, ref_lp, ids, mask, weighted_graph, epochs=EPOCHS,
                                weights=weights, device=device)
        cw_m = h["compute_ranking_metrics"](np.array(cw_lps[-1]), oracle_scores)
        weight_results["confidence_weighted"]["agreement"].append(cw_m["pairwise_agreement"])
        del pi_cw, pi_ref
        torch.cuda.empty_cache()

        print(f"  Run {run_idx+1}: UW={uw_m['pairwise_agreement']:.3f} CW={cw_m['pairwise_agreement']:.3f}")

    ablation2_results = {
        name: {"agreement_mean": float(np.mean(data["agreement"])),
               "agreement_std": float(np.std(data["agreement"]))}
        for name, data in weight_results.items()
    }

    # Ablation 3: Graph sampling structure
    print("\n=== Ablation 3: Graph Sampling Structure ===")
    sampling_fns = {
        "random": h["sample_pairs_random"],
        "chain": h["sample_pairs_chain"],
        "star": h["sample_pairs_star"],
    }
    structure_results = {name: {"agreement": [], "bt_acc": []} for name in sampling_fns}

    for run_idx in range(NUM_RUNS):
        seed = BASE_SEED + run_idx + 9000
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        prompt = "".join(random.choices(string.ascii_letters + string.digits, k=20))
        pi_ref = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        pi_ref.config.use_cache = False
        pi_ref.eval()

        perm = np.random.permutation(N)
        oracle_scores = np.empty(N)
        for rank, idx in enumerate(perm):
            oracle_scores[idx] = rank + 1

        completions = []
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            for _ in range(N):
                out = pi_ref.generate(
                    **{k: v.clone() for k, v in inputs.items()},
                    max_length=20, do_sample=True, top_k=50,
                    pad_token_id=tokenizer.eos_token_id
                )
                completions.append(tokenizer.decode(out[0], skip_special_tokens=True))

        tokens = tokenizer(completions, return_tensors="pt", padding=True, truncation=True)
        ids = tokens["input_ids"].to(device)
        mask = tokens["attention_mask"].to(device)
        with torch.no_grad():
            ref_lp = h["get_log_prob_sums"](pi_ref, ids, mask)

        for name, sample_fn in sampling_fns.items():
            # Use same seed offset for fair comparison
            np.random.seed(seed + hash(name) % 10000)
            sampled = sample_fn(N, K)
            labeled = h["label_pairs"](sampled, oracle_scores)
            bt_scores = h["fit_bradley_terry"](N, labeled)
            bt_agree, bt_total = h["count_pair_agreements"](bt_scores, oracle_scores)
            structure_results[name]["bt_acc"].append(bt_agree / bt_total)

            full_graph = h["construct_full_graph"](bt_scores)
            pi_m = copy.deepcopy(pi_ref).train()
            lps = h["train_dpo"](pi_m, ref_lp, ids, mask, full_graph, epochs=EPOCHS, device=device)
            m = h["compute_ranking_metrics"](np.array(lps[-1]), oracle_scores)
            structure_results[name]["agreement"].append(m["pairwise_agreement"])
            del pi_m
            torch.cuda.empty_cache()

        del pi_ref
        torch.cuda.empty_cache()
        print(f"  Run {run_idx+1}: " + " ".join(
            f"{name}={np.mean(structure_results[name]['agreement']):.3f}" for name in sampling_fns
        ))

    ablation3_results = {
        name: {
            "agreement_mean": float(np.mean(data["agreement"])),
            "agreement_std": float(np.std(data["agreement"])),
            "bt_accuracy_mean": float(np.mean(data["bt_acc"])),
        }
        for name, data in structure_results.items()
    }

    # Save all ablation results
    all_ablations = {
        "graph_estimation_methods": ablation1_results,
        "confidence_weighting": ablation2_results,
        "graph_sampling_structure": ablation3_results,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/exp3_ablations.json", "w") as f:
        json.dump(all_ablations, f, indent=2)

    # Plot ablation 1
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(ablation1_results.keys())
    means = [ablation1_results[n]["agreement_mean"] for n in names]
    stds = [ablation1_results[n]["agreement_std"] for n in names]
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=["tab:gray", "tab:blue", "tab:green", "tab:red"])
    ax.set_ylabel("Pairwise Agreement")
    ax.set_title("Graph Estimation Methods (N=30, K=30)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/exp3_graph_methods.png", dpi=150)
    plt.close()

    results_vol.commit()
    print("\n=== Experiment 3 (Ablations) Complete ===")
    print(json.dumps(all_ablations, indent=2))
    return all_ablations


# ============================================================
# Experiment 4: Qwen3-1.7B + Reward Model (Scaled Up)
# ============================================================

@app.function(**COMMON_KWARGS)
def exp4_reward_model_scaled():
    """
    Scaled-up reward model experiment with Qwen3-1.7B.
    Tests N={15,30,50} with reward model oracle.
    """
    import copy
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

    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    h = _ggdpo_helpers()
    BASE_SEED = 44

    policy_model_name = "Qwen/Qwen3-1.7B"
    reward_model_name = "Skywork/Skywork-Reward-V2-Qwen3-1.7B"

    tokenizer = AutoTokenizer.from_pretrained(policy_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_name, torch_dtype=torch.bfloat16
    ).to(device).eval()

    topics = [
        "democracy", "technology", "education", "climate", "healthcare",
        "economics", "ethics", "artificial intelligence", "privacy", "globalization",
    ]

    N_values = [15, 30, 50]
    EPOCHS = 50
    MAX_NEW_TOKENS = 256

    def generate_completions(model, prompt, n, max_new_tokens=256):
        model.eval()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        completions = []
        with torch.no_grad():
            for _ in range(n):
                out = model.generate(
                    **{k: v.clone() for k, v in inputs.items()},
                    max_new_tokens=max_new_tokens, do_sample=True, top_k=50,
                    pad_token_id=tokenizer.pad_token_id
                )
                text = tokenizer.decode(out[0], skip_special_tokens=True)
                if text.startswith(prompt):
                    text = text[len(prompt):]
                completions.append(text)
        return completions

    def get_reward_scores(prompt, completions, batch_size=4):
        scores = []
        for i in range(0, len(completions), batch_size):
            batch = completions[i:i + batch_size]
            texts = [f"{prompt}\n\n{c}" for c in batch]
            inputs = tokenizer(texts, return_tensors="pt", padding=True,
                             truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = reward_model(**inputs)
                batch_scores = outputs.logits.squeeze(-1).float().cpu().numpy()
                scores.extend(batch_scores.tolist())
        return np.array(scores)

    all_results = []

    for N in N_values:
        K = N
        NUM_RUNS = min(len(topics), 10)

        print(f"\n=== N={N}, K={K} ===")
        run_data = {"dpo": [], "ggdpo": [], "bt_acc": [],
                    "dpo_reward": [], "ggdpo_reward": []}

        for run_idx in range(NUM_RUNS):
            seed = BASE_SEED + run_idx
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            topic = topics[run_idx]
            prompt = f"Write a detailed, well-structured essay about {topic}."
            print(f"  Run {run_idx+1}: topic='{topic}'")

            pi_ref = AutoModelForCausalLM.from_pretrained(
                policy_model_name, torch_dtype=torch.bfloat16
            ).to(device)
            pi_ref.config.use_cache = False
            pi_ref.eval()

            pi_dpo = copy.deepcopy(pi_ref).train()
            pi_ggdpo = copy.deepcopy(pi_ref).train()

            completions = generate_completions(pi_ref, prompt, N, MAX_NEW_TOKENS)
            oracle_scores = get_reward_scores(prompt, completions)

            sampled_pairs = h["sample_pairs_random"](N, K)
            labeled_pairs = h["label_pairs"](sampled_pairs, oracle_scores)

            bt_scores = h["fit_bradley_terry"](N, labeled_pairs)
            bt_agree, bt_total = h["count_pair_agreements"](bt_scores, oracle_scores)
            run_data["bt_acc"].append(bt_agree / bt_total)

            full_graph = h["construct_full_graph"](bt_scores)

            tokens = tokenizer(completions, return_tensors="pt", padding=True, truncation=True)
            ids = tokens["input_ids"].to(device)
            mask = tokens["attention_mask"].to(device)
            with torch.no_grad():
                ref_lp = h["get_log_prob_sums"](pi_ref, ids, mask)

            dpo_lps = h["train_dpo"](pi_dpo, ref_lp, ids, mask, labeled_pairs,
                                     epochs=EPOCHS, lr=1e-5, beta=0.1, device=device,
                                     model_is_bf16=True)
            ggdpo_lps = h["train_dpo"](pi_ggdpo, ref_lp, ids, mask, full_graph,
                                       epochs=EPOCHS, lr=1e-5, beta=0.1, device=device,
                                       model_is_bf16=True)

            dpo_m = h["compute_ranking_metrics"](np.array(dpo_lps[-1]), oracle_scores)
            ggdpo_m = h["compute_ranking_metrics"](np.array(ggdpo_lps[-1]), oracle_scores)
            run_data["dpo"].append(dpo_m["pairwise_agreement"])
            run_data["ggdpo"].append(ggdpo_m["pairwise_agreement"])

            # Evaluate: generate new completions and score them
            dpo_eval = generate_completions(pi_dpo, prompt, 10, MAX_NEW_TOKENS)
            ggdpo_eval = generate_completions(pi_ggdpo, prompt, 10, MAX_NEW_TOKENS)
            dpo_eval_scores = get_reward_scores(prompt, dpo_eval)
            ggdpo_eval_scores = get_reward_scores(prompt, ggdpo_eval)
            run_data["dpo_reward"].append(float(np.mean(dpo_eval_scores)))
            run_data["ggdpo_reward"].append(float(np.mean(ggdpo_eval_scores)))

            del pi_ref, pi_dpo, pi_ggdpo
            torch.cuda.empty_cache()

            print(f"    DPO={dpo_m['pairwise_agreement']:.3f} GGDPO={ggdpo_m['pairwise_agreement']:.3f} "
                  f"DPO_reward={np.mean(dpo_eval_scores):.3f} GGDPO_reward={np.mean(ggdpo_eval_scores):.3f}")

        result = {
            "N": N, "K": K,
            "dpo_agreement_mean": float(np.mean(run_data["dpo"])),
            "dpo_agreement_std": float(np.std(run_data["dpo"])),
            "ggdpo_agreement_mean": float(np.mean(run_data["ggdpo"])),
            "ggdpo_agreement_std": float(np.std(run_data["ggdpo"])),
            "bt_accuracy_mean": float(np.mean(run_data["bt_acc"])),
            "dpo_reward_mean": float(np.mean(run_data["dpo_reward"])),
            "dpo_reward_std": float(np.std(run_data["dpo_reward"])),
            "ggdpo_reward_mean": float(np.mean(run_data["ggdpo_reward"])),
            "ggdpo_reward_std": float(np.std(run_data["ggdpo_reward"])),
        }
        all_results.append(result)
        print(f"  Avg: DPO_agree={result['dpo_agreement_mean']:.3f} GGDPO_agree={result['ggdpo_agreement_mean']:.3f}")
        print(f"  Avg: DPO_reward={result['dpo_reward_mean']:.3f} GGDPO_reward={result['ggdpo_reward_mean']:.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/exp4_reward_model_scaled.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ns = [r["N"] for r in all_results]
    ax1.bar([x - 0.2 for x in range(len(ns))], [r["dpo_agreement_mean"] for r in all_results],
            0.4, label="DPO", yerr=[r["dpo_agreement_std"] for r in all_results], capsize=3)
    ax1.bar([x + 0.2 for x in range(len(ns))], [r["ggdpo_agreement_mean"] for r in all_results],
            0.4, label="GGDPO", yerr=[r["ggdpo_agreement_std"] for r in all_results], capsize=3)
    ax1.set_xticks(range(len(ns)))
    ax1.set_xticklabels([str(n) for n in ns])
    ax1.set_xlabel("N (completions)")
    ax1.set_ylabel("Oracle Agreement")
    ax1.set_title("Reward Model Agreement")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar([x - 0.2 for x in range(len(ns))], [r["dpo_reward_mean"] for r in all_results],
            0.4, label="DPO", yerr=[r["dpo_reward_std"] for r in all_results], capsize=3)
    ax2.bar([x + 0.2 for x in range(len(ns))], [r["ggdpo_reward_mean"] for r in all_results],
            0.4, label="GGDPO", yerr=[r["ggdpo_reward_std"] for r in all_results], capsize=3)
    ax2.set_xticks(range(len(ns)))
    ax2.set_xticklabels([str(n) for n in ns])
    ax2.set_xlabel("N (completions)")
    ax2.set_ylabel("Post-Alignment Reward Score")
    ax2.set_title("Post-Alignment Quality")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/exp4_reward_model_scaled.png", dpi=150)
    plt.close()

    results_vol.commit()
    print("\n=== Experiment 4 (Reward Model Scaled) Complete ===")
    return all_results


# ============================================================
# Experiment 5: UltraFeedback Benchmark
# ============================================================

@app.function(**COMMON_KWARGS)
def exp5_ultrafeedback():
    """
    UltraFeedback benchmark: subsample pairs, GGDPO expands to full graph.
    Uses Qwen3-1.7B with DPO training on real preference data.
    Evaluates sample efficiency frontier.
    """
    import json
    import os
    import random
    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
    from peft import get_peft_model, LoraConfig, TaskType

    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    h = _ggdpo_helpers()
    BASE_SEED = 42

    model_name = "Qwen/Qwen3-1.7B"
    reward_model_name = "Skywork/Skywork-Reward-V2-Qwen3-1.7B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token

    print("Loading UltraFeedback dataset...")
    ds = load_dataset("openbmb/UltraFeedback", split="train")
    print(f"Loaded {len(ds)} examples")

    # Filter to examples with >= 4 completions that have overall_score
    valid_examples = []
    for ex in ds:
        completions = ex.get("completions", [])
        if len(completions) >= 4:
            rated = []
            for c in completions[:4]:
                # overall_score is a direct field on each completion (float)
                rating = c.get("overall_score", None)
                if rating is not None:
                    try:
                        rating = float(rating)
                    except (ValueError, TypeError):
                        rating = None
                if rating is not None:
                    rated.append({"response": c["response"], "rating": rating})
            if len(rated) >= 4:
                valid_examples.append({
                    "instruction": ex["instruction"],
                    "completions": rated[:4],
                })
        if len(valid_examples) >= 2000:
            break

    print(f"Valid examples with 4+ rated completions: {len(valid_examples)}")

    # Split: 500 train, 200 eval (balanced for compute budget)
    random.seed(BASE_SEED)
    random.shuffle(valid_examples)
    train_examples = valid_examples[:500]
    eval_examples = valid_examples[500:700]
    print(f"Train: {len(train_examples)}, Eval: {len(eval_examples)}")

    def build_dpo_pairs(examples, k_pairs_per_prompt):
        """Build DPO training pairs. Returns list of (prompt, chosen, rejected)."""
        pairs = []
        for ex in examples:
            n = len(ex["completions"])
            ratings = [c["rating"] for c in ex["completions"]]
            # All possible pairs
            all_pairs_idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
            # Sample k pairs
            k = min(k_pairs_per_prompt, len(all_pairs_idx))
            sampled = random.sample(all_pairs_idx, k)
            labeled = h["label_pairs"](sampled, ratings)
            for w, l in labeled:
                pairs.append((ex["instruction"], ex["completions"][w]["response"],
                             ex["completions"][l]["response"]))
        return pairs

    def build_ggdpo_pairs(examples, k_pairs_per_prompt):
        """Build GGDPO training pairs: sample k, fit BT, expand to full graph."""
        pairs = []
        for ex in examples:
            n = len(ex["completions"])
            ratings = [c["rating"] for c in ex["completions"]]
            all_pairs_idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
            k = min(k_pairs_per_prompt, len(all_pairs_idx))
            sampled = random.sample(all_pairs_idx, k)
            labeled = h["label_pairs"](sampled, ratings)
            # Fit BT from sampled pairs
            bt_scores = h["fit_bradley_terry"](n, labeled, n_iters=500)
            # Expand to full graph
            full_graph = h["construct_full_graph"](bt_scores)
            for w, l in full_graph:
                pairs.append((ex["instruction"], ex["completions"][w]["response"],
                             ex["completions"][l]["response"]))
        return pairs

    def build_full_dpo_pairs(examples):
        """Build DPO pairs using ALL available pairs (upper bound)."""
        pairs = []
        for ex in examples:
            n = len(ex["completions"])
            ratings = [c["rating"] for c in ex["completions"]]
            all_pairs_idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
            labeled = h["label_pairs"](all_pairs_idx, ratings)
            for w, l in labeled:
                pairs.append((ex["instruction"], ex["completions"][w]["response"],
                             ex["completions"][l]["response"]))
        return pairs

    def train_dpo_on_pairs(model_name, dpo_pairs, num_epochs=3, batch_size=4, lr=5e-6, max_length=512):
        """Train a model with LoRA DPO on the given pairs."""
        if len(dpo_pairs) == 0:
            print("    WARNING: No DPO pairs to train on!")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16
            ).to(device)
            return model, 0.0

        # Load frozen reference model
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        # Load policy model with LoRA
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(device)
        model.config.use_cache = False

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0,
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        total_loss = 0
        num_batches = 0

        for epoch in range(num_epochs):
            random.shuffle(dpo_pairs)
            for start in range(0, len(dpo_pairs), batch_size):
                batch = dpo_pairs[start:start + batch_size]

                chosen_texts = [f"{p}\n\n{c}" for p, c, _ in batch]
                rejected_texts = [f"{p}\n\n{r}" for p, _, r in batch]

                chosen_enc = tokenizer(chosen_texts, return_tensors="pt", padding=True,
                                     truncation=True, max_length=max_length).to(device)
                rejected_enc = tokenizer(rejected_texts, return_tensors="pt", padding=True,
                                       truncation=True, max_length=max_length).to(device)

                optimizer.zero_grad()

                # Policy log probs
                chosen_lp = h["get_log_prob_sums"](model, chosen_enc["input_ids"], chosen_enc["attention_mask"])
                rejected_lp = h["get_log_prob_sums"](model, rejected_enc["input_ids"], rejected_enc["attention_mask"])

                # Reference log probs from frozen reference model
                with torch.no_grad():
                    ref_chosen_lp = h["get_log_prob_sums"](ref_model, chosen_enc["input_ids"], chosen_enc["attention_mask"])
                    ref_rejected_lp = h["get_log_prob_sums"](ref_model, rejected_enc["input_ids"], rejected_enc["attention_mask"])

                beta = 0.1
                logits = beta * ((chosen_lp - ref_chosen_lp) - (rejected_lp - ref_rejected_lp))
                loss = -nn.functional.logsigmoid(logits).mean()

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if num_batches % 100 == 0:
                    print(f"    Epoch {epoch+1}/{num_epochs}, Batch {num_batches}, loss={total_loss/num_batches:.4f}")

        del ref_model
        torch.cuda.empty_cache()

        avg_loss = total_loss / max(num_batches, 1)
        return model, avg_loss

    def evaluate_with_reward_model(model, eval_examples, reward_model, n_gen=2, max_new_tokens=128):
        """Generate completions and score with reward model."""
        model.eval()
        all_scores = []

        for idx, ex in enumerate(eval_examples[:50]):  # Evaluate on 50 examples
            prompt = ex["instruction"]
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

            completions = []
            with torch.no_grad():
                for _ in range(n_gen):
                    out = model.generate(
                        **{k: v.clone() for k, v in inputs.items()},
                        max_new_tokens=max_new_tokens, do_sample=True, top_k=50,
                        pad_token_id=tokenizer.pad_token_id
                    )
                    text = tokenizer.decode(out[0], skip_special_tokens=True)
                    completions.append(text)

            # Score using reward model tokenizer
            for comp in completions:
                enc = reward_tokenizer(comp, return_tensors="pt", truncation=True, max_length=512).to(device)
                with torch.no_grad():
                    score = reward_model(**enc).logits.squeeze().float().item()
                    all_scores.append(score)

            if (idx + 1) % 10 == 0:
                print(f"    Eval: {idx+1}/50 examples scored")

        return float(np.mean(all_scores)), float(np.std(all_scores))

    # Load reward model for evaluation
    print("Loading reward model...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_name, torch_dtype=torch.bfloat16
    ).to(device).eval()

    # Sample efficiency frontier: vary K
    K_values = [1, 2, 3, 6]  # 6 = all pairs for 4 completions
    results = []

    for K in K_values:
        print(f"\n=== K={K} pairs per prompt ===")

        torch.manual_seed(BASE_SEED)
        np.random.seed(BASE_SEED)
        random.seed(BASE_SEED)

        if K >= 6:
            # Full DPO (upper bound)
            pairs = build_full_dpo_pairs(train_examples)
            label = f"Full DPO (K=6)"
        else:
            pairs = build_dpo_pairs(train_examples, K)
            label = f"DPO (K={K})"

        print(f"  DPO pairs: {len(pairs)}")
        dpo_model, dpo_loss = train_dpo_on_pairs(model_name, pairs)
        dpo_reward_mean, dpo_reward_std = evaluate_with_reward_model(
            dpo_model, eval_examples, reward_model
        )
        print(f"  DPO: loss={dpo_loss:.4f}, reward={dpo_reward_mean:.3f}+-{dpo_reward_std:.3f}")
        del dpo_model
        torch.cuda.empty_cache()

        # GGDPO (only if K < 6)
        if K < 6:
            torch.manual_seed(BASE_SEED)
            np.random.seed(BASE_SEED)
            random.seed(BASE_SEED)

            ggdpo_pairs = build_ggdpo_pairs(train_examples, K)
            print(f"  GGDPO pairs (expanded): {len(ggdpo_pairs)}")
            ggdpo_model, ggdpo_loss = train_dpo_on_pairs(model_name, ggdpo_pairs)
            ggdpo_reward_mean, ggdpo_reward_std = evaluate_with_reward_model(
                ggdpo_model, eval_examples, reward_model
            )
            print(f"  GGDPO: loss={ggdpo_loss:.4f}, reward={ggdpo_reward_mean:.3f}+-{ggdpo_reward_std:.3f}")
            del ggdpo_model
            torch.cuda.empty_cache()
        else:
            ggdpo_loss = dpo_loss
            ggdpo_reward_mean = dpo_reward_mean
            ggdpo_reward_std = dpo_reward_std

        results.append({
            "K": K,
            "oracle_queries_per_prompt": K,
            "dpo_pairs_total": len(pairs),
            "ggdpo_pairs_total": len(ggdpo_pairs) if K < 6 else len(pairs),
            "dpo_loss": dpo_loss,
            "dpo_reward_mean": dpo_reward_mean,
            "dpo_reward_std": dpo_reward_std,
            "ggdpo_loss": ggdpo_loss if K < 6 else None,
            "ggdpo_reward_mean": ggdpo_reward_mean,
            "ggdpo_reward_std": ggdpo_reward_std,
        })

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/exp5_ultrafeedback.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot: Sample Efficiency Frontier
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = [r["K"] for r in results]
    dpo_rewards = [r["dpo_reward_mean"] for r in results]
    ggdpo_rewards = [r["ggdpo_reward_mean"] for r in results]

    ax.plot(ks, dpo_rewards, "o-", label="DPO", color="tab:blue", markersize=8)
    ax.plot(ks, ggdpo_rewards, "x-", label="GGDPO", color="tab:orange", markersize=8)
    ax.set_xlabel("Oracle Queries per Prompt (K)")
    ax.set_ylabel("Post-Alignment Reward Score")
    ax.set_title("Sample Efficiency Frontier: UltraFeedback")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/exp5_sample_efficiency.png", dpi=150)
    plt.close()

    results_vol.commit()
    print("\n=== Experiment 5 (UltraFeedback) Complete ===")
    print(json.dumps(results, indent=2))
    return results


# ============================================================
# Run all experiments
# ============================================================

@app.local_entrypoint()
def main():
    """Run all experiments sequentially."""
    print("Starting GGDPO experiments...")

    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Synthetic N/K Sweep")
    print("=" * 60)
    r1 = exp1_synthetic_sweep.remote()

    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Scaling N")
    print("=" * 60)
    r2 = exp2_scaling_n.remote()

    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Ablations")
    print("=" * 60)
    r3 = exp3_ablations.remote()

    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Reward Model Scaled")
    print("=" * 60)
    r4 = exp4_reward_model_scaled.remote()

    print("\n" + "=" * 60)
    print("EXPERIMENT 5: UltraFeedback")
    print("=" * 60)
    r5 = exp5_ultrafeedback.remote()

    print("\n\nAll experiments complete!")
