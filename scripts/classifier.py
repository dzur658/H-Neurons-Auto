import os
import json
import argparse
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import cross_validate
from transformers import AutoConfig

import psutil
import warnings

try:
    from acceleration.cuda_grid_search import perform_owlqn_constrained_search
    GPU_AVAILABLE = True

except ImportError as e:
    GPU_AVAILABLE = False
    print("WARNING: FAILED TO IMPORT CUDA LIBRARIES. OWL-QN WILL NOT BE AVAILABLE. INSTALL 'cuda' OPTIONAL DEPENDENCIES FOR GPU ACCELERATION.")
    print("Failure Reason:", e)

def parse_args():
    parser = argparse.ArgumentParser(description="Train H-Neuron detector with flexible directory inputs.")
    # Model config
    parser.add_argument("--model_path", type=str, required=True, help="Path to HF model for config")
    
    # Training inputs
    parser.add_argument("--train_ids", type=str, help="Path to train_qids.json")
    parser.add_argument("--train_ans_acts", type=str, help="Directory of answer tokens activations for training")
    parser.add_argument("--train_other_acts", type=str, help="Directory of other tokens activations (required for 3-vs-1)")
    
    # Testing inputs
    parser.add_argument("--test_ids", type=str, help="Path to test_qids.json")
    parser.add_argument("--test_acts", type=str, help="Directory of activations to evaluate (usually answer_tokens)")
    
    # Model Persistence
    parser.add_argument("--save_model", type=str, default="models/detector.pkl")
    parser.add_argument("--load_model", type=str, help="Load a pre-trained model for evaluation")
    
    # Training Parameters
    parser.add_argument("--train_mode", type=str, choices=["1-vs-1", "3-vs-1"], default="3-vs-1")
    parser.add_argument("--penalty", type=str, choices=["l1", "l2"], default="l1")

    # or perform grid search to find optimal C value
    parser.add_argument("--constrained_search", default=False, help="Whether to perform a constrained search (auto tune) for C parameter")

    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--solver", type=str, choices=["liblinear", "saga", "qn"], default="liblinear") # change to saga if needed
    
    return parser.parse_args()

def load_data(ids_path, ans_acts_dir, other_acts_dir=None, mode="1-vs-1"):
    """
    Flexible data loader.
    1-vs-1: False Answer Tokens (Label 1) vs True Answer Tokens (Label 0).
    3-vs-1: False Answer Tokens (Label 1) vs (True Ans + True Other + False Other) (Label 0).
    """
    with open(ids_path, "r") as f:
        id_map = json.load(f)
    
    X, y = [], []

    # 1. Load False Answer Tokens -> Always Label 1 (Positive)
    for qid in tqdm(id_map["f"], desc="Loading False Ans (Label 1)"):
        path = os.path.join(ans_acts_dir, f"act_{qid}.npy")
        if os.path.exists(path):
            X.append(np.load(path).flatten())
            y.append(1)

    # 2. Load True Answer Tokens -> Always Label 0 (Negative)
    for qid in tqdm(id_map["t"], desc="Loading True Ans (Label 0)"):
        path = os.path.join(ans_acts_dir, f"act_{qid}.npy")
        if os.path.exists(path):
            X.append(np.load(path).flatten())
            y.append(0)

    # 3. Load Other Tokens if 3-vs-1 mode is enabled
    if mode == "3-vs-1":
        if not other_acts_dir:
            raise ValueError("train_other_acts directory is required for 3-vs-1 mode.")
        
        for label_key in ["t", "f"]:
            for qid in tqdm(id_map[label_key], desc=f"Loading Other Tokens - {label_key} (Label 0)"):
                path = os.path.join(other_acts_dir, f"act_{qid}.npy")
                if os.path.exists(path):
                    X.append(np.load(path).flatten())
                    y.append(0)

    return np.array(X), np.array(y)

def run_evaluation(model, X, y, dataset_name="Test"):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    
    acc = accuracy_score(y, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y, preds, average="binary")
    auroc = roc_auc_score(y, probs)
    
    print(f"\n--- Results: {dataset_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUROC:     {auroc:.4f}")

def calculate_dynamic_n_jobs(X_train, overhead_multiplier, os_reserve_gb=4):
    """
    Calculates the maximum safe concurrent workers based on current hardware capacity.
    
    Args:
        X_train: Your feature array.
        overhead_multiplier: Scikit-Learn and liblinear need working memory on top 
                             of the array size. 2.5 is a very safe buffer.
        os_reserve_gb: How much RAM to leave entirely untouched for safety.
    """
    # 1. Query the Hardware
    total_cores = os.cpu_count()
    available_ram_bytes = psutil.virtual_memory().available 
    
    # 2. Establish Safety Buffers
    max_cpu_workers = max(1, total_cores - 2) # Leave 2 cores for OS/Networking
    usable_ram_bytes = available_ram_bytes - (os_reserve_gb * 1024**3)
    
    # 3. Calculate Memory Footprint per Worker
    array_bytes = X_train.nbytes
    # Each worker gets a copy of the array, plus needs room for gradient calculations
    ram_per_worker = array_bytes * overhead_multiplier
    
    # 4. Calculate RAM Bottleneck
    if ram_per_worker > 0:
        max_ram_workers = int(max(0, usable_ram_bytes) // ram_per_worker)
    else:
        max_ram_workers = max_cpu_workers
        
    # 5. The Final Verdict: The lowest ceiling wins
    n_jobs = min(max_cpu_workers, max_ram_workers)
    n_jobs = max(1, n_jobs) # Absolute minimum is 1

    print("="*50)
    print("💻 DYNAMIC HARDWARE ALLOCATION")
    print("="*50)
    print(f"System Cores:         {total_cores}")
    print(f"Available RAM:        {available_ram_bytes / 1024**3:.2f} GB")
    print(f"Array Size in RAM:    {array_bytes / 1024**3:.2f} GB")
    print(f"Est. Worker Need:     {ram_per_worker / 1024**3:.2f} GB per job")
    print("-" * 50)
    print(f"Max Workers (CPU):    {max_cpu_workers}")
    print(f"Max Workers (RAM):    {max_ram_workers}")
    print("="*50)
    print(f"🚀 Launching Grid Search with n_jobs = {n_jobs}")
    
    return n_jobs

def get_total_neurons(model_path):
    print(f"Loading model config from {model_path} to calculate total neurons...")

    config = AutoConfig.from_pretrained(model_path)

    # compatibility with vision language models
    if hasattr(config, "text_config") and hasattr(config.text_config, 'intermediate_size'):
        total_neurons = config.text_config.intermediate_size
        return total_neurons
    elif hasattr(config, 'intermediate_size'):
        total_neurons = config.intermediate_size
        return total_neurons
    else:
        raise ValueError("Unable to determine total neurons from model config. Please ensure the model has an 'intermediate_size' attribute in its config or text_config.")

def evaluate_model(model, test_x, test_y):
    preds = model.predict(test_x)

    test_accuracy = accuracy_score(test_y, preds)
    return test_accuracy

def perform_c_constrained_search(X_train, y_train, args, sparsity_limit=0.01):
    # we need to know how many total neurons are in the network
    # we assume that the number of neurons should not exceed 0.1% of total neurons in the model, as the H-neurons paper suggests
    total_neurons = get_total_neurons(args.model_path)

    # calcualte ideal n_jobs
    if args.solver == "liblinear":
        n_jobs_dynamic = calculate_dynamic_n_jobs(X_train, overhead_multiplier=20.0)
    elif args.solver == "saga":
        n_jobs_dynamic = calculate_dynamic_n_jobs(X_train, overhead_multiplier=1.5)
    elif args.solver == "qn" and GPU_AVAILABLE:
        # seperate module for CUDA-accelerated OWL-QN grid search

        model = perform_owlqn_constrained_search(X_train, y_train, args, total_neurons, sparsity_limit)
        return model
    else:
        raise ValueError(f"Unsupported solver for dynamic n_jobs calculation: {args.solver}")

    # Suppress convergence warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    # 1. Define the physical constraint
    max_allowed_neurons = int(total_neurons * sparsity_limit)
    print(f"Architecture Limit: {max_allowed_neurons} neurons max ({sparsity_limit*100}% of {total_neurons})")
    print(f"Using n_jobs={n_jobs_dynamic} for parallel CPU cross-validation.")

    print("Splitting data into CV set and hold out set...")
    x_cv, X_test, y_cv, y_test = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )
    
    # 2. The Logarithmic Generator
    def c_generator(start=0.001, step_multiplier=1.5):
        current = start
        while True:
            yield current
            current *= step_multiplier

    best_score = 0.0
    best_c = None
    best_neuron_count = 0

    print("-" * 60)
    print(f"{'C-Value':<10} | {'CV Accuracy':<15} | {'Active Neurons':<15} | {'Status':<10} | {'Accuracy on Hold-Out Set':<20}")
    print("-" * 60)

    for c in c_generator():
        test_model = LogisticRegression(
            penalty=args.penalty, 
            solver=args.solver, 
            C=c, 
            class_weight="balanced", 
            max_iter=1000, 
            random_state=42
        )

        # 3. Parallel Execution
        # return_estimator=True lets us inspect the models in memory without retraining
        cv_results = cross_validate(
            test_model, 
            X_cv, 
            y_cv, 
            cv=5, 
            scoring='accuracy', 
            n_jobs=n_jobs_dynamic,  # Spreads the 5 folds across your CPU cores
            return_estimator=True
        )

        avg_score_train = sum(cv_results['test_score']) / len(cv_results['test_score'])
        avg_score_test = evaluate_model(test_model, X_test, y_test)
        
        # 4. Check Sparsity across the folds
        fold_sparsities = [np.sum(est.coef_ != 0) for est in cv_results['estimator']]
        active_neurons = int(np.mean(fold_sparsities))

        # 5. The Architecture Constraint Check
        if active_neurons > max_allowed_neurons:
            print(f"{c:<10.4f} | {avg_score_train:<15.4f} | {active_neurons:<15} | ❌ VIOLATION (Breaking Loop)")
            break # Hit the physical limit. Stop computing immediately.

        status = "✅ Valid"
        if active_neurons == 0:
            status = "⚠️ Dead (0 neurons)"

        print(f"{c:<10.4f} | {avg_score_train:<15.4f} | {active_neurons:<15} | {status} | {avg_score_test:<20.4f}")

        # Track the best valid model
        if avg_score_test > best_score and active_neurons > 0:
            best_score = avg_score_train
            best_c = c
            best_neuron_count = active_neurons

    print("-" * 60)
    print(f"🏆 Optimal Guardrail Found programmatically:")
    print(f"C: {best_c:.4f} | Accuracy: {best_score:.4f} | Neurons: {best_neuron_count}")
    print("-" * 60)

    # 6. Final Definitive Fit
    # CV returns 5 fold-specific models. We now train one final master model 
    # on the entire dataset using the perfectly isolated C value.
    print(f"Fitting final definitive model on full dataset with C={best_c:.4f}...")
    best_model = LogisticRegression(
        penalty=args.penalty, 
        solver=args.solver, 
        C=best_c, 
        class_weight="balanced", 
        max_iter=1000, 
        random_state=42
    )
    best_model.fit(X_train, y_train)

    return best_model

def main():
    args = parse_args()
    model = None

    # --- Training Phase ---
    if args.load_model:
        print(f"Loading pre-trained model: {args.load_model}")
        model = joblib.load(args.load_model)
    elif args.train_ids and args.train_ans_acts:
        print(f"Training in {args.train_mode} mode...")
        X_train, y_train = load_data(
            args.train_ids, 
            args.train_ans_acts, 
            other_acts_dir=args.train_other_acts, 
            mode=args.train_mode
        )

        # perform grid search if requested
        if args.constrained_search:
            model = perform_c_constrained_search(X_train, y_train, args)
        else:
            # default to original static C method

            c = args.C
        
            model = LogisticRegression(
                penalty=args.penalty, C=c, solver=args.solver, 
                max_iter=1000, random_state=42, verbose=1
            )
            model.fit(X_train, y_train)
        
        if args.save_model:
            os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
            joblib.dump(model, args.save_model)
            print(f"Model saved. Identified {np.sum(model.coef_[0] > 0)} potential H-Neurons.")
            
        run_evaluation(model, X_train, y_train, "Training Set")
    else:
        print("Please provide --load_model OR (--train_ids AND --train_ans_acts)")
        return

    # --- Evaluation Phase ---
    if args.test_ids and args.test_acts:
        print(f"Evaluating on {args.test_ids}...")
        # Evaluation is typically 1-vs-1 (True Ans vs False Ans)
        X_test, y_test = load_data(args.test_ids, args.test_acts, mode="1-vs-1")
        run_evaluation(model, X_test, y_test, "Test Set")

if __name__ == "__main__":
    main()