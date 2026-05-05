import cupy as cp
import numpy as np
from cuml.linear_model import LogisticRegression as cuLogReg
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression as SkLearnLogReg

def convert_to_cpu(cu_model):

    # pass parameters into the new empty sklearn model
    cpu_model = SkLearnLogReg(
        penalty=cu_model.penalty,
        C=cu_model.C,
        class_weight=cu_model.class_weight
    )

    # translate to np arrays
    cpu_coef = cp.asnumpy(cu_model.coef_)
    cpu_intercept = cp.asnumpy(cu_model.intercept_)
    cpu_classes = cp.asnumpy(cu_model.classes_)

    # assign the translated parameters to the CPU model
    cpu_model.coef_ = cpu_coef
    cpu_model.intercept_ = cpu_intercept
    cpu_model.classes_ = cpu_classes

    # transfer features
    cpu_model.n_features_in_ = cu_model.n_features_in_

    # return the CPU model ready for inference
    return cpu_model

def perform_owlqn_constrained_search(X_train, y_train, args, total_neurons, sparsity_limit=0.001):
    """
    Dynamically finds the optimal C by walking up the penalty curve until 
    the model violates the strict < 0.1% architectural sparsity constraint.
    """
    print("Initializing GPU Tensors...")
    X_train_gpu = cp.asarray(X_train)
    y_train_gpu = cp.asarray(y_train)

    # track the ROI on adding more neurons to prevent overfitting to max neurons allowed
    previous_accuracy = 0.5 # binary classification baseline for ROI tracking
    previous_roi = float('inf')
    previous_neuron_count = 0
    consecutive_dips = 0

    MIN_ACCEPTABLE_ROI = 0.001 # we want at least a 0.1% accuracy improvement for each additional neuron to consider it worthwhile
    MIN_CIRCUIT_SIZE = 5 # Do not punish until at least 5 neurons are active
    MAX_SEARCH_STEPS = 80 # hard stop to avoid unbounded generator loops

    # 1. Define the physical constraint
    max_allowed_neurons = int(total_neurons * sparsity_limit)
    print(f"Architecture Limit: {max_allowed_neurons} neurons max ({sparsity_limit*100}% of {total_neurons})")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_indices = list(skf.split(np.zeros(len(y_train)), y_train))

    # A logarithmic generator allows us to dynamically test C without a hardcoded list
    # Starts at 0.01 and multiplies by 1.15 each step
    def c_generator(start=0.01, step_multiplier=1.15):
        current = start
        while True:
            yield current
            current *= step_multiplier

    best_score = float('-inf')
    best_c = None
    best_neuron_count = 0

    print("-" * 60)
    print(f"{'C-Value':<10} | {'CV Accuracy':<15} | {'Active Neurons':<15} | {'Status':<20}")
    print("-" * 60)

    for _, c in zip(range(MAX_SEARCH_STEPS), c_generator()):
        # --- Cross Validation Block ---
        fold_scores = []
        fold_sparsities = []

        for train_idx, val_idx in fold_indices:
            X_train_fold, y_train_fold = X_train_gpu[train_idx], y_train_gpu[train_idx]
            X_val_fold, y_val_fold = X_train_gpu[val_idx], y_train_gpu[val_idx]

            model = cuLogReg(
            penalty=args.penalty,
            solver='qn',
            C=c,
            class_weight='balanced',
            tol=1e-4,
            max_iter=10000 # compute time traded for better convergence
            )

            model.fit(X_train_fold, y_train_fold)

            score = model.score(X_val_fold, y_val_fold)
            fold_scores.append(score)
            fold_sparsities.append(int(cp.sum(model.coef_ != 0)))

            del X_train_fold, y_train_fold, X_val_fold, y_val_fold
            del model

        avg_score_train = sum(fold_scores) / len(fold_scores)

        # --- The Programmatic Constraint Check ---
        # Use the worst fold to enforce a conservative architecture limit.
        active_neurons = max(fold_sparsities)

        if active_neurons > max_allowed_neurons:
            print(f"{c:<10.4f} | {avg_score_train:<15.4f} | {active_neurons:<15} | ❌ VIOLATION (Breaking Loop)")
            break # The physical threshold was crossed. Stop the search.
        
        # calculate marginal ROI
        delta_acc = avg_score_train - previous_accuracy
        delta_neurons = active_neurons - previous_neuron_count

        if delta_neurons > 0:
            current_roi = delta_acc / delta_neurons
        elif delta_neurons == 0 and delta_acc > 0:
            # continue momementum; accuracy went up for free
            current_roi = float('inf')
        else:
            current_roi = 0 # no improvement or negative improvement

        # check for diminishing ROIs
        if current_roi < MIN_ACCEPTABLE_ROI and active_neurons >= MIN_CIRCUIT_SIZE:
            consecutive_dips += 1
            status = f"⚠️  ROI Dip {consecutive_dips}/3"
        else:
            # momentum was regained, the dip counter can be reset
            consecutive_dips = 0
            status = "✅ Rising/Stable"


        if active_neurons == 0:
            status = "⚠️  Dead (0 neurons)"

        print(f"{c:<10.4f} | {avg_score_train:<15.4f} | {active_neurons:<15} | {status:<20}")

        # --- Model Tracking ---
        # Track only the C value; we will refit one definitive model at the end.
        if avg_score_train > best_score and active_neurons > 0 and consecutive_dips < 3:
            best_score = avg_score_train
            best_c = c
            best_neuron_count = active_neurons

        # break if consecutive dip count goes over 3
        if consecutive_dips >= 3:
            print(f"⛔ Stopping search due to 3 consecutive ROI dips. No improvement after {best_neuron_count} neurons.")
            break

        # update history
        previous_accuracy = avg_score_train
        previous_neuron_count = active_neurons
        previous_roi = current_roi

        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

    if best_c is None:
        raise RuntimeError(
            "No valid model was found under the sparsity limit. "
            "Try a larger sparsity_limit or a smaller starting C."
        )

    print("-" * 60)
    print("🏆 Optimal Guardrail Found programmatically:")
    print(f"C: {best_c:.4f} | Accuracy: {best_score:.4f} | Neurons: {best_neuron_count}")
    print("-" * 60)

    print(f"Fitting final definitive model on full dataset with C={best_c:.4f}...")
    final_model = cuLogReg(
        penalty=args.penalty,
        solver='qn',
        C=best_c,
        class_weight='balanced',
        tol=1e-4,
        max_iter=10000
    )
    final_model.fit(X_train_gpu, y_train_gpu)

    final_active_neurons = int(cp.sum(final_model.coef_ != 0))
    if final_active_neurons > max_allowed_neurons:
        raise RuntimeError(
            f"Final model violates sparsity limit after refit "
            f"({final_active_neurons} > {max_allowed_neurons})."
        )

    export_best_model = convert_to_cpu(final_model)

    return export_best_model