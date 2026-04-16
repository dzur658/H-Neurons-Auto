import cupy as cp
import numpy as np
from cuml.linear_model import LogisticRegression as cuLogReg
from cuml.metrics import accuracy_score
import gc
import copy
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
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

def evaluate_model(model, test_x, test_y):
    preds = model.predict(test_x)
    test_accuracy = accuracy_score(test_y, preds)
    return test_accuracy

def perform_owlqn_constrained_search(X_train, y_train, args, total_neurons, sparsity_limit=0.001):
    """
    Dynamically finds the optimal C by walking up the penalty curve until 
    the model violates the strict < 0.1% architectural sparsity constraint.
    """
    print("Splitting data into CV set and hold out set...")
    X_cv, X_test, y_cv, y_test = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    print("Initializing GPU Tensors...")
    X_train_gpu = cp.asarray(X_cv)
    y_train_gpu = cp.asarray(y_cv)
    X_test_gpu = cp.asarray(X_test)
    y_test_gpu = cp.asarray(y_test)

    # 1. Define the physical constraint
    max_allowed_neurons = int(total_neurons * sparsity_limit)
    print(f"Architecture Limit: {max_allowed_neurons} neurons max ({sparsity_limit*100}% of {total_neurons})")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_indices = list(skf.split(np.zeros(len(y_train)), y_train))

    # A logarithmic generator allows us to dynamically test C without a hardcoded list
    # Starts at 0.01 and multiplies by 1.5 each step
    def c_generator(start=0.01, step_multiplier=1.5):
        current = start
        while True:
            yield current
            current *= step_multiplier

    best_score = 0.0
    best_model = None
    best_c = None
    best_neuron_count = 0

    print("-" * 60)
    print(f"{'C-Value':<10} | {'CV Accuracy':<15} | {'Active Neurons':<15} | {'Status':<10} | {'Accuracy on Hold-Out Set':<20}")
    print("-" * 60)

    for c in c_generator():
        # --- Cross Validation Block ---
        fold_scores = []
        for train_idx, val_idx in fold_indices:
            X_train_fold, y_train_fold = X_train_gpu[train_idx], y_train_gpu[train_idx]
            X_val_fold, y_val_fold = X_train_gpu[val_idx], y_train_gpu[val_idx]

            model = cuLogReg(penalty=args.penalty, solver='qn', C=c, class_weight='balanced', tol=1e-4)
            model.fit(X_train_fold, y_train_fold)
            
            score = model.score(X_val_fold, y_val_fold)
            fold_scores.append(score)
            
            del X_train_fold, y_train_fold, X_val_fold, y_val_fold

        avg_score_train = sum(fold_scores) / len(fold_scores)
        avg_score_test = evaluate_model(model, X_test_gpu, y_test_gpu)

        # --- The Programmatic Constraint Check ---
        # We look at the final model of the CV folds to check sparsity
        active_neurons = int(cp.sum(model.coef_ != 0))

        if active_neurons > max_allowed_neurons:
            print(f"{c:<10.4f} | {avg_score_train:<15.4f} | {active_neurons:<15} | ❌ VIOLATION (Breaking Loop)")
            break # The physical threshold was crossed. Stop the search.

        status = "✅ Valid"
        if active_neurons == 0:
            status = "⚠️ Dead (0 neurons)"

        print(f"{c:<10.4f} | {avg_score_train:<15.4f} | {active_neurons:<15} | {status} | {avg_score_test:<20.4f}")

        # --- Model Tracking ---
        # We only save it if it's the highest accuracy AND it has at least 1 active neuron
        if avg_score_train > best_score and active_neurons > 0:
            best_score = avg_score_train
            best_c = c
            best_model = copy.deepcopy(model)
            best_neuron_count = active_neurons

        cp._default_memory_pool.free_all_blocks()
        gc.collect()
        del model

    print("-" * 60)
    print("🏆 Optimal Guardrail Found programmatically:")
    print(f"C: {best_c:.4f} | Accuracy: {best_score:.4f} | Neurons: {best_neuron_count}")
    print("-" * 60)

    export_best_model = convert_to_cpu(best_model)

    return export_best_model