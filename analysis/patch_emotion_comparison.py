from pathlib import Path

path = Path("analysis/run_emotion_branch_comparison.py")
text = path.read_text(encoding="utf-8")

old_args = '''    parser.add_argument(
        "--emotion-weight",
        type=float,
        default=0.65,
    )
    parser.add_argument(
        "--appraisal-weight",
        type=float,
        default=0.35,
    )
'''

new_args = '''    parser.add_argument(
        "--emotion-weight",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--appraisal-weight",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--min-emotion-candidate-score",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--appraisal-add-threshold",
        type=float,
        default=0.72,
    )
    parser.add_argument(
        "--neutral-protection-threshold",
        type=float,
        default=0.55,
    )
'''

old_call = '''        comparison = compare_branches(
            emotion_scores=emotion_scores,
            appraisal_scores=appraisal_emotions,
            top_k=args.top_k,
            emotion_weight=args.emotion_weight,
            appraisal_weight=args.appraisal_weight,
        )
'''

new_call = '''        comparison = compare_branches(
            emotion_scores=emotion_scores,
            appraisal_scores=appraisal_emotions,
            top_k=args.top_k,
            emotion_weight=args.emotion_weight,
            appraisal_weight=args.appraisal_weight,
            candidate_pool_size=args.candidate_pool_size,
            min_emotion_candidate_score=(
                args.min_emotion_candidate_score
            ),
            appraisal_add_threshold=(
                args.appraisal_add_threshold
            ),
            neutral_protection_threshold=(
                args.neutral_protection_threshold
            ),
        )
'''

old_json = '''            "integrated_scores_json": json.dumps(
                comparison["integrated_scores"],
                ensure_ascii=False,
                sort_keys=True,
            ),
'''

new_json = '''            "integrated_scores_json": json.dumps(
                comparison["integrated_scores"],
                ensure_ascii=False,
                sort_keys=True,
            ),
            "integrated_candidate_labels_json": json.dumps(
                comparison["candidate_labels"],
                ensure_ascii=False,
            ),
            "appraisal_added_candidate": (
                comparison["added_by_appraisal"]
            ),
'''

old_meta = '''        "emotion_weight": args.emotion_weight,
        "appraisal_weight": args.appraisal_weight,
        "top_k": args.top_k,
'''

new_meta = '''        "emotion_weight": args.emotion_weight,
        "appraisal_weight": args.appraisal_weight,
        "candidate_pool_size": args.candidate_pool_size,
        "min_emotion_candidate_score": (
            args.min_emotion_candidate_score
        ),
        "appraisal_add_threshold": (
            args.appraisal_add_threshold
        ),
        "neutral_protection_threshold": (
            args.neutral_protection_threshold
        ),
        "integration_method": "candidate_gated_late_fusion",
        "top_k": args.top_k,
'''

for old, new, label in [
    (old_args, new_args, "argument block"),
    (old_call, new_call, "compare_branches call"),
    (old_json, new_json, "output JSON block"),
    (old_meta, new_meta, "metadata block"),
]:
    if old not in text:
        raise RuntimeError(f"Could not find {label}.")
    text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
print("Patched:", path)
