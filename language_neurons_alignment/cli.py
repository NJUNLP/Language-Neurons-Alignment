import argparse

def main():
    parser = argparse.ArgumentParser(
        prog="lna",
        description="Language Neurons Alignment"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # activation
    act = sub.add_parser("activation", help="Collect activations for one language (vLLM)")
    act.add_argument("--model-name", choices=["Mistral", "Llama"], default="Mistral")
    act.add_argument("--model-path", required=True, help="Absolute path to HF/vLLM model dir or identifier")
    act.add_argument("-l", "--lang", required=True)
    act.add_argument("-d", "--data", required=True, help="Path to dataset JSON")
    act.add_argument("-s", "--dataset", default="mgsm")

    # identify
    ident = sub.add_parser("identify", help="Build masks from (full - prefix) activations and count appear histogram")
    ident.add_argument("--model-name", choices=["Mistral", "Llama"], default="Mistral")
    ident.add_argument("--model-path", required=True)
    ident.add_argument("-r", "--top-rate", type=float, default=0.10)
    ident.add_argument("-l", "--xlambda", type=float, default=0.00)
    ident.add_argument("-s", "--dataset", default="mgsm")
    ident.add_argument("-b", "--bar", type=float, default=0.95, help="activation_bar_ratio")

    # autosweep
    sweep = sub.add_parser("autosweep", help="Search max lambda s.t. appear[10]==0 (binary search)")
    sweep.add_argument("--model-name", choices=["Mistral", "Llama"], default="Mistral")
    sweep.add_argument("--model-path", required=True)
    sweep.add_argument("-r", "--top-rate", type=float, default=0.10)
    sweep.add_argument("-s", "--dataset", default="mgsm")
    sweep.add_argument("--lo", type=float, default=0.0)      
    sweep.add_argument("--hi", type=float, default=0.2)      
    sweep.add_argument("--eps", type=float, default=1e-3)
    sweep.add_argument("-b", "--bar", type=float, default=0.95)

    # ppl
    ppl = sub.add_parser("ppl", help="Compute PPL grid and plot heatmaps")
    ppl.add_argument("--model-name", choices=["Mistral", "Llama"], default="Mistral")
    ppl.add_argument("--model-path", required=True)
    ppl.add_argument("-r", "--top-rate", type=str, default="0.10")
    ppl.add_argument("-l", "--xlambda", type=str, default="0.00")
    ppl.add_argument("-s", "--dataset", default="mgsm")
    ppl.add_argument("-d", "--data", required=True, help="Path to dataset JSON (same as activation)")

    args = parser.parse_args()

    if args.cmd == "activation":
        from .activation import collect_activation
        collect_activation(
            model_name=args.model_name,
            model_path=args.model_path,
            lang=args.lang,
            dataset_path=args.data,
            dataset=args.dataset,
        )

    elif args.cmd == "identify":
        from .pipeline import run_identify_and_count
        path, appear = run_identify_and_count(
            model_name=args.model_name,
            model_path=args.model_path,
            dataset=args.dataset,
            toprate=args.top_rate,
            lam=args.xlambda,
            activation_bar_ratio=args.bar,
        )
        print(f"Saved mask at {path}")
        print(f"Appear histogram: {appear}")

    elif args.cmd == "autosweep":
        from .autosweep import autosweep_lambda
        best, appear = autosweep_lambda(
            model_name=args.model_name,
            model_path=args.model_path,
            dataset=args.dataset,
            toprate=args.top_rate,
            lo_init=args.lo,
            hi_init=args.hi,
            eps=args.eps,
            activation_bar_ratio=args.bar
        )
        print(f"Best lambda (appear[10]==0): {best:.6f}")
        print(f"Appear histogram: {appear}")

    elif args.cmd == "ppl":
        from .ppl import run_grid_ppl
        run_grid_ppl(
            model_name=args.model_name,
            model_path=args.model_path,
            dataset=args.dataset,
            toprate=args.top_rate,
            xlambda=args.xlambda,
            data_path=args.data,
        )


if __name__ == "__main__":
    main()
