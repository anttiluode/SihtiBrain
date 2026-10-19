"""Reproduce the Gate 4 tuft-to-basal learning results and six-panel chart."""
import argparse
import json
from pathlib import Path
import numpy as np
from apical_credit import (Gate4, basal_state, basal_features, benchmark,
                           train, tuft_to_soma)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("runs/apical_credit"))
    p.add_argument("--epochs", type=int, default=35)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    results = benchmark(seeds=(0, 1, 2, 3), epochs=args.epochs)
    noise = {}
    for gain in (3., .05):
        for sigma in (0., .03, .1):
            key = f"gain_{gain}_noise_{sigma}"
            noise[key] = [train(s, epochs=args.epochs, context_gain=gain,
                                noise=sigma)[1] for s in range(4)]
    initial = Gate4.new(0)
    first_window = {}
    for label, target, x in (
        ("cluster", 1., [1., 1., 0., 0.]),
        ("split", 0., [1., 0., 1., 0.])
    ):
        r = initial.window(x, target)
        first_window[label] = dict(
            first_window_soma=float(basal_state(x)[0]),
            basal_features=basal_features(x).tolist(),
            local_eligibility=r['eligibility'].tolist(),
            tuft_current=r['tuft_current'],
            tuft_to_soma_voltage_effect=r['passive_soma_effect'],
            ping_free=r['ping_free'], ping_later=r['ping_later'],
            calcium_log_ratio=r['local_delta'].tolist(),
            synaptic_update_before_learning=r['updates'].tolist())
    summary = {
        "protocol": {
            "n_compartments": 9,
            "branch_feature": "sigmoid(10*(root_voltage-2.0)) fixed",
            "teacher_location": "tuft node 8 only; signed supervised context",
            "passive_tuft_to_soma_transfer": tuft_to_soma(),
            "context_gain": 3.,
            "ping_later": "1+0.85*tanh(4*3*transfer*(target-prediction))",
            "plastic_sites": "two basal branch readout weights and somatic bias",
            "local_rule": "eta*eligibility*log(Ca_later/Ca_free)",
            "train_n": 72, "test_n": 192, "seeds": 4,
            "epochs": args.epochs, "lr": .11, "amplitude_jitter": .08,
            "limitations": [
                "later context explicitly conveys target-minus-prediction",
                "nonlinear branch sieve is hand-selected, not learned",
                "pings are two timed probes, not a biophysical spike generator",
                "delay cutoff models finite eligibility, not biophysical trace decay",
                "ordinary direct-gradient baseline also solves this task"]},
        "first_window": first_window,
        "conditions": results,
        "noise_sweep": noise
    }
    (args.output/"metrics.json").write_text(json.dumps(summary, indent=2)+"\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3,2,figsize=(13,12),layout="constrained")
    fig.suptitle("SihtiBrainV2 Gate 4: tuft context → shaped ping → basal read/write",
                 fontsize=15)

    ax = axes[0,0]
    ax.bar([0,1],[first_window[k]["first_window_soma"] for k in ("cluster","split")])
    ax.set(xticks=[0,1],xticklabels=["cluster","split"],
           title="A. First window: identical somatic voltage", ylabel="soma voltage",
           ylim=(0,1.8))

    ax = axes[0,1]
    xx=np.arange(2)
    ax.bar(xx-.18,first_window["cluster"]["basal_features"],width=.36,label="cluster")
    ax.bar(xx+.18,first_window["split"]["basal_features"],width=.36,label="split")
    ax.set(xticks=xx,xticklabels=["left basal branch","right basal branch"],
           title="B. Nonlinear branch eligibility (fixed sieve)",
           ylabel="local activation")
    ax.legend(fontsize=8)

    conditions=["tuft_ping","no_ping","no_context","shuffled_context",
                "no_basal_trace","long_delay","linear_basal","no_trunk",
                "direct_gradient"]
    names=["Tuft + ping","No ping","No context","Shuffled target",
           "Trace erased","Expired trace","Linear basal","No trunk","Direct SGD"]
    means=[np.mean([r["accuracy"] for r in results[k]]) for k in conditions]
    minima=[min(r["accuracy"] for r in results[k]) for k in conditions]
    maxima=[max(r["accuracy"] for r in results[k]) for k in conditions]
    ix=np.arange(len(conditions))
    ax=axes[1,0]
    ax.bar(ix,means)
    ax.errorbar(ix,means,yerr=[np.array(means)-minima,
                              np.array(maxima)-means],fmt="none",
                ecolor="black",capsize=3)
    ax.axhline(2/3,ls=":",color="gray",label="majority baseline")
    ax.set(ylim=(0,1.07),xticks=ix,xticklabels=names,
           ylabel="fraction correct",title="C. Held-out accuracy (4 seeds)")
    ax.tick_params(axis="x",rotation=42,labelsize=8)
    ax.legend(fontsize=7)

    ax=axes[1,1]
    for key,label in (("tuft_ping","Tuft + ping"),
                      ("shuffled_context","Shuffled context"),
                      ("direct_gradient","Direct SGD")):
        h=[entry["history"] for entry in results[key]]
        times=[row["epoch"] for row in h[0]]
        avg=[np.mean([run[i]["accuracy"] for run in h])
             for i in range(len(times))]
        ax.plot(times,avg,marker="o",label=label)
    ax.set(xlabel="epoch",ylabel="held-out accuracy",ylim=(0,1.03),
           title="D. Learning across training windows")
    ax.legend(fontsize=8)

    ax=axes[2,0]
    for gain,label in ((3.,"strong contextual channel"),
                       (.05,"weak contextual channel")):
        sigmas=(0.,.03,.1)
        y=[np.mean([r["accuracy"] for r in noise[f"gain_{gain}_noise_{s}"]])
           for s in sigmas]
        ax.plot(np.array(sigmas)*100,y,marker="o",label=label)
    ax.set(xlabel="calcium measurement noise (%)",
           ylabel="held-out accuracy",ylim=(.65,1.04),
           title="E. Calcium read noise vs. teacher strength")
    ax.legend(fontsize=8)

    ax=axes[2,1]
    for label,sites in (("cluster",[0,1]),("split",[2,3])):
        r=first_window[label]
        ax.bar(sites,[r["ping_free"],r["ping_later"]])
    ax.set(xticks=range(4),xticklabels=["early C","later C","early S","later S"],
           ylabel="dimensionless pulse amplitude",
           title="F. Same early readout, different later tuft signals")

    fig.savefig(args.output/"apical_credit.png",dpi=170)
    plt.close(fig)
    for key in conditions:
        print(f"{key:19s} accuracy = " +
              str([round(r["accuracy"],3) for r in results[key]]))
    print("Saved",args.output)


if __name__=="__main__":
    main()
