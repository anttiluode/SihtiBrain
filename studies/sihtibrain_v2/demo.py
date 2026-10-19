"""Reproduce SihtiBrainV2 metrics and visualizations on a CPU."""
import argparse
import json
from pathlib import Path
import numpy as np
from sihtibrain_v2 import (experiment, encode, probe, reset_and_read,
                          passive_trajectory, context_from_neighbor,
                          shaped_ping, live_loop)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=Path("runs/first_echo"))
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    m = experiment()
    context = context_from_neighbor([1,1,0,0])
    c, s = encode([1,1,0,0]), encode([1,0,1,0])
    lc = live_loop(c,apical=context,writer_eta=.3)
    ls = live_loop(s,apical=context,writer_eta=.3)
    mc = live_loop(c,apical=context,writer_eta=0.)
    ms = live_loop(s,apical=context,writer_eta=0.)
    m["independent_neighbor_context"] = context.tolist()
    m["live_loop"] = {
        "dt": .025, "start_step": 35, "branch_return_delay_steps": 8,
        "cluster_final_slow_g": lc["slow_g"][-1].tolist(),
        "split_final_slow_g": ls["slow_g"][-1].tolist(),
        "persistent_echo_difference_after_reset":
            reset_and_read(lc["slow_g"][-1]) - reset_and_read(ls["slow_g"][-1]),
        "frozen_writer_echo_difference_after_reset":
            reset_and_read(mc["slow_g"][-1]) - reset_and_read(ms["slow_g"][-1])
    }
    (args.output / "metrics.json").write_text(json.dumps(m,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(m,indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cp, ce = passive_trajectory([1,1,0,0])
    sp, se = passive_trajectory([1,0,1,0])
    fig, a = plt.subplots(3,2,figsize=(13,11),layout="constrained")
    fig.suptitle("SihtiBrainV2 | hidden voltage -> contextual echo -> changed wire",fontsize=15)
    t = np.arange(cp.shape[0])*.025
    a[0,0].plot(t,cp[:,0],label="cluster soma")
    a[0,0].plot(t,sp[:,0],ls="--",label="split soma")
    a[0,0].plot(t,cp[:,1],label="cluster left root")
    a[0,0].plot(t,sp[:,1],ls=":",label="split left root")
    a[0,0].set(title="1. Basal encoding: hidden branch arrangement",xlabel="time",ylabel="voltage")
    a[0,0].legend(fontsize=8)

    labels=["fixed W echo","contextual echo","reset read"]
    keys=["fixed_operator_echo","gated_echo","post_reset_echo"]
    clustered=[m["cluster"][k] for k in keys]
    splitted=[m["split"][k] for k in keys]
    x=np.arange(3)
    a[0,1].bar(x-.18,clustered,width=.36,label="cluster")
    a[0,1].bar(x+.18,splitted,width=.36,label="split")
    a[0,1].set_xticks(x,labels)
    a[0,1].tick_params(axis="x",labelsize=8)
    a[0,1].set(title="2. Identical initial soma; distinct echo",ylabel="ping response")
    a[0,1].legend(fontsize=8)

    a[1,0].plot(t,ce,label="cluster")
    a[1,0].plot(t,se,ls="--",label="split")
    a[1,0].set(title="3. Passive electrical energy descends",xlabel="time",ylabel="E")
    a[1,0].legend(fontsize=8)

    for en,label in ((c,"cluster"),(s,"split")):
        levels=np.linspace(0,1,51)
        pulse=shaped_ping(en.voltage[0])
        responses=[probe(en,apical=(level,0),ping=pulse).echo for level in levels]
        a[1,1].plot(levels,responses,label=label)
    a[1,1].set(title="4. Apical strength controls echo",xlabel="left apical input",ylabel="probe echo")
    a[1,1].legend(fontsize=8)

    tx=np.arange(len(lc["voltage"]))*.025
    a[2,0].plot(tx,lc["voltage"][:,0],label="cluster soma")
    a[2,0].plot(tx,ls["voltage"][:,0],ls="--",label="split soma")
    a[2,0].plot(tx,mc["voltage"][:,0],ls=":",label="cluster, frozen writer",alpha=.7)
    a[2,0].axvline((35+8)*.025,ls=":",color="gray",label="return gate arrives")
    a[2,0].set(title="5. Causal return-path timing",xlabel="time",ylabel="soma voltage")
    a[2,0].legend(fontsize=8)

    a[2,1].plot(tx,lc["slow_g"][:,0],label="cluster stored g_left")
    a[2,1].plot(tx,ls["slow_g"][:,0],ls="--",label="split stored g_left")
    a[2,1].plot(tx,lc["fast_g"][:,0],ls=":",label="cluster transient g_left",alpha=.7)
    a[2,1].set(title="6. Unsupervised local cable write",xlabel="time",ylabel="left conductance")
    a[2,1].legend(fontsize=8)
    fig.savefig(args.output/"sihtibrain_v2.png",dpi=165)
    plt.close(fig)


if __name__=="__main__":
    main()
