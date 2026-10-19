"""Interactive AIbrainV2 lab. Standard-library Tkinter + numpy only.

Run: python gui.py. Four blue squares make a cue; choose one of eight values.
All memories receive identical writes. Query shows the stationary graph's
echo alongside a direct fast-weight read and compact indexed lookup.
"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from operator_memory import (ConnectionMemory, IndexedMemory, CUE_BITS,
                             VALUE_BITS, dataset, swapped_queries)

PALETTE=("#0b1220","#18263c","#73c5f6","#b5d6ef","#e6a96a")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AIbrainV2 | Memory as a response")
        self.geometry("980x680")
        self.configure(bg=PALETTE[0])
        self.graph=ConnectionMemory.empty(True)
        self.fast=ConnectionMemory.empty(False)
        self.indexed=IndexedMemory()
        self.cue=np.zeros(CUE_BITS)
        self.value=tk.IntVar(value=0)
        self.saved=[]
        self.status=tk.StringVar(value="Select exactly four cue squares.")
        self._draw()

    def _draw(self):
        style=ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame",background=PALETTE[0])
        style.configure("TLabel",background=PALETTE[0],foreground=PALETTE[3],
                        font=("Segoe UI",11))
        style.configure("Header.TLabel",font=("Segoe UI",16,"bold"))
        wrapper=ttk.Frame(self,padding=15)
        wrapper.pack(fill="both",expand=True)
        ttk.Label(wrapper,text="AIbrainV2 — queryable connection geometry",
                  style="Header.TLabel").pack(anchor="w")
        ttk.Label(wrapper,text="Same 512-byte connection matrix, same writes. Different read geometry. "
                  "Indexed control uses 3 bytes per stored association.").pack(anchor="w",pady=7)
        row=ttk.Frame(wrapper);row.pack(fill="both",expand=True)
        left=ttk.Frame(row);left.pack(side="left",anchor="n",padx=(0,25))
        ttk.Label(left,text="Cue: toggle exactly four bits").pack(anchor="w",pady=(15,7))
        grid=ttk.Frame(left);grid.pack()
        self.bits=[]
        for i in range(CUE_BITS):
            b=tk.Button(grid,text=str(i+1),width=6,height=2,
                 font=("Segoe UI",10),relief="flat",
                 command=lambda i=i:self.flip(i))
            b.grid(row=i//4,column=i%4,padx=3,pady=3)
            self.bits.append(b)
        self.paint_bits()
        ttk.Label(left,text="Binding value").pack(anchor="w",pady=(18,6))
        radios=ttk.Frame(left);radios.pack(anchor="w")
        for j in range(VALUE_BITS):
            rb=ttk.Radiobutton(radios,text=f"{j}",variable=self.value,value=j)
            rb.grid(row=j//4,column=j%4,padx=13,pady=5)
        commands=ttk.Frame(left);commands.pack(anchor="w",pady=18)
        actions=[("Write association",self.write),("Query",self.query),
                 ("Swap cue bit",self.corrupt),("Reset activity",self.activity_reset),
                 ("Erase all connections",self.erase),("Load 8 examples",self.populate)]
        for i,(label,func) in enumerate(actions):
            ttk.Button(commands,text=label,command=func).grid(
                row=i//2,column=i%2,padx=3,pady=5,sticky="ew")
        right=ttk.Frame(row);right.pack(side="left",fill="both",expand=True)
        self.bars={}
        for name in ("operator","fast_weight","indexed"):
            ttk.Label(right,text={"operator":"Passive graph echo",
                "fast_weight":"Direct fast weight","indexed":"Indexed nearest-key"}[name]).pack(
                    anchor="w",pady=(10,0))
            bar=tk.Canvas(right,height=100,width=515,bg=PALETTE[1],
                          highlightthickness=0)
            bar.pack(anchor="w",pady=(4,10))
            self.bars[name]=bar
        ttk.Label(wrapper,textvariable=self.status,wraplength=940).pack(
            anchor="w",pady=8)
        ttk.Label(wrapper,text="Note: these are stored random associations, not learned semantics. "
                  "A fresh query clears transient voltages automatically. "
                  "Writing another label adds conductance; it does not erase old edges.",
                  wraplength=900).pack(anchor="w")

    def paint_bits(self):
        for i,b in enumerate(self.bits):
            active=bool(self.cue[i])
            b.configure(bg=PALETTE[2] if active else PALETTE[1],
                        fg=PALETTE[0] if active else "white",
                        activebackground=PALETTE[2])

    def flip(self,i):
        self.cue[i]=1-self.cue[i]
        self.paint_bits()

    def require_key(self):
        if self.cue.sum()!=4:
            messagebox.showwarning("Four bits required","Select exactly four active cue bits.")
            return False
        return True

    def write(self):
        if not self.require_key():
            return
        for model in (self.graph,self.fast,self.indexed):
            model.write(self.cue,int(self.value.get()))
        self.saved.append((self.cue.copy(),int(self.value.get())))
        self.status.set(f"Written value {self.value.get()}. Stored graph/fast weights: 512 bytes each; "
                        f"indexed: {self.indexed.storage_bytes} bytes.")
        self.query()

    def query(self):
        if not self.require_key():
            return
        q=self.cue.copy()
        graph=self.graph.read(q).reshape(-1)
        fast=self.fast.read(q).reshape(-1)
        ix=int(self.indexed.predict(q)[0])
        for name,scores in (("operator",graph),("fast_weight",fast),
                            ("indexed",np.eye(8)[ix])):
            self.show_bars(self.bars[name],scores)
        self.status.set(f"Graph predicts {int(np.argmax(graph))}, direct weights "
                        f"{int(np.argmax(fast))}, indexed {ix}. Indexed store: "
                        f"{self.indexed.storage_bytes}B; graph/fast: 512B each.")

    def show_bars(self,can,scores):
        can.delete("all")
        scores=np.asarray(scores,dtype=float)
        maximum=max(float(scores.max()),1e-12)
        for k,val in enumerate(scores):
            x=12+k*62
            height=60*max(0,float(val))/maximum
            can.create_rectangle(x,68-height,x+40,68,fill=PALETTE[2],
                                 outline="")
            can.create_text(x+20,82,text=str(k),fill=PALETTE[3])
        can.create_text(500,10,text="argmax="+str(int(np.argmax(scores))),
                        anchor="ne",fill=PALETTE[4],font=("Segoe UI",10))

    def corrupt(self):
        if not self.require_key():
            return
        self.cue=swapped_queries(np.asarray([self.cue]),len(self.saved)+117,1)[0]
        self.paint_bits()
        self.query()

    def activity_reset(self):
        for x in (self.graph,self.fast,self.indexed):
            x.reset_activity()
        self.status.set("Transient state cleared; connection memory remains.")

    def erase(self):
        for x in (self.graph,self.fast,self.indexed):
            x.erase_connections()
        self.saved.clear()
        self.status.set("Connection state erased. No associations remain.")
        if self.cue.sum()==4:
            self.query()

    def populate(self):
        self.erase()
        keys,labels=dataset(56,8)
        for key,label in zip(keys,labels):
            for model in (self.graph,self.fast,self.indexed):
                model.write(key,int(label))
        self.saved=list(zip(keys,labels))
        self.cue=keys[0].copy()
        self.value.set(int(labels[0]))
        self.paint_bits()
        self.query()


if __name__=="__main__":
    App().mainloop()
