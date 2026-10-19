/* Same float32 connection writer and double-precision echo as aibrain/memory.py.
   Classic script: opens offline from file://; also exports for Node parity tests. */
(function (root) {
  'use strict';
  function vector(x, size) {
    const a = Array.from(x);
    if (!a.length || (size !== undefined && a.length !== size)) throw Error('wrong vector size');
    if (!a.every(Number.isFinite)) throw Error('vectors must be finite');
    return a;
  }
  function split(x) {
    const a = vector(x);
    const scale = Math.max(...a.map(Math.abs));
    const core = scale ? a.reduce((s,v)=>s+v/scale/a.length,0)*scale : 0;
    const residue = a.map(v=>v-core);
    if(!residue.every(Number.isFinite))throw Error('split residue exceeds finite numeric range');
    return {core, residue};
  }
  function cue(x) {
    const a = vector(x), scale = Math.max(...a.map(Math.abs));
    if(!scale)return a.map(()=>0);
    const r = split(a.map(v=>v/scale)).residue;
    const norm = Math.sqrt(r.reduce((s,v)=>s+v*v,0));
    return norm>1e-12 ? r.map(v=>v/norm) : r.map(()=>0);
  }
  class ConnectionMemory {
    constructor(dim=64, values=8) {
      if (!Number.isInteger(dim)||!Number.isInteger(values)||dim<1||values<1) throw Error('positive integer sizes required');
      this.dim=dim; this.values=values; this.eta=.5; this.leak=1;
      this.weights=new Float32Array(dim*values);
      this.activity=new Float64Array(values);
    }
    write(x,v) {
      x=vector(x,this.dim);v=vector(v,this.values);
      if(x.some(a=>a<0||a>1))throw Error('write key must be in [0,1]');
      if(v.some(a=>a<0||a>1+1e-8))throw Error('value must be a probability vector');
      const total=v.reduce((s,a)=>s+a,0);
      if(Math.abs(total-1)>1e-8)throw Error('value must be a probability vector');
      v=v.map(a=>a/total);
      for(let i=0;i<this.dim;i++)for(let j=0;j<this.values;j++) {
        const index=i*this.values+j, rate=this.eta*x[i];
        this.weights[index]=(1-rate)*this.weights[index]+rate*v[j];
      }
    }
    read(x) {
      const q=cue(vector(x,this.dim));
      for(let j=0;j<this.values;j++) {
        let drive=0,degree=0;
        for(let i=0;i<this.dim;i++) {const w=this.weights[i*this.values+j];drive+=w*q[i];degree+=w;}
        this.activity[j]=drive/(this.leak+degree);
      }
      return Array.from(this.activity);
    }
    clearActivity(){this.activity.fill(0);}
    erase(){this.weights.fill(0);this.clearActivity();}
  }
  const api={ConnectionMemory,split,cue};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  else root.AIBrain=api;
})(typeof globalThis!=='undefined'?globalThis:this);
