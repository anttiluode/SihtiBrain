import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { spawnSync } from 'node:child_process';
const require = createRequire(import.meta.url);
const { ConnectionMemory, split } = require('../demo/memory.js');

const python = spawnSync('python', ['-c', `
import json
from aibrain.benchmark import episode
from aibrain.memory import ConnectionMemory
x,y=episode(41,16)
m=ConnectionMemory()
import numpy as np
for k,label in zip(x,y):m.write(k,np.eye(8)[label])
print(json.dumps({'keys':x.tolist(),'labels':y.tolist(),'weights':m.weights.tolist(),'echoes':[m.read(k).tolist() for k in x]}))
`], {encoding:'utf8'});
assert.equal(python.status, 0, python.stderr);
const fixture = JSON.parse(python.stdout);
const m = new ConnectionMemory(64, 8);
fixture.keys.forEach((x,i) => m.write(x, Array.from({length:8}, (_,j) => Number(j===fixture.labels[i]))));
const near = (a,b) => assert.ok(Math.abs(a-b)<1e-10, `${a} != ${b}`);
fixture.keys.forEach((x,i) => {
  const {core,residue} = split(x);
  x.forEach((value,j) => near(core+residue[j],value));
  m.read(x).forEach((value,j) => near(value,fixture.echoes[i][j]));
});
fixture.weights.flat().forEach((value,i) => near(m.weights[i],value));
const before = Array.from(m.weights);
m.clearActivity();
assert.deepEqual(Array.from(m.weights), before);
assert.ok(m.activity.every(x=>x===0));
m.erase();
assert.ok(m.read(fixture.keys[0]).every(x=>x===0));
assert.throws(()=>m.write(Array(64).fill(NaN), Array(8).fill(0)), /finite/);
assert.throws(()=>m.read(Array(63).fill(0)), /size/);
assert.throws(()=>m.write(fixture.keys[0], Array(8).fill(1)), /probability/);
assert.throws(()=>new ConnectionMemory(0,8), /positive/);
console.log('JS/Python parity, reconstruction, resets, erasure and validation passed.');

assert.throws(()=>m.write(fixture.keys[0], [1.000009,0,0,0,0,0,0,0]), /probability/);
for(let i=0;i<64;i++) m.write(fixture.keys[0],[1.000000009,0,0,0,0,0,0,0]);
assert.ok(m.weights.every(x=>x>=0 && x<=1));
assert.ok(m.read(Array(64).fill(1e308)).every(x=>x===0));
assert.ok(m.read(Array.from({length:64},(_,i)=>i%2?1e308:-1e308)).every(Number.isFinite));
console.log('Large finite queries and strict bounded-value validation passed.');
