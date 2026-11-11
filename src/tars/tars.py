"""TARS/CASE as a MuJoCo model: four slabs on a common axle, each with a hinge and a slide.
Assumed: slabs 1.6 m x 0.30 m x 0.22 m, 22.5 kg each (90 kg, the weight of the film's puppet)."""
import math, numpy as np, mujoco
G=9.81; LS=1.6; W=0.30; D=0.22; MS=22.5; RF=0.03; YS=(-0.48,-0.16,0.16,0.48)
def k2_slab():           # radius of gyration about the axle, relative to the spoke length l = LS/2
    l=LS/2; return (LS**2+D**2)/12/l**2
def xml(angles=(0,0,0,0), g=G, slope=0.0, axle_m=4.0, extra_hub=0.0, floor=True):
    gx,gz=g*math.sin(slope),-g*math.cos(slope)
    slabs=[]
    for i,(y,a) in enumerate(zip(YS,angles)):
        slabs.append(f'''
      <body name="arm{i}" pos="0 {y} 0">
        <inertial pos="0 0 0" mass="0.1" diaginertia="1e-3 1e-3 1e-3"/>
        <joint name="h{i}" type="hinge" axis="0 1 0" ref="0"/>
        <body name="slab{i}">
          <joint name="s{i}" type="slide" axis="0 0 1" range="-0.15 0.15"/>
          <geom name="body{i}" type="box" size="{D/2} {W/2-0.005} {LS/2-RF}" mass="{MS}" rgba=".72 .72 .70 1" contype="0" conaffinity="0"/>
          <geom name="footA{i}" type="sphere" pos="0 0 {-(LS/2-RF)}" size="{RF}" mass="0.01" friction="1.2 0.005 0.0001"/>
          <geom name="footB{i}" type="sphere" pos="0 0 {LS/2-RF}" size="{RF}" mass="0.01" friction="1.2 0.005 0.0001"/>
        </body>
      </body>''')
    acts=''.join(f'<position name="ph{i}" joint="h{i}" kp="40000" kv="2000"/><position name="ps{i}" joint="s{i}" kp="60000" kv="3000"/>' for i in range(4))
    return f'''<mujoco>
  <option timestep="2e-4" gravity="{gx:.6f} 0 {gz:.6f}" integrator="implicitfast"/>
  <default><geom solref="0.002 1" solimp="0.99 0.999 0.0001"/></default>
  <worldbody>
    {'<geom name="floor" type="plane" size="0 0 1" friction="1.2 0.005 0.0001"/>' if floor else ''}
    <body name="axle" pos="0 0 1.0">
      <joint name="x" type="slide" axis="1 0 0"/><joint name="z" type="slide" axis="0 0 1"/><joint name="pitch" type="hinge" axis="0 1 0"/>
      <geom type="cylinder" fromto="0 -0.6 0 0 0.6 0" size="0.04" mass="{axle_m}" contype="0" conaffinity="0" rgba=".3 .3 .3 1"/>
      {f'<geom type="sphere" size="0.05" mass="{extra_hub}" contype="0" conaffinity="0"/>' if extra_hub else ''}
      {''.join(slabs)}
    </body>
  </worldbody>
  <actuator>{acts}</actuator>
</mujoco>'''
def ratio(n,k2): return (k2+math.cos(2*math.pi/n))/(1+k2)
