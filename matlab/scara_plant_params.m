function p = scara_plant_params()
%SCARA_PLANT_PARAMS  SCARA-teach-4RPRR 参数（本仓库副本，来源：既有 SCARA 模型 scara_params.m）
%
% 说明：为了本项目的 Simulink 阶段自成闭环，这里保留一份与既有 SCARA 模型
% 逐位一致的参数；`matlab/verify_scara_vs_previous.m` 已证明两者等价（M/G/C 偏差 <1e-12）。
p.a1 = 0.225;  p.a2 = 0.225;  p.d1 = 0.300;  p.d4 = 0.0;  p.stroke = 0.200;
p.q1_lim = deg2rad([-130 130]);
p.q2_lim = deg2rad([-145 145]);
p.q4_lim = deg2rad([-360 360]);
p.q3_lim = [0 p.stroke];
p.m1 = 3.0;  p.lc1 = 0.110;
p.m2 = 1.5;  p.lc2 = 0.105;
p.m3 = 2.0;  p.mload = 1.0;  p.mt = p.m3 + p.mload;
p.I1 = p.m1*p.a1^2/12;
p.I2 = p.m2*p.a2^2/12;
p.I4 = 2.0e-3;
p.g  = 9.81;
p.Fv = [0.50 0.40 20.0 0.02];
p.Fc = [0.30 0.25 10.0 0.01];
p.pi1 = p.I1 + p.m1*p.lc1^2 + (p.m2+p.mt)*p.a1^2;
p.pi2 = p.I2 + p.m2*p.lc2^2 + p.mt*p.a2^2;
p.pi3 = p.m2*p.a1*p.lc2 + p.mt*p.a1*p.a2;
p.piZ = p.mt;
p.pi4 = p.I4;
end
